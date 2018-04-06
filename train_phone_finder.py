# Import libraries
import sys
import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

# Get the prediction function to find accuracy
from find_phone import predict_position

# Import Keras modules
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')

# Supress Tensorflow error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def crop_phone(img, position, num_samples=50):
    height, width = img.shape
    phone_images = []
    
    phone_cropped = img[int(position[1]*height)-22 : int(position[1]*height)+22,
                        int(position[0]*width)-22 : int(position[0]*width)+22]
    
    for i in range(num_samples):
        random.seed(i)
        p1 = random.random()
        if p1 >= 0.25:
            opt = [1,2,3,4]
            t = random.choice(opt)
            phone_images.append(rotate_image(phone_cropped,t))
        else:
            phone_images.append(phone_cropped)
    return phone_images


def rotate_image(image,times=1):
    return np.rot90(image,times)


def crop_floor(img, position, num_samples=50):
    height, width = img.shape
    floor_images = []
    
    # left margin
    if position[0] < 0.3:
        for i in range(num_samples):
            random.seed(i)
            row = random.uniform(0.1,0.8)
            col = random.uniform(0.4,0.8)
            floor_cropped = img[int(row*height): int(row*height)+44,int(col*width): int(col*width)+44]
            floor_images.append(floor_cropped)
    
    # right margin
    elif position[0] > 0.7:
        for i in range(num_samples):
            random.seed(i+1)
            row = random.uniform(0.1,0.8)
            col = random.uniform(0.1,0.5)
            floor_cropped = img[int(row*height): int(row*height)+44,int(col*width): int(col*width)+44]
            floor_images.append(floor_cropped)

    # top margin
    elif position[1] < 0.3:
        for i in range(num_samples):
            random.seed(i+2)
            row = random.uniform(0.5,0.8)
            col = random.uniform(0.1,0.8)
            floor_cropped = img[int(row*height): int(row*height)+44,int(col*width): int(col*width)+44]
            floor_images.append(floor_cropped)
            
    # bottom margin
    elif position[1] > 0.7:
        for i in range(num_samples):
            random.seed(i+3)
            row = random.uniform(0.1,0.5)
            col = random.uniform(0.1,0.8)
            floor_cropped = img[int(row*height): int(row*height)+44,int(col*width): int(col*width)+44]
            floor_images.append(floor_cropped)
            
    else:
        for i in range(num_samples):
            random.seed(i)
            p1 = random.random()
            
            if p1 >= 0.5:
                row = random.uniform(0.1,position[1]-0.2)
            else:
                row = random.uniform(position[1]+0.1,0.8)
            
            random.seed(i+10)
            p2 = random.random()

            if p2 >= 0.5:
                col = random.uniform(0.1,position[0]-0.2)
            else:
                col = random.uniform(position[1]+0.1,0.8)
                
            floor_cropped = img[int(row*height): int(row*height)+44,int(col*width): int(col*width)+44]
            floor_images.append(floor_cropped)
    return floor_images


def read_images(images, positions):
    X_phone, X_floor = [], []

    for i in tqdm(range(len(images))):
        img = plt.imread(images[i])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        phone_img = crop_phone(img, positions[i])

        floor_img = crop_floor(img, positions[i])

        X_phone.extend(phone_img)
        X_floor.extend(floor_img)
    X_phone = np.array(X_phone)
    X_floor = np.array(X_floor)
    
    return X_phone, X_floor


def prepare_data(X_phone, X_floor):
    X_phone = np.array(X_phone)
    X_floor = np.array(X_floor)
    
    X = np.vstack((X_phone, X_floor))
    y = np.hstack((np.ones(len(X_phone)), np.zeros(len(X_floor))))
    
    X, y = shuffle(X, y, random_state = 42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape data to match CNN input format
    X_train = X_train.reshape(X_train.shape[0], 1, 44, 44).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 44, 44).astype('float32')
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    return X_train, X_test, y_train, y_test


# Convolutional Neural Network design
def create_model(X_train, X_test, y_train, y_test):
    # to get reproducible results
    np.random.seed(0)
    tf.set_random_seed(0)
    
    # create model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(1, 44, 44), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    # Compile model
    sgd = optimizers.SGD(lr=0.1, decay=1e-2)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    # Earlystopping
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=0, mode='auto')
    callbacks_list = [earlystop]
    # Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, 
                    callbacks = callbacks_list, batch_size=128)
    # save model in HDF5 format
    model.save("model.h5")
    print "Saved model to disk!"
    plot_model_history(history)
    return model


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig('model_history.png')


def evaluate_model(model, X_test, y_test):
    # Final evaluation of the model on cropped images
    scores = model.evaluate(X_test, y_test, verbose=0)
    print "Model test accuracy on cropped images: {}".format(scores[1]*100)


def test_accuracy(images, positions, model):
    count = 0
    for i in tqdm(range(len(images))):
        pos_predicted = predict_position(images[i], model)
        pos_actual = positions[i]
        # Find the Euclidean distance between the 2 points
        dist = ((pos_actual[0]-pos_predicted[0])**2 + (pos_actual[1]-pos_predicted[1])**2)**0.5
        if dist < 0.05:
            count += 1
    print "Accuracy of the model is {}.".format(float(count)/len(images))


def main():
    path = sys.argv[1]
    txt_file = glob.glob(os.path.join(path, '*.txt'))
    with open(str(txt_file[0])) as f:
        lines = f.read().splitlines()

    images = []
    positions = []
    for line in lines:
        img, x, y = line.split()
        pos = (float(x),float(y))
        images.append(path+'/'+img)
        positions.append(pos)
    
    X_phone, X_floor = read_images(images, positions)
    print "Completed reading images from the dataset directory!"
    
    # Generate training and testing data
    X_train, X_test, y_train, y_test =  prepare_data(X_phone, X_floor)
    
    # Train model
    model = create_model(X_train, X_test, y_train, y_test)
    
    # # Load model from disk
    # model = load_model('model.h5')
    # print "Model loaded from disk!"

    # Test model
    evaluate_model(model, X_test, y_test)
    
    print "Calculating accuracy of position prediction for all images in the dataset..."
    # Find accuracy of predicted positions
    test_accuracy(images, positions, model)


if __name__ == "__main__":
    main()
