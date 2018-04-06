# Import libraries
import sys
import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# To load the saved model
from keras.models import load_model

# Suppress Tensorflow error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                 xy_window=(32, 32), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    
    # Initialize a list to append window positions to
    window_list = []
    
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(img, windows, clf):
    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (44, 44)) 
        #4) Extract features for that window using single_img_features()
        test_img  = test_img.reshape(1, 1, 44, 44).astype('float32')
        test_img = test_img/255.0
        prediction = clf.predict(test_img, batch_size=1, verbose=0, steps=None)
        if prediction[0][0] > 0.7:
            on_windows.append(window)
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def predict_position(image, model):
    image_color = plt.imread(image)
    image = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

    draw_image = np.copy(image_color)

    windows = slide_window(image, x_start_stop=[0, image.shape[1]], y_start_stop=[0, image.shape[0]], 
                        xy_window=(44, 44), xy_overlap=(0.8, 0.8))                  

    # Find probable windows
    hot_windows = search_windows(image, windows, model)

    heat = np.zeros_like(image_color[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
    heat = heat.astype('uint8')

    # Find the contours 
    im2, contours, hierarchy = cv2.findContours(heat,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    image = np.copy(draw_image)

    # Find the biggest area (assumed to be the phone)
    c = max(contours, key = cv2.contourArea)

    # Remove false positives (black strips) by checking the area of the contour
    area = cv2.contourArea(c)
    
    if area > 10000:
        desc_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cont in desc_contours:
            if cv2.contourArea(cont) < 10000:
                c = cont
                break

    x,y,w,h = cv2.boundingRect(c)

    # find the bounding box (rectangle)
    bbox = [(x,y),(x+w,y+h)]

    # draw the phone contour
    title_p1 = (bbox[0][0],bbox[0][1])
    title_p2 = (bbox[0][0]+70,bbox[0][1]-30)
    cv2.rectangle(image,bbox[0],bbox[1],(0,0,255),2)
    cv2.rectangle(image,title_p1, title_p2,(255,51,157),-1)
    cv2.putText(image, "phone", (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    #show the image
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    # predict position of the phone
    return [round((float(bbox[0][0]+bbox[1][0])/2)/draw_image.shape[1],4), round((float(bbox[0][1]+bbox[1][1])/2)/draw_image.shape[0],4)]


def main():
    fn = sys.argv[1]
    # Load CNN model from disk
    model = load_model('model.h5')
    #print "Model loaded from disk!"
    pos = predict_position(fn, model)
    print pos[0], pos[1]

if __name__ == "__main__":
    main()
