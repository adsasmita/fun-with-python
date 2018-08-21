import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Lambda, Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K

import itertools
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from numpy.random import random, permutation, randn, normal, uniform, choice

import matplotlib.pyplot as plt
np.set_printoptions(precision=4, linewidth=100)



def GRID_lenet(model_path,
			   model_weights_fname,
			   img_width = 50,
			   img_height = 50):

	if not os.path.exists(model_path): os.mkdir(model_path)
	# Matching tensor input dimensions
	if K.image_data_format() == 'channels_first':
	    input_shape = (3, img_width, img_height)
	else:
	    input_shape = (img_width, img_height, 3)

	# CUSTOM CNN ARCHITECTURE
	# Modified LeNet architecture in Keras==2.0.8, weights_name = 'gridlenet50_v1.h5'
	lenet = Sequential()
	# First Set of Conv->BatchNorm->ReLU->MaxPool2x2
	lenet.add(Lambda(lambda x: x * 1.0 / 255, input_shape=input_shape))
	lenet.add(Conv2D(64, (5, 5), padding='same'))
	lenet.add(BatchNormalization())
	lenet.add(Activation('relu'))
	lenet.add(MaxPooling2D(pool_size=(2, 2)))
	# Second Set of Conv->BatchNorm->ReLU->MaxPool2x2
	lenet.add(Conv2D(64, (5, 5), padding='same'))
	lenet.add(BatchNormalization())
	lenet.add(Activation('relu'))
	lenet.add(MaxPooling2D(pool_size=(2, 2)))
	# Third Set of Conv->BatchNorm->ReLU->MaxPool2x2
	lenet.add(Conv2D(64, (5, 5), padding='same'))
	lenet.add(BatchNormalization())
	lenet.add(Activation('relu'))
	lenet.add(MaxPooling2D(pool_size=(2, 2)))
	# Flatten into FC Layer
	lenet.add(Flatten())
	# First Set of FC->BatchNorm->ReLU->Dropout
	lenet.add(Dense(1024))
	lenet.add(BatchNormalization())
	lenet.add(Activation('relu'))
	lenet.add(Dropout(0.5))
	# Second Set of FC->BatchNorm->ReLU->Dropout
	lenet.add(Dense(1024))
	lenet.add(BatchNormalization())
	lenet.add(Activation('relu'))
	lenet.add(Dropout(0.5))
	# Final Softmax Layer
	lenet.add(Dense(2))
	lenet.add(Activation('softmax'))
	# Compile Model
	lenet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Load Trained Model Weights
	lenet.load_weights(model_path + model_weights_fname)

	return lenet


def create_clf_boxes(probs, clf_thresh=0.50, kernel=250, stride=10, resize=1.0):
    """
    Create clf_boxes containing:
    - x1 / xmin: upper-left x-coordinate
    - y1 / ymin: upper-left y-coordinate
    - x2 / xmax: bottom-right x-coordinate
    - y2 / ymax: bottom-right y-coordinate
    - probs: predict_probas from classifier model 
    """
    (det_y, det_x) = np.where((probs>clf_thresh))
    x1 = det_x * stride * resize
    y1 = det_y * stride * resize
    x2 = (x1 + kernel) * resize
    y2 = (y1 + kernel) * resize
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    probs_clf = probs[probs>clf_thresh]
    clf_boxes = np.array([x1,y1,x2,y2,xc,yc,probs_clf], dtype='float32').T
    return clf_boxes


def non_max_suppression(clf_boxes, overlapThresh=0.25):
    probs = clf_boxes[:,-1]
    # if there are no boxes, return an empty list
    if len(clf_boxes) == 0:
        return []
    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if clf_boxes.dtype.kind == "i":
        clf_boxes = clf_boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = clf_boxes[:, 0]
    y1 = clf_boxes[:, 1]
    x2 = clf_boxes[:, 2]
    y2 = clf_boxes[:, 3]
    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2
    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs
    # sort the indexes
    idxs = np.argsort(idxs)
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
        
    if len(pick) == 0:
        return []
    # return only the bounding boxes that were picked
    if len(pick) > 0:
        return clf_boxes[pick].astype("int")


def draw_bounding_boxes(image, boxes, color=(0,255,0), lw=2):
    if len(boxes) > 0:
        for (x1,y1,x2,y2) in np.nditer((boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3])):
            cv2.rectangle(image, (x1,y1), (x2,y2), color, lw)


def draw_detection_circles(image, boxes, radius=140, color=(0,255,0), lw=2):
    if len(boxes) > 0:
        for (xc,yc) in np.nditer((boxes[:,4],boxes[:,5])):
            cv2.circle(image,(xc,yc), radius, color, lw)


def draw_center_dots(image, boxes, radius=15, color=(0,255,0), lw=-1):
    if len(boxes) > 0:
        for (xc,yc) in np.nditer((boxes[:,4],boxes[:,5])):
            cv2.circle(image,(xc,yc), radius, color, lw)


def detect_palm(sourceImg_fname,
                inputs_path,
                model,
                clf_thresh=0.5,
                overlapThresh=0.25,
                kernel=250,
                stride=10,
                resize=0.2,
                batch_size=2048):
    tic = time.time()
    sourceImg_path = inputs_path + sourceImg_fname
    sourceImg = cv2.imread(sourceImg_path)
    sourceImg_resized = cv2.resize(sourceImg, (0,0), fx=resize, fy=resize)
    sourceImg_resized = np.expand_dims(sourceImg_resized, axis=0)
    kernel_resized = int(kernel * resize)
    stride_resized = int(stride * resize)

    # Crop main sourceImg into thousands of windows
    with tf.Session() as sess:
        test_data = tf.extract_image_patches(images=sourceImg_resized,
                                               ksizes=[1, kernel_resized, kernel_resized, 1],
                                               strides=[1, stride_resized, stride_resized, 1],
                                               rates=[1, 1, 1, 1], padding='VALID').eval()

    # Reshape and transpose to fit into Keras ConvNet Model
    test_data = test_data.reshape(-1,kernel_resized,kernel_resized,3)
    # Switch BGR (from OpenCV3) -> RGB
    test_data = test_data[:,:,:,[2, 1, 0]]
    
    toc1 = time.time()
    # Run classification to every single one of window
    print('')
    print('[Detection Started] Image Filename : {}'.format(sourceImg_fname))
    probs = model.predict_proba(test_data, batch_size=batch_size)[:,0]
    print('')
    toc2 = time.time()

    # Reshape Resulting Window into a rectangle of Prediction Probabilities
    x_pred_num = (sourceImg_resized.shape[2]-kernel_resized)//stride_resized + 1
    y_pred_num = (sourceImg_resized.shape[1]-kernel_resized)//stride_resized + 1
    probs = probs.reshape((y_pred_num,x_pred_num))

    print('### Inference Finished ###')
    print('Number of boxes in x, y directions: {}, {}'.format(x_pred_num, y_pred_num))

    # Table containing all the sliding windows
    all_boxes = create_clf_boxes(probs, clf_thresh=0.0, kernel=kernel, stride=stride)
    # Sliding windows above certain classification proba threshold
    clf_boxes = create_clf_boxes(probs, clf_thresh=clf_thresh, kernel=kernel, stride=stride)
    # Apply Non-Max Suppression and return their x1,y1,x2,y2 coordinates of filtered boxes and their probas
    nms_boxes = non_max_suppression(clf_boxes, overlapThresh=overlapThresh)

    print('Number of qualified classification boxes: {}'.format(clf_boxes.shape[0]))
    print('### Detection boxes after Non-Max-Suppression: {} ###'.format(len(nms_boxes)))

    ### SAVING RESULTS INTO SEPARATE PNGs
    # RECTANGLES
    # load the sourceImg and clone it
    sourceImg = cv2.imread(sourceImg_path)
    sourceImg_clf = sourceImg.copy()
    sourceImg_nms = sourceImg.copy()
    # Draw Detection Rectangles
    # draw_bounding_boxes(sourceImg, all_boxes)
    draw_bounding_boxes(sourceImg_clf, clf_boxes)
    draw_bounding_boxes(sourceImg_nms, nms_boxes)
    # Save sourceImgs of Detection Rectangles
    # cv2.imwrite(results_rectangles_path+'rectangles_'+os.path.splitext(sourceImg_fname)[0]+'_orig'+os.path.splitext(sourceImg_fname)[1], sourceImg)
    cv2.imwrite(results_rectangles_path+'rectangles_'+os.path.splitext(sourceImg_fname)[0]+'_clf'+os.path.splitext(sourceImg_fname)[1], sourceImg_clf)
    cv2.imwrite(results_rectangles_path+'rectangles_'+os.path.splitext(sourceImg_fname)[0]+'_nms'+os.path.splitext(sourceImg_fname)[1], sourceImg_nms)
    # CIRCLES
    # load the sourceImg and clone it
    sourceImg = cv2.imread(sourceImg_path)
    sourceImg_clf = sourceImg.copy()
    sourceImg_nms = sourceImg.copy()
    # Draw Detection Circles
    # draw_detection_circles(sourceImg, all_boxes)
    draw_detection_circles(sourceImg_clf, clf_boxes)
    draw_detection_circles(sourceImg_nms, nms_boxes)
    # Save sourceImgs of Detection Circles
    # cv2.imwrite(results_circles_path+'circles_'+os.path.splitext(sourceImg_fname)[0]+'_orig'+os.path.splitext(sourceImg_fname)[1], sourceImg)
    cv2.imwrite(results_circles_path+'circles_'+os.path.splitext(sourceImg_fname)[0]+'_clf'+os.path.splitext(sourceImg_fname)[1], sourceImg_clf)
    cv2.imwrite(results_circles_path+'circles_'+os.path.splitext(sourceImg_fname)[0]+'_nms'+os.path.splitext(sourceImg_fname)[1], sourceImg_nms)
    # DOTS
    # load the sourceImg and clone it
    sourceImg = cv2.imread(sourceImg_path)
    sourceImg_clf = sourceImg.copy()
    sourceImg_nms = sourceImg.copy()
    # Draw Detection Rectangles
    # draw_center_dots(sourceImg, all_boxes)
    draw_center_dots(sourceImg_clf, clf_boxes)
    draw_center_dots(sourceImg_nms, nms_boxes)
    # Save sourceImgs of Detection Rectangles
    # cv2.imwrite(results_dots_path+'dots_'+os.path.splitext(sourceImg_fname)[0]+'_orig'+os.path.splitext(sourceImg_fname)[1], sourceImg)
    cv2.imwrite(results_dots_path+'dots_'+os.path.splitext(sourceImg_fname)[0]+'_clf'+os.path.splitext(sourceImg_fname)[1], sourceImg_clf)
    cv2.imwrite(results_dots_path+'dots_'+os.path.splitext(sourceImg_fname)[0]+'_nms'+os.path.splitext(sourceImg_fname)[1], sourceImg_nms)
    
    # Initialize Empty Dataframe For Single Image Result
    boxes_proba_df = pd.DataFrame(columns=['x1','y1','x2','y2','xc','yc'])
    
    # Combine Non-Max-Suppression Results (With Probabilities) into a single dataframe
    if len(nms_boxes) > 0:
        boxes_proba = nms_boxes[:,:-1]
        boxes_proba_df = pd.DataFrame(boxes_proba, columns=['x1','y1','x2','y2','xc','yc'])
    
    # Combine Dataframes Into A Summary Dataframe
    palm_num_df = pd.DataFrame(np.arange(len(boxes_proba_df)), columns=['Palm Number'])
    fname_df = pd.DataFrame([sourceImg_fname], columns=['Filename'])
    summary_detections_df = pd.concat([boxes_proba_df,fname_df,palm_num_df], axis=1)
    summary_detections_df = summary_detections_df.fillna(value = sourceImg_fname , axis=1)   
    
    # Record Time and Print Results
    toc3 = time.time()
    time_clf = toc2 - toc1
    time_all = toc3 - tic
    
    # Create Summary Dataframe Containing Image Filename and Number of Detected Palms
    summary_line = {'Filename': [sourceImg_fname], 'Palms': [len(nms_boxes)], 'Proc Time All': [time_all], 'Proc Time Clf': [time_clf]}
    summary_line_df = pd.DataFrame.from_dict(summary_line)
    
    print('Time taken for all {0} sliding window classification: {1:.4f} s'.format(x_pred_num*y_pred_num, time_clf))
    print('Time taken for all processes from beginning to end: {0:.4f} s'.format(time_all))
    print('### All processes finished and all 6 PNGs are saved ###')
    
    return summary_detections_df, summary_line_df


def demo_detect(inputs_path,
                model,
                clf_thresh=0.5,
                overlapThresh=0.25,
                kernel=250,
                stride=10,
                resize=0.2,
                batch_size=2048):
    
    main_tic = time.time()
    # Initialize Combined Dataframes
    summary_detections_df_full = pd.DataFrame(columns=['x1','y1','x2','y2','xc','yc','Filename','Palm Number']) 
    summary_line_df_full = pd.DataFrame(columns=['Filename','Palms','Proc Time All','Proc Time Clf']) 
    
    for Img_fname in os.listdir(inputs_path):
        # MAIN FUNCTION CALLER
        summary_detections_df, summary_line_df = detect_palm(sourceImg_fname=Img_fname,
                                                      inputs_path=inputs_path,
                                                      model=model,
                                                      clf_thresh=clf_thresh,
                                                      overlapThresh=overlapThresh,
                                                      kernel=kernel,
                                                      stride=stride,
                                                      resize=resize,
                                                      batch_size=batch_size)
        # Append Resulting Dataframe to Main Dataframe
        summary_detections_df_full = summary_detections_df_full.append(summary_detections_df, ignore_index=True)
        summary_line_df_full = summary_line_df_full.append(summary_line_df, ignore_index=True)

    # Aggregated Total Number of Trees Across All Images in Input Directory
    total_palm = summary_line_df_full['Palms'].values.sum()
    
    # Saving Results into CSV
    time_now = str(datetime.now())[:-7]
    time_now = time_now.replace(':','_').replace(' ','_')
    summary_detections_fname = time_now + '_detections_summary.csv'
    summary_line_fname = time_now + '_line_summary.csv'
    summary_detections_df_full.to_csv(results_csv_path + summary_detections_fname)
    summary_line_df_full.to_csv(results_csv_path + summary_line_fname)
    
    # Print and Return All Results
    main_toc = time.time()

    print('')
    print('######### ALL IMAGES PROCESSED ########')    
    print('#### TOTAL PALMS DETECTED = {} ######'.format(total_palm))
    print('#### TOTAL TIME TAKEN = {0:.3f} s #####'.format(main_toc - main_tic))
    print('#######################################')
    print('')
    print('Saving {}'.format(summary_detections_fname))
    print('Saving {}'.format(summary_line_fname))
    print('### All images results saved to directory: {}'.format(results_path))
    print('### All CSV results saved to directory: {}'.format(results_csv_path))
    print('')

    return summary_detections_df_full, summary_line_df_full

def initial_size_check(inputs_path = 'inputs/',
                       inputs_big_storage = 'inputs_big/',
                       image_width_limit = 6100,
                       image_height_limit = 5100,
                       buffer_window = 180):
    for sourceImg_fname in os.listdir(path=inputs_path):
        sourceImg_path = inputs_path + sourceImg_fname
        sourceImg = cv2.imread(sourceImg_path)
        src_image_height = sourceImg.shape[0]
        src_image_width = sourceImg.shape[1]

        if src_image_width > image_width_limit or src_image_height > image_height_limit:
            print('')
            print('Info: Input image size is too big! [{}]'.format(sourceImg_fname))
            print('Info: This is not a critical error - but an automated mechanism to prevent memory overflow')
            print('Info: Maximum allowed input image size (width, height) = (6100, 5100)')
            print('Info: Initiating protocol to partition large images into smaller chunks...')
            xb_num = src_image_width // image_width_limit + 1
            yb_num = src_image_height // image_height_limit + 1
            for xb in range(xb_num):
                for yb in range(yb_num):
                    cropped_img_fname = sourceImg_fname[:-4]+'_{}_{}.png'.format(yb+1,xb+1)
                    cv2.imwrite(inputs_path+cropped_img_fname,
                                sourceImg[yb*image_height_limit-yb*buffer_window:
                                          (yb+1)*image_height_limit-yb*buffer_window,
                                          xb*image_width_limit-xb*buffer_window:
                                          (xb+1)*image_width_limit-xb*buffer_window])
                    print('Saving cropped file [{}] to directory [{}]'.format(cropped_img_fname,inputs_path))
            os.rename(sourceImg_path, inputs_big_storage+sourceImg_fname)
            print('Original image [{}] has been moved to directory [{}]'.format(sourceImg_fname,inputs_big_storage))
            print('========================')


if __name__ == '__main__':

    # Fill All necessary inputs here
    # CHANGE BATCH_SIZE VALUE TO THESE RECOMMENDED VALUES:
    # USING CPU, 8GB RAM: batch_size = 128
    # USING GPU GTX1050, 16GB RAM: batch_size = 1024
    # USING GPU GTX1070, 16GB RAM: batch_size = 2048
	batch_size = 2048
    # NOT RECOMMENDED TO CHANGE THESE VALUES (UNLESS SPECIFIED OTHERWISE)
    # Dimensions of our images.
	img_width, img_height = 50, 50
	# Detection Parameters
	clf_thresh = 0.5
	overlapThresh = 0.25
	kernel = 250
	stride = 10
	resize = 0.2
	# Input parameters
	nb_train_samples = 4460
	nb_validation_samples = 500
	epochs = 10
	# Size Limit Parameters
	image_width_limit = 6100
	image_height_limit = 5100
	buffer_window = 180
	# Directory Paths Here                    
	path = 'data/'
	train_data_dir = path +'train/'
	validation_data_dir = path + 'valid/'
	model_path = path + 'models/'
	model_weights_fname = 'gridlenet50_v3.h5'
	inputs_path = 'inputs/'
	inputs_big_storage = 'inputs_big/'
	results_path = 'results/'
	results_rectangles_path = results_path + 'rectangles/'
	results_circles_path = results_path + 'circles/'
	results_dots_path = results_path + 'dots/'
	results_csv_path = results_path + 'csv/'

	# Obtain Convolutional Neural Network Model

	lenet = GRID_lenet(model_path = model_path,
					   model_weights_fname = model_weights_fname ,
					   img_width = img_width,
					   img_height = img_height)


	initial_size_check(inputs_path = inputs_path,
	                   inputs_big_storage = inputs_big_storage,
	                   image_width_limit = image_width_limit,
	                   image_height_limit = image_height_limit,
	                   buffer_window = buffer_window)

	summary_detections_df_full, summary_line_df_full = demo_detect(inputs_path=inputs_path,
																   model=lenet,
																   clf_thresh=clf_thresh,
																   overlapThresh=overlapThresh,
																   kernel=kernel,
																   stride=stride,
																   resize=resize,
																   batch_size=batch_size)







