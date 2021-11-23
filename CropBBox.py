# This class was adapted from the a combination of the crop_objects() provided in:
# https://github.com/theAIGuysCode/yolov4-custom-functions
# With some help from yolov4/common/base_class.py and yolov4/common/media.py

import os
import cv2
import numpy as np

# function for cropping each detection and saving as new image
def crop_objects(img, bboxes, path):
    class_names = {}
    with open('modelData/coco.names', 'r') as data:
        for ID, name in enumerate(data):
            class_names[ID] = name.strip('\n')
    # create dictionary to hold count of objects for image name
    counts = dict()
    # un-normalize bboxes
    image = np.copy(img)
    height, width, _ = image.shape
    bboxes = bboxes * np.array([width, height, width, height, 1, 1])

    for bbox in bboxes:
        # get count of class for part of image name
        class_index = int(bbox[4])
        class_name = class_names[class_index]
        counts[class_name] = counts.get(class_name, 0) + 1
        # get box values
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        half_w = int(bbox[2] / 2)
        half_h = int(bbox[3] / 2)
        # calculate x and y min and max
        xmin = c_x - half_w
        if xmin < 5:
            xmin = 5
        ymin = c_y - half_h
        if ymin < 5:
            ymin = 5
        xmax = c_x + half_w
        ymax = c_y + half_h
        # crop detection from image and take an additional 5 pixels around all edges
        cropped_img = image[ymin - 5:ymax + 5, xmin - 5:xmax + 5]
        # construct image name and join it to path for saving crop properly
        img_name = class_name + '_' + str(counts[class_name]) + '.jpg'
        img_path = os.path.join(path, img_name)
        # save image
        cv2.imwrite(img_path, cropped_img)
