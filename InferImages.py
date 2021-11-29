# Note that for this class to work, inference() in yolov4/common/base_class.py
# Must be modified by adding "return bboxes" at the end of the function.
import glob

import cv2
from yolov4.tf import YOLOv4
import os
import CropBBox

class InferImages:

    def __init__(self):

        self.yolo = YOLOv4()
        self.yolo.config.parse_names("modelData/adversary.names")
        self.yolo.config.parse_cfg("modelData/yolov4.cfg")
        self.yolo.make_model()
        self.yolo.load_weights("modelData/yolov4.weights", weights_type="yolo")
        self.yolo.summary(summary_type="yolo")
        self.yolo.summary()

    def inferImages(self):
        # These shenanigans were done to account for APRICOT's image filenames in the "dev" folder
        filenames = glob.glob('D:\\Coding\\PyCharmProjects\\ComputerSecurityProject\\APRICOT\\Images\\test\\*.jpg')
        filenames.sort()
        for image_path in filenames:
            # Retrieve image and run inference
            print(image_path)
            bboxes = self.yolo.inference(media_path=image_path)
            # Create a directory for the cropped images
            #original_image = cv2.imread(image_path)
            #crop_path = os.path.join(os.getcwd(), 'croppedImages', imgfile)
            #try:
             #   os.mkdir(crop_path)
            #except FileExistsError:
             #   pass
            # Crop the original image into separate images that only include the bounding boxes
            #CropBBox.crop_objects(original_image, bboxes, crop_path)

InferImages().inferImages()
