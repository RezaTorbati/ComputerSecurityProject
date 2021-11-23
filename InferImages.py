# Note that for this class to work, inference() in yolov4/common/base_class.py
# Must be modified by adding "return bboxes" at the end of the function.

import cv2
from yolov4.tf import YOLOv4
import os
import CropBBox

class InferImages:

    def __init__(self, img_folder, num_images):
        self.img_folder = img_folder
        self.num_images = num_images

        self.yolo = YOLOv4()
        self.yolo.config.parse_names("modelData/coco.names")
        self.yolo.config.parse_cfg("modelData/yolov4.cfg")
        self.yolo.make_model()
        self.yolo.load_weights("modelData/yolov4.weights", weights_type="yolo")
        self.yolo.summary(summary_type="yolo")
        self.yolo.summary()

    def inferImages(self):
        # These shenanigans were done to account for APRICOT's image filenames in the "dev" folder
        imgfile = ''
        for i in range(self.num_images):
            if i in range(0, 38):
                imgfile = f'frs4_{i}'
                if i == 4 or i == 9 or i == 32:
                    imgfile = f'frs4_{i+1}'
                    ++i

            elif i in range(38, 49):
                imgfile = f'frs10_{i - 38}'
                if i == 38 + 2:
                    imgfile = f'frs10_{i - 38+1}'
                    ++i

            elif i in range(49, 84):
                imgfile = f'rrc3_{i - 49}'

            elif i in range(84, 97):
                imgfile = f'rrc6_{i - 84}'

            elif i in range(97, 131):
                imgfile = f'sms2_{i - 97}'
                if i == 97 + 15 or i == 97 + 28:
                    imgfile = f'sms2_{i - 97+1}'
                    ++i

            elif i in range(131, 144):
                imgfile = f'sms8_{i - 131}'

            # Retrieve image and run inference
            image_path = f'{self.img_folder}\\{imgfile}.jpg'
            bboxes = self.yolo.inference(media_path=image_path)
            # Create a directory for the cropped images
            original_image = cv2.imread(image_path)
            crop_path = os.path.join(os.getcwd(), 'croppedImages', imgfile)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            # Crop the original image into separate images that only include the bounding boxes
            CropBBox.crop_objects(original_image, bboxes, crop_path)

InferImages(img_folder='D:\\Coding\\PyCharmProjects\\ComputerSecurityProject\\APRICOT\\Images\\dev', num_images=144).inferImages()
