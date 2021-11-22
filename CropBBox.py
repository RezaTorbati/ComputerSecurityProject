import cv2
from yolov4.tf import YOLOv4
import numpy as np
from pynput.keyboard import Key, Controller

class CropBBox:

    def __init__(self, img_folder, num_images):
        self.img_folder = img_folder
        self.num_images = num_images
        self.keyboard = Controller()

        self.yolo = YOLOv4()
        self.yolo.config.parse_names("modelData/coco.names")
        self.yolo.config.parse_cfg("modelData/yolov4.cfg")
        self.yolo.make_model()
        self.yolo.load_weights("modelData/yolov4.weights", weights_type="yolo")
        self.yolo.summary(summary_type="yolo")
        self.yolo.summary()

    def inferImages(self):
        # These shenanigans were done to account for APRICOT's image filenames, to put back to normal just use img_id:
        imgfile = ''
        for i in range(self.num_images):
            if i in range(0, 38):
                imgfile = f'frs4_{i}'
                if i == 4 or i == 9 or i == 32:
                    imgfile = f'frs4_{i+1}'

                elif i in range(38, 49):
                    imgfile = f'frs10_{i - 38}'
                    if i == 38 + 2:
                        imgfile = f'frs10_{i - 38+1}'

                elif i in range(49, 84):
                    imgfile = f'rrc3_{i - 49}'

                elif i in range(84, 97):
                    imgfile = f'rrc6_{i - 84}'

                elif i in range(97, 131):
                    imgfile = f'sms2_{i - 97}'
                    if i == 97 + 15 or i == 97 + 28:
                        imgfile = f'sms2_{i - 97+1}'

                elif i in range(131, 144):
                    imgfile = f'sms8_{i - 131}'

                # Retrieve image.
                image_path = f'{self.img_folder}\\{imgfile}.jpg'
                self.yolo.inference(media_path=image_path, cv_frame_size=(640, 480))
                self.keyboard.press('q')
                self.keyboard.release('q')


CropBBox(img_folder='D:\\Coding\\PyCharmProjects\\ComputerSecurityProject\\APRICOT\\Images\\dev', num_images=144).inferImages()
