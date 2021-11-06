import cv2
from yolov4.tf import YOLOv4
from GaussianNoise import gaussianTest
import platform
import numpy as np
from PIL import Image

yolo = YOLOv4()

yolo.config.parse_names("modelData/coco.names")
yolo.config.parse_cfg("modelData/yolov4.cfg")

yolo.make_model()
yolo.load_weights("modelData/yolov4.weights", weights_type="yolo")
yolo.summary(summary_type="yolo")
yolo.summary()

im = np.asarray(Image.open('images/stopSignsBall.png').convert("RGB"))

gaussianTest(yolo, im, writeName='images/GaussianResults/stopSignsBall.jpg')

'''
results = yolo.predict(im, .25) #.25 is standard
for r in results:
    print(f'Center x: {r[0]:.3f}')
    print(f'Center y: {r[1]:.3f}')
    print(f'Width: {r[2]:.3f}')
    print(f'Height: {r[3]:.3f}')
    print(f'Class ID: {r[4]:.0f}')
    print(f'Confidence: {r[5]:.3f}\n')
'''
#test = yolo.inference(media_path="kite.jpg")
#yolo.inference(media_path="road.mp4", is_image=False)

'''
preference = cv2.CAP_V4L2
deviceLocation = "/dev/video0"
if platform.system()=='Windows':
    preference = cv2.CAP_DSHOW
    deviceLocation = 0

yolo.inference(
    deviceLocation,
    is_image=False,
    cv_apiPreference=preference,    
    cv_frame_size=(640, 480),    
    cv_fourcc="YUYV",
)
'''
