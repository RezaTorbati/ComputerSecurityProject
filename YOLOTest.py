import cv2
import platform

from yolov4.tf import YOLOv4

yolo = YOLOv4()

yolo.config.parse_names("coco.names")
yolo.config.parse_cfg("yolov4.cfg")

yolo.make_model()
yolo.load_weights("yolov4.weights", weights_type="yolo")
yolo.summary(summary_type="yolo")
yolo.summary()

#yolo.inference(media_path="kite.jpg")
#yolo.inference(media_path="road.mp4", is_image=False)

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
