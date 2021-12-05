import cv2
from yolov4.tf import YOLOv4
from GaussianNoise import gaussianTest
import platform
import numpy as np
from PIL import Image
import os

yolo = YOLOv4()

yolo.config.parse_names("ModelData/coco.names")
yolo.config.parse_cfg("ModelData/yolov4.cfg")

yolo.make_model()
yolo.load_weights("ModelData/yolov4.weights", weights_type="yolo")
yolo.summary(summary_type="yolo")
yolo.summary()

gaussianTest(yolo, 'example.jpg', writeName='')
'''
count = 0
variances = [.1, .4, .45]
writer = open('results.txt', 'w')
files = os.listdir('images/')

for v in variances:
    totalAdversaries = 0
    adversaries = 0
    totalPositives = 0
    positives = 0
    totalNoNoiseFN = 0
    noNoiseFN = 0
    totalNoisyFN = 0
    noisyFN = 0
    totalNoNoiseAdversary = 0
    noNoiseAdversary = 0
    totalNoisyAdversary = 0
    noisyAdversary = 0

    print('Starting ', v)
    count = 0
    for f in files:
        count+=1
        im = 'images/' + f
        noNoiseFN, noNoiseAdversary, noisyFN, noisyAdversary, positives, adversaries = gaussianTest(yolo, im, showImage = False, verbose = False, var=v, writeName='')
        
        totalAdversaries += adversaries
        totalPositives += positives
        totalNoNoiseFN += noNoiseFN
        totalNoNoiseAdversary += noNoiseAdversary
        totalNoisyFN += noisyFN
        totalNoisyAdversary += noisyAdversary
        
        if count%100 == 0:
            print("Finished image ", count)

    print("Finished ", v)
    print("Total true positives: ", totalPositives)
    print("Total adversaries: ", totalAdversaries)
    print("Total No Noise False Negatives: ", totalNoNoiseFN)
    print("Total No Noise Adversaries Detected: ", totalNoNoiseAdversary)
    print("Total Noisy False Negatives: ", totalNoisyFN)
    print("Total Noisy Adversaries Detected: ", totalNoisyAdversary)
    
    writer.write("Results for " + str(v) + " variance on Gaussian Noise\n")
    writer.write("Total true positives: "+ str(totalPositives)+"\n")
    writer.write("Total adversaries: "+ str(totalAdversaries)+"\n")
    writer.write("Total No Noise False Negatives: "+ str(totalNoNoiseFN)+"\n")
    writer.write("Total No Noise Adversaries Detected: "+ str(totalNoNoiseAdversary)+"\n")
    writer.write("Total Noisy False Negatives: "+ str(totalNoisyFN)+"\n")
    writer.write("Total Noisy Adversaries Detected: "+ str(totalNoisyAdversary)+"\n\n")
    
writer.close()
'''

