import cv2
import numpy as np
from PIL import Image
from yolov4.tf import YOLOv4

#Prints out the yolo detections of the image with and without noise
def gaussianTest(yolo, imName, prob=.25, writeName='', showImage = True, var=.25, verbose = True):
    image = np.asarray(Image.open(imName).convert("RGB"))

    noNoiseResults = yolo.predict(image, prob) #gets initial yolo predictions

    #Gets the noisy predictions
    noisy = addNoise(image, var)
    noisyResults = yolo.predict(noisy, prob)

    label = imName.replace("images", "labels")
    label = label.replace(".jpg", ".txt")
    
    noNoiseFN, noNoiseFP, truePositives, trueAdversaries = getResults(noNoiseResults, label)
    noisyFN, noisyFP, truePositives, trueAdversaries = getResults(noisyResults, label)

    if verbose and not (truePositives == 0 and trueAdversaries==0):
        print('True positives: ', truePositives)
        print('True adversaries: ', trueAdversaries)
        
        print('No Noise False Negatives: ', noNoiseFN)
        print('No Noise Adversaries Detected: ', noNoiseFP)
        
        print('Noisy False Negatives: ', noisyFN)
        print('Noisy Adversaries Detected: ', noisyFP)

    if writeName != '' or showImage:
        #gets the bounding boxes around the images
        image = yolo.draw_bboxes(image, noNoiseResults)
        noisy = yolo.draw_bboxes(noisy, noisyResults)
        
        #Combines the images into one   
        images = np.vstack((image.astype(np.uint8),noisy.astype(np.uint8)))
    
    #Writes if specified
    if writeName != '':
        Image.fromarray(images).save(writeName)
        
    #Displays the image if specified
    if showImage:
        Image.fromarray(images).show()
        
    return noNoiseFN, noNoiseFP, noisyFN, noisyFP, truePositives, trueAdversaries

def getResults(yoloResults, label):
    falseNegatives = 0
    falsePositives = 0
    truePositives = 0
    trueAdversaries = 0
    try:
        labels = open(label, 'r')
        
        for line in labels:
            line = line.split()
            if int(line[0]) == 80:
                trueAdversaries += 1
            else:
                truePositives += 1
            
            detected = False
            for r in yoloResults:
                if match(line, r):
                    detected = True
                    
            if detected and int(line[0]) == 80:
                falsePositives+=1
            
            if not detected and int(line[0]) != 80:
                falseNegatives+=1
        labels.close()
    
        return falseNegatives, falsePositives, truePositives, trueAdversaries
    except:
        return 0,0,0,0
    
#Checks to see if the given label and the given result match
def match(label, result):
    tolerance = .05
    #print(label, " ", result)
    if(float(label[1]) > result[0] - tolerance and float(label[1]) < result[0] + tolerance):
            if(float(label[2]) > result[1] - tolerance and float(label[2]) < result[1] + tolerance):
                    if(float(label[3]) > result[2] - tolerance and float(label[3]) < result[2] + tolerance):
                        if(float(label[4]) > result[3] - tolerance and float(label[4]) < result[3] + tolerance):
                            return True
    return False
   
#Adds gaussian noise to an image and returns it   
def addNoise(image, var, mean=0):
    row, col, ch = image.shape
    noise = np.random.normal(mean, var**.5, (row,col,ch))
    noise = noise.reshape(row, col, ch)
    noisy = image + noise
    return noisy
