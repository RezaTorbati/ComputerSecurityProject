import cv2
import numpy as np
from PIL import Image
from yolov4.tf import YOLOv4

#Prints out the yolo detections of the image with and without noise
def gaussianTest(yolo, image, prob=.25, writeName='', showImage = True, var=.25):
    noNoiseResults = yolo.predict(image, var) #gets initial yolo predictions
    
    #Gets the noisy predictions
    noisy = addNoise(image, .25)
    noisyResults = yolo.predict(noisy, .25)
    
    #Assumes that YOLO still detects the same number of objects...
    for r, v in enumerate(noNoiseResults):
        try: #Incase the noisy detections go out of bounds
            print(r, " ", v)
            print(f'Center x: Original {v[0]:.3f}, Noisy {noisyResults[r][0]:.3f}')
            print(f'Center y: {v[1]:.3f}, Original Noisy {noisyResults[r][1]:.3f}')
            print(f'Width: Original {v[2]:.3f}, Noisy {noisyResults[r][2]:.3f}')
            print(f'Height: Original {v[3]:.3f}, Noisy {noisyResults[r][3]:.3f}')
            print(f'Class ID: Original {v[4]:.0f}, Noisy {noisyResults[r][4]:.0f}')
            print(f'Confidence: Original {v[5]:.3f}, Noisy {noisyResults[r][5]:.3f}\n')
        except IndexError:
            print("It seems that there are fewer images detected with Gaussian noise. Proceed with caution")

    
    #gets the bounding boxes around the images
    image = yolo.draw_bboxes(image, noNoiseResults)
    noisy = yolo.draw_bboxes(noisy, noisyResults)
    
    #Combines the images into one   
    images = np.vstack((image.astype(np.uint8),noisy.astype(np.uint8)))
    
    #Writes if specified
    if writeName != '':
        Image.fromarray(images).save(writeName)
        
    #Displays the image if specified
    if(showImage):
        Image.fromarray(images).show()
 
#Adds gaussian noise to an image and returns it   
def addNoise(image, var, mean=0):
    row, col, ch = image.shape
    noise = np.random.normal(mean, var**.5, (row,col,ch))
    noise = noise.reshape(row, col, ch)
    noisy = image + noise
    return noisy
