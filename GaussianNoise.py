import cv2
import numpy as np
from PIL import Image
from yolov4.tf import YOLOv4

#Prints out the yolo detections of the image with and without noise
#Currently only returns bounds of detections that differ in labels
def gaussianTest(yolo, image, prob=.25, writeName='', showImage = True, var=.25):
    disagreements = []
    
    noNoiseResults = yolo.predict(image, var) #gets initial yolo predictions
    
    #Gets the noisy predictions
    noisy = addNoise(image, .25)
    noisyResults = yolo.predict(noisy, .25)
    
    #Prints out each yolo no noise detection and the corresponding detection with noise
    #NOTE: does not print out info for detections with noise without a corresponding detection without noise
    for r, v in enumerate(noNoiseResults):
        matches = getMatch(v, noisyResults)
        
        if len(matches) == 0:
            print(f'Center x: Original {v[0]:.3f}, Noisy N/A')
            print(f'Center y: Original {v[1]:.3f}, Noisy N/A')
            print(f'Width: Original {v[2]:.3f}, Noisy N/A')
            print(f'Height: Original {v[3]:.3f}, Noisy N/A')
            print(f'Class ID: Original {v[4]:.0f}, Noisy N/A')
            print(f'Confidence: Original {v[5]:.3f}, Noisy N/A\n')    
        
        else:
            for i in matches:
                if v[4] != i[4]:
                    disagreements.append(i)        
            
            print(f'Center x: Original {v[0]:.3f} Noisy ', end='')
            [print(f'{i[0]:.3f}', sep=' ') for i in matches]
            
            print(f'Center y: Original {v[1]:.3f}, Noisy ', end='')
            [print(f'{i[1]:.3f}', sep=' ') for i in matches]
            
            print(f'Width: Original {v[2]:.3f}, Noisy ', end='')
            [print(f'{i[2]:.3f}', sep=' ') for i in matches]
            
            print(f'Height: Original {v[3]:.3f}, Noisy ', end='')
            [print(f'{i[3]:.3f}', sep=' ') for i in matches]
            
            print(f'Class ID: Original {v[4]:.0f}, Noisy ', end='')
            [print(f'{i[4]:.0f}', sep=' ') for i in matches]
            
            print(f'Confidence: Original {v[5]:.3f}, Noisy ', end='')
            [print(f'{i[5]:.3f}', sep=' ') for i in matches]
            print()

    
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
        
    return disagreements

#No noise is a single detection, noisy is the full set of noisy detections
def getMatch(noNoise, noisy):
    matches = []
    for n in noisy:
        if(n[0] > noNoise[0] - .01 and n[0] < noNoise[0] + .01):
            if(n[1] > noNoise[1] - .01 and n[1] < noNoise[1] + .01):
                    if(n[2] > noNoise[2] - .01 and n[2] < noNoise[2] + .01):
                        if(n[3] > noNoise[3] - .01 and n[3] < noNoise[3] + .01):
                            matches.append(n)
    return matches
    
#Adds gaussian noise to an image and returns it   
def addNoise(image, var, mean=0):
    row, col, ch = image.shape
    noise = np.random.normal(mean, var**.5, (row,col,ch))
    noise = noise.reshape(row, col, ch)
    noisy = image + noise
    return noisy
