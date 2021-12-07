# ComputerSecurityProject
Group 14's project for Computer Security at the University of Oklahoma. Goal is to detect when someone is trying to use adversarial perturbations to fool YOLOv4. <br>
To accomplish this, we implimented three methods: the Gaussian noise test, a statistical test and a custom model. <br>
<br>
## Running the Gaussian Noise test
1. Install the dependencies. This can be done with: <br>
```
    python3 -m pip install opencv-python
    python3 -m pip install tensorflow
    python3 -m pip install yolov4
```
2. Download the weights [here](https://drive.google.com/file/d/15P4cYyZ2Sd876HKAEWSmeRdFl_j-0upi/view) and put them in Gaussian/ModelData
3. Modify line 19 of Gaussian/analyze.py to analyze the image of choice and then run it with <br>
    `python analyze.py` <br>
   For this image to be analyzed, there must be a corresponding YOLO label file, such as the given "examples.txt"
4. Alternatively, create a folder called "images" in the Gaussian folder and put however many images of interest in it. Then remove the comments at lines 20 and 74 of analyze.py and then run `python analyze.py`. This will analyze all images in the folder and print out the aggregate results. For this to return anything useful, you must also have a folder called "labels" with YOLO style labelings for each image.

## Running the Statistical Test
1. After installing the dependencies for the Gaussian test, you'll just need to ` pip install torch ` 
2. StatisticalDetection.py expects two paths with folders of images. APRICOT can be downloaded [here](https://apricot.mitre.org/download/) and COCO can be downloaded [here](https://cocodataset.org/#download)
3. Once you have the image folders, replace lines 101 & 102 with one or both of the folder paths. Then, just run `python StatsDetect/StatisticalDetection.py` to obtain the MMD value.

## Running the Custom Version of YOLO
1. Run the following in CustomModel: <br>
```
    git clone https://github.com/AlexeyAB/darknet
    cd darknet
    make
```
2. Download the Default YOLO weights from [here](https://drive.google.com/file/d/15P4cYyZ2Sd876HKAEWSmeRdFl_j-0upi/view) and put in them in CustomModel/DefaultData
3. Download the custom YOLO weights from [here](https://drive.google.com/file/d/1FFYXSInyHK0S2GhzOpaXyJKXjGdPW9vH/view?usp=sharing) and put them in CustomModel/CustomData
4. For the default model, in CustomData/darknet run <br>
    `./darknet detector test  ../DefaultData/coco.data ../DefaultData/yolov4.cfg ../DefaultData/yolov4.weights ../adversaryEx.jpg -thresh 0.3` <br>
    And observe the predicted results in CustomModel/darknet/predictions.jpg
5. For the custom model, in CustomData/darknet run <br>
    `./darknet detector test  ../CustomData/adversary.data ../CustomData/adversary.cfg ../CustomData/adversary.weights ../adversaryEx.jpg -thresh 0.3` <br>
    And observe the predicted results in CustomModel/darknet/predictions.jpg
    

