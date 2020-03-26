## Unified Gesture Recognition and Fingertip Detection
A unified convolutional neural network (CNN) algorithm for both hand gesture recognition and fingertip detection is
presented here. The proposed algorithm uses a single network to predict both finger class probabilities and fingertips positional 
output in one evaluation. From the finger class probabilities, the gesture is recognized and using both of the
information fingertips are localized. Instead of directly regressing fingertips position from the fully connected (FC) layer of the 
CNN, we regress the ensemble of fingertips position from the fully convolutional network (FCN) and subsequently take ensemble 
average to regress the final fingertips positional output.

## Update 
Included ```robust real-time hand detection with yolo``` for better performance and most of the code has been cleaned and 
restructured for ease of use. 
previous versions can be found [here](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/releases).

[![GitHub stars](https://img.shields.io/github/stars/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/network)
[![GitHub issues](https://img.shields.io/github/issues/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/issues)
[![Version](https://img.shields.io/badge/version-1.1-orange.svg?longCache=true&style=flat)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality)
[![GitHub license](https://img.shields.io/github/license/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/blob/master/LICENSE)

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/77615813-6de9cc80-6f5a-11ea-9172-a95e5604147c.gif" width="400">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/72676259-5f45eb80-3ab9-11ea-96d7-436f160a4b84.png" width="600">
</p>

## Requirements
- [x] TensorFlow-GPU==1.15.0
- [x] Keras==2.2.4
- [x] ImgAug==0.2.6
- [x] OpenCV==4.2.0
- [x] Weights: [```Download the pre-trained weights```](https://mega.nz/#F!y9dBAKiK!gDd8AZCax2IIUGo4W4ixUw) files of the unified gesture recognition and fingertip detection model and put the weights folder in the working directory.

[![Downloads](https://img.shields.io/badge/download-weights-green.svg?style=popout-flat&logo=mega)](https://mega.nz/#F!6stCxY5b!oB-3279KkhfhRULQFQO7yQ)
[![Downloads](https://img.shields.io/badge/download-weights-blue.svg?style=popout-flat&logo=dropbox)](https://www.dropbox.com/sh/7pbfrgaor678eft/AAA8r5ADlMde0WkAtJQO_lo5a?dl=0)

The ```weights``` folder contains four weights files. The ```classes5.h5``` is for first five classes, ```classes8.h5``` 
is for first eight classes. ```yolo.h5``` and ```solo.h5``` are for the yolo and solo method of hand detection.

## Dataset
The proposed gesture recognition and fingertip detection model is trained by employing [```Scut-Ego-Gesture Dataset```](http://www.hcii-lab.net/data/SCUTEgoGesture/index.htm) which has a total of eleven different
single hand gesture datasets. Among the eleven different gesture datasets, eight of them are considered for the experimentation. 
The detail explanation about the partition of the dataset along with the list of the images used in the training, validation, and 
the test set is provided in the 
[```dataset/```](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/tree/master/dataset#dataset-description) 
folder.

## Network Architecture 
To implement the algorithm, the following network architecture is proposed where a single CNN is utilized for both hand gesture recognition and fingertip detection. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60171959-82fbc880-982d-11e9-8c66-ee0109c5368d.jpg">
</p>

## Prediction 
To run the prediction on a single image run the ```predict.py``` file. It will run the prediction in the sample image stored 
in the ```data/``` folder. Here. the output for the ```sample.jpg``` image. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/77616112-139d3b80-6f5b-11ea-81f0-977d50d44c4e.jpg" width="350">
</p>

## Run in Real-Time!
To run in real-time simply clone the repository and download the weights file and then run the ```real-time.py``` file. 
```
directory > python real-time.py
```
In real-time execution, there are two stages. In the first stage, the hand can be detected by using both ```you only look
once (yolo)``` or ```single object localization (solo)``` algorithm. By default yolo will be used here. The detected hand 
portion is then cropped and fed to the second stage for gesture recognition and fingertip detection. 

## Output
Here is the output of the unified gesture recognition and fingertip detection model for all of the 8 classes of the dataset 
where not only each fingertip is detected but also each finger is classified.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60171964-85f6b900-982d-11e9-8f20-af40be2172f8.jpg">
</p>
