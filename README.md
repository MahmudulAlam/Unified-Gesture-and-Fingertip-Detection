## Unified Gesture Recognition and Fingertip Detection
A unified convolutional neural network (CNN) algorithm for both hand gesture recognition and fingertip detection is
presented here. The proposed algorithm uses a single network to predict both finger class probabilities and fingertips positional 
output in one evaluation. From the finger class probabilities, the gesture is recognized and using both of the
information fingertips are localized. Instead of directly regressing fingertips position from the fully connected (FC) layer of the 
CNN, we regress the ensemble of fingertips position from the fully convolutional network (FCN) and subsequently take ensemble 
average to regress the final fingertips positional output.

<p align="center">
  <img src="data/unified.gif" width="350">
</p>

## Requirements
- [x] TensorFlow-GPU==1.11.0
- [x] Keras==2.2.4
- [x] OpenCV==3.4.4
- [x] ImgAug==0.2.6
- [x] Weights: Download the trained weights files for gesture recognition and fingertip detection model and put the weights folder in the working directory. 

[![Downloads](https://img.shields.io/badge/download-weights-green.svg?style=popout-flat&logo=mega)](https://mega.nz/#F!y9dBAKiK!gDd8AZCax2IIUGo4W4ixUw)

The ```weights``` folder contains three weights files. The ```comparison.h5``` is for first five classes and ```performance.h5``` 
is for first eight classes. ```solo.h5``` is for hand detection.

## Dataset
The proposed gesture recognition and fingertip detection model is trained by employing [```Scut-Ego-Gesture Dataset```](http://www.hcii-lab.net/data/SCUTEgoGesture/index.htm) which has a total of eleven different
single hand gesture datasets. Among the eleven different gesture datasets, eight of them are considered for the experimentation. 
The detail explanation about the partition of the dataset along with the list of the images used in the training, validation, and 
the test set is provided in the 
[```dataset/```](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/tree/master/dataset#dataset-description) 
folder.

## Network-Architecture 
To implement the algorithm, the following network architecture is proposed where a single CNN is utilized for both hand gesture recognition and fingertip detection. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60171959-82fbc880-982d-11e9-8c66-ee0109c5368d.jpg">
</p>

## Run in Real-Time!
To run in real-time simply clone the repository and download the weights file and then run the ```real-time.py``` file. 
```
directory > python real-time.py
```
In real-time execution, there are two stages. In the first stage, the hand is detected using single object localization (SOLO) algorithm 
which is a simple single object multi-class object detection algorithm primarily published in my [Fingertip Mixed Reality](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality) repository. The detected hand portion is then cropped and fed to the second stage for gesture recognition and fingertip detection. 

## Output
Here is the output of the unified gesture recognition and fingertip detection model for all of the classes of the dataset where 
not only each fingertip is detected but also each finger is classified.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60171964-85f6b900-982d-11e9-8f20-af40be2172f8.jpg">
</p>
