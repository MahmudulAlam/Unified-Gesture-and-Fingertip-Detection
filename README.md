## Unified Gesture Recognition and Fingertip Detection 👋
[![GitHub stars](https://img.shields.io/github/stars/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/network)
[![GitHub issues](https://img.shields.io/github/issues/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/issues)
[![Version](https://img.shields.io/badge/version-2.0-orange.svg?longCache=true&style=flat)](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality)
[![GitHub license](https://img.shields.io/github/license/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/blob/master/LICENSE)
<img src="https://user-images.githubusercontent.com/37298971/123714340-f8d70800-d82a-11eb-9742-042a5d9334a1.png" width="28">

<p align="justify">
  A unified convolutional neural network (CNN) algorithm for both hand gesture recognition and fingertip detection at the same time. The proposed algorithm uses a single network to predict both finger class probabilities for classification and fingertips positional output for regression in one single evaluation. From the finger class probabilities, the gesture is recognized, and using both of the information fingertips are localized. Instead of directly regressing the fingertips position from the fully connected (FC) layer of the CNN, we regress an ensemble of fingertips position from a fully convolutional network (FCN) and subsequently take ensemble average to regress the final fingertips positional output.
</p>

## Update 🔥
Included ```robust real-time hand detection using yolo``` for better smooth performance in the first stage of the detection system and most of the code has been cleaned and restructured for ease of use. To get the previous versions, please visit the release [section](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/releases).

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/77615813-6de9cc80-6f5a-11ea-9172-a95e5604147c.gif" width="400">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/72676259-5f45eb80-3ab9-11ea-96d7-436f160a4b84.png" width="600">
</p>

## Requirements 🐍
- TensorFlow-GPU==2.2.0  ```pip install tensorflow-gpu==2.2.0```
- OpenCV==4.2.0  ```pip install opencv-python==4.2.0```
- ImgAug==0.2.6  ```pip install imgaug==0.2.6```
- Weights: [```Download the pre-trained weights```](https://mega.nz/#F!6stCxY5b!oB-3279KkhfhRULQFQO7yQ) files of the unified gesture recognition and fingertip detection model and put the ```weights/``` folder in the working directory.

[![Downloads](https://img.shields.io/badge/download-weights-green.svg?style=popout-flat&logo=mega)](https://mega.nz/#F!6stCxY5b!oB-3279KkhfhRULQFQO7yQ)
[![Downloads](https://img.shields.io/badge/download-weights-blue.svg?style=popout-flat&logo=dropbox)](https://www.dropbox.com/sh/7pbfrgaor678eft/AAA8r5ADlMde0WkAtJQO_lo5a?dl=0)

The ```weights/``` folder contains three weights files. The ```fingertip.h5``` is for unified gesture recognition and fingertip detection. ```yolo.h5``` and ```solo.h5``` are for the yolo and solo method of hand detection. [(what is solo?)](https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/tree/master/hand_detector/solo)

## Paper 📚
[![Paper](https://img.shields.io/badge/paper-ScienceDirect-ff6c00.svg?longCache=true&style=flat)](https://doi.org/10.1016/j.patcog.2021.108200)
[![Paper](https://img.shields.io/badge/paper-ArXiv-ff0a0a.svg?longCache=true&style=flat)](https://arxiv.org/abs/2101.02047)
<img src="https://user-images.githubusercontent.com/37298971/141495879-3b3d4ba7-955a-4c1c-bf55-24af319da78d.png" width="128">

To get more information about the proposed method and experiments, please go through the [```Elsevier```](https://www.sciencedirect.com/science/article/abs/pii/S0031320321003824) or [```ArXiv```](https://arxiv.org/abs/2101.02047) version of the paper. Cite the paper as: 

```bibtex
@article{alam2022unified,
  title={Unified learning approach for egocentric hand gesture recognition and fingertip detection},
  author={Alam, Mohammad Mahmudul and Islam, Mohammad Tariqul and Rahman, SM Mahbubur},
  journal={Pattern Recognition},
  volume={121},
  pages={108200},
  year={2022},
  publisher={Elsevier}
}
```

## Dataset :card_index_dividers:
[![Dataset](https://img.shields.io/badge/download-dataset-bf1bfa.svg?longCache=true&style=flat)](https://drive.google.com/drive/folders/14LIb1Zj5yXgmvbifwn5A1fiaiL0r-RRh?usp=sharing)

<p align="justify">
  The proposed gesture recognition and fingertip detection model is trained by employing <a href="https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w11/Wu_YOLSE_Egocentric_Fingertip_ICCV_2017_paper.pdf">Scut-Ego-Gesture Dataset</a> which has a total of eleven different single hand gesture datasets. Among the eleven different gesture datasets, eight of them are considered for experimentation. A detailed explanation about the partition of the dataset along with the list of the images used in the training, validation, and the test set is provided in the <a href="https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection/tree/master/dataset#dataset-description">dataset/</a> folder. You can <a href="https://drive.google.com/drive/folders/14LIb1Zj5yXgmvbifwn5A1fiaiL0r-RRh?usp=sharing">download</a> 💾:file_folder: the pre-processed dataset that was used for experimentation. The shared folder also contains two python scripts to load the dataset.
</p>

## Network Architecture 🕸️
To implement the algorithm, the following network architecture is proposed where a single CNN is utilized for both hand gesture recognition and fingertip detection. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60171959-82fbc880-982d-11e9-8c66-ee0109c5368d.jpg">
</p>

## Prediction 👽
To get the prediction on a single image run the ```predict.py``` file. It will run the prediction in the sample image stored in the ```data/``` folder. Here is the output for the ```sample.jpg``` image. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/77616112-139d3b80-6f5b-11ea-81f0-977d50d44c4e.jpg" width="350">
</p>

## Real-Time! :camera_flash:
To run in real-time simply clone the repository and download the weights file and then run the ```real-time.py``` file. 
```python
directory > python real-time.py
```
In real-time execution, there are two stages. In the first stage, the hand can be detected by using either ```you only look once (yolo)``` or ```single object localization (solo)``` algorithm. By default, ```yolo``` will be used here. The detected hand portion is then cropped and fed to the second stage for gesture recognition and fingertip detection. 

## Output 🎨
Here is the output of the unified gesture recognition and fingertip detection model for all of the 8 classes of the dataset 
where not only each fingertip is detected but also each finger is classified.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/60171964-85f6b900-982d-11e9-8f20-af40be2172f8.jpg">
</p>

## Contact Me! 🐛📢🌎🚩🚧:mailbox_with_mail:
<p>
  If you have any queries or concerns, please feel free to <a href=https://mahmudulalam.github.io/#contact>contact me</a>.
</p>
