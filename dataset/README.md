## Dataset Description 
In the experimentation, [SCUT-Ego-Gesture-Dataset](http://www.hcii-lab.net/data/SCUTEgoGesture/index.htm) is employed. The database 
contains a total of eleven different single hand gesture datasets. Among the eleven different gesture datasets, eight of them is utilized
here for gesture recognition and fingertip detection. The eight datasets include 29; 337 RGB hand images in the egocentric vision of
resolution 640 x 480. The dataset is partitioned in the following order. 

* First, 10% images from the full database is taken for the test set by randomly sampling one every ten images. 
* Then, 5% images from the remaining database is taken for the validation set by randomly sampling one every twenty images. 
* Finally, the rest of the images of the database is employed for the training set. 

The list of the images used in the test, validation, and training set is provided in the corresponding folders in the ```.txt``` files of each 
class. The label folder contains all the ground truths labels of the images of each class. The images can be downloaded from the [dataset](http://www.hcii-lab.net/data/SCUTEgoGesture/index.htm)
website, however, the zip file is password protected and to get the password, fill their application form and send an email to them.

The following figure shows the example images of the dataset of each class. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/59550487-02b3a880-8f8d-11e9-9e22-b84689f0ca41.png" width="640">
</p>
