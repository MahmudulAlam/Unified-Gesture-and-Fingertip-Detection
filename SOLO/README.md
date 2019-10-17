## SOLO
SOLO stands for [```single object localization```](https://github.com/MahmudulAlam/Fingertip-Mixed-Reality/tree/master/SOLO)
which has been developed for fast and efficient multi-class single object detection.
In this case, for the purpose of localizing hand for gesture recognition and fingertip detection, it is employed for the purpose of
hand detection. SOLO divides the input image into grid cells and predicts the probability of having an object in each grid cell. 
Each grid cell is mapped in the output matrix of the CNN using binary representation. Grid cells that have at least ```50%```
overlapping with the hand bounding box are labeled as binary :one: and the rest of the cells are labeled as binary :zero:


<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/65219825-3b5aa500-dadb-11e9-92f3-a1f75eef1b5a.jpg" width="550">
</p>

## Network Architecture
For prediction, a fully convolutional network (FCN) is designed based on the common knowledge of the field. The network architecture is
inspired by the visual geometry group (VGG) model for image classification. 

```
input = Input(shape=(416, 416, 3))

# Block 01
x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(input)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

# Block 02
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

# Block 03
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

# Block 04
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

# Block 05
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(1, (1, 1), activation='sigmoid')(x)
output = Reshape((13, 13), name='output')(x)

model = Model(input, output)
```

Here, in each of the convolutional layer ```3 x 3``` filters are used and followed by a rectified linear unit (ReLU) activation 
function except for the final convolutional layer where ```1 x 1``` filter is used to keep the output size same as the desired output
size and the sigmoid activation function is applied for normalized output.

## Loss Function
To train the network, following binary cross-entropy loss function is defined where ![H](https://latex.codecogs.com/gif.latex?%7B%5Cmathbb%7BH%7D%7D)
and ![H^hat](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cmathbb%7BH%7D%7D) are the ground truth and the predicted
output matrix of the SOLO. Here, N and M represent the length of the output matrix and the batch size respectively.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/65953360-9be7cb80-e465-11e9-8e9a-838d9cc75b1c.jpg" width="600">
</p>

The loss function is optimized using the ADAM optimizer with standard hyperparameter values. 

## Output of SOLO and Generalized Learning
Although the SOLO algorithm is trained using the images having hand in each of them, the algorithm is able to avoid predicting any hand 
in the images where the hand is absent which shows the generalization in the learning of the grid-wise prediction. The following figure 
shows the outputs of the SOLO algorithm detecting hand in the image of the test set shown in (a) and not detecting any hand in the image 
where the hand is absent in the egocentric vision shown in (b) indicating the generalized learning of the algorithm.

<p align="center">
  <img src="https://user-images.githubusercontent.com/37298971/66996135-fe8dc800-f0f1-11e9-8a46-76c88ec5db4b.png" width="600">
</p>
