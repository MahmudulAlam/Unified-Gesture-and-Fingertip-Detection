import cv2
import numpy as np
from scipy import ndimage
from SOLO.model import model
from SOLO.visualize import visualize
from SOLO.preprocess.find_folder import find_folder

model = model()
model.load_weights('../weights/solo.h5')

""" pre-processing """
dataset_directory = '../../EgoGesture Dataset/'
image_name = 'ComputerScreen_Single_One_color_250.jpg'
folder_name = find_folder(image_name)
image = cv2.imread(dataset_directory + '/' + folder_name + '/' + image_name)
image = cv2.resize(image, (416, 416))
img = image / 255.0
img = np.expand_dims(img, axis=0)
grid_output = model.predict(img)
grid_output = grid_output[0]
grid_output = (grid_output > 0.5).astype(int)
print(grid_output)

blob, nBlob = ndimage.label(grid_output)

try:
    biggest_blob = np.bincount(blob.flat)[1:].argmax() + 1
    grid_output = (blob == biggest_blob).astype(int)
except ValueError:
    pass
visualize(image=image, output=grid_output)
