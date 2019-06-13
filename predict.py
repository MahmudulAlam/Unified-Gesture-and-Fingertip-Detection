import cv2
import numpy as np
from net.network import model
from preprocess.label_gen_test import label_generator_testset

image_name = 'ChuangyeguBusstop_Single_Seven_color_246.jpg'

""" Key points Detection """
model = model()
model.summary()
model.load_weights('weights/performance.h5')


def classify(image):
    image = np.asarray(image)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    probability, position = model.predict(image)
    probability = probability[0]
    position = position[0]
    return probability, position


def class_finder(prob):
    cls = ''
    # classes = ['SingleOne', 'SingleTwo', 'SingleThree', 'SingleFour', 'SingleFive',
    #            'SingleSix', 'SingleSeven', 'SingleEight']
    # index numbers
    classes = [0, 1, 2, 3, 4, 5, 6, 7]

    if np.array_equal(prob, np.array([0, 1, 0, 0, 0])):
        cls = classes[0]
    elif np.array_equal(prob, np.array([0, 1, 1, 0, 0])):
        cls = classes[1]
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 0])):
        cls = classes[2]
    elif np.array_equal(prob, np.array([0, 1, 1, 1, 1])):
        cls = classes[3]
    elif np.array_equal(prob, np.array([1, 1, 1, 1, 1])):
        cls = classes[4]
    elif np.array_equal(prob, np.array([1, 0, 0, 0, 1])):
        cls = classes[5]
    elif np.array_equal(prob, np.array([1, 1, 0, 0, 1])):
        cls = classes[6]
    elif np.array_equal(prob, np.array([1, 1, 0, 0, 0])):
        cls = classes[7]

    return cls


image, tl, cropped_image, ground_truths = label_generator_testset(image_name=image_name, initial='Test')
height, width, _ = cropped_image.shape
gt_prob, gt_pos = ground_truths

""" Predictions """
prob, pos = classify(image=cropped_image)
pos = np.mean(pos, 0)
print('prob: ', prob)
print('MEAN Position: ', pos)

""" Post processing """
prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
for i in range(0, len(pos), 2):
    pos[i] = pos[i] * width + tl[0]
    pos[i + 1] = pos[i + 1] * height + tl[1]

""" Calculations """
# Classification
gt_cls = class_finder(prob=gt_prob)
pred_cls = class_finder(prob=prob)
print('GT Class: ', gt_cls)

if gt_cls == pred_cls:
    # Regression
    squared_diff = np.square(gt_pos - pos)
    print('GT positions: ', gt_pos)
    print('PR positions: ', pos)
    print('SQ DIFF: ', squared_diff)

    error = 0
    for i in range(0, 5):
        if prob[i] == 1:
            value = np.sqrt(squared_diff[2 * i] + squared_diff[2 * i + 1])
            error = error + value
            print('ERROR: ', value)
    error = error / sum(prob)
    print('AVG ERROR: ', error)

# Drawing finger tips
index = 0
color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
for c, p in enumerate(prob):
    if p > 0.5:
        # ground truths
        image = cv2.circle(image, (int(gt_pos[index]), int(gt_pos[index + 1])), radius=12,
                           color=(0, 0, 0), thickness=-2)
        # predictions
        image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                           color=color[c], thickness=-2)

    index = index + 2

cv2.imshow('Prediction', image)
cv2.waitKey(0)
