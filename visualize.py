import cv2
import numpy as np
from preprocess.data_generator import label_generator


def visualize(img, prob, key):
    index = 0

    # preprocess
    img = np.asarray(img, dtype=np.uint8)
    prob = prob.squeeze()
    key = key.squeeze()

    color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
    for c, p in enumerate(prob):
        if p > 0.5:
            img = cv2.circle(img, (int(key[index]), int(key[index + 1])), radius=5, color=color[c], thickness=-2)
        index = index + 2

    cv2.imshow('Unified Gesture & Fingertips Detection', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    image, probability, keypoints = label_generator(directory='./dataset/', dtype='train', sample=0)
    visualize(img=image, prob=probability, key=keypoints)
