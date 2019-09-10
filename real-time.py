import cv2
import numpy as np
from net.network import model
from SOLO.solo import Detector


""" SOLO Hand Detection """
detect_hand = Detector(weights='weights/solo.h5', threshold=0.8)

""" Key points Detection """
model = model()
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


cam = cv2.VideoCapture(0)

while True:
    """ capturing image """
    ret, image = cam.read()

    if ret is False:
        break

    """ if hand is detected """
    tl, br = detect_hand.detect(image=image)
    if tl and br is not None:
        cropped_image = image[tl[1]:br[1], tl[0]: br[0]]
        height, width, _ = cropped_image.shape

        """ fingertips detection """
        prob, pos = classify(image=cropped_image)
        pos = np.mean(pos, 0)

        """ Post processing """
        prob = np.asarray([(p >= 0.5) * 1.0 for p in prob])
        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + tl[0]
            pos[i + 1] = pos[i + 1] * height + tl[1]

        """ drawing """
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (0, 255, 0), 2)
        for c, p in enumerate(prob):
            if p > 0.5:
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
            index = index + 2

    if cv2.waitKey(1) & 0xff == 27:
        break

    """ displaying image """
    cv2.imshow('real-time hand & fingertip detection', image)

cam.release()
cv2.destroyAllWindows()
