import cv2
import numpy as np
from SOLO.solo import Detector

""" Real-time SOLO Hand Recognition """
detect_hand = Detector(weights='SOLO/weights/solo.h5', threshold=0.8)
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

        """ drawing bounding box """
        image = cv2.rectangle(image, (tl[0], tl[1]), (br[0], br[1]), (0, 255, 0), 2)

    if cv2.waitKey(1) & 0xff == 27:
        break

    """ displaying image """
    cv2.imshow('Real-time Single Object Localization (SOLO)', image)

cam.release()
cv2.destroyAllWindows()
