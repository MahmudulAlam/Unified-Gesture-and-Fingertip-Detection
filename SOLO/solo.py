import cv2
import numpy as np
from SOLO.model import model
from SOLO.preprocess.flag import flag

grid_size = flag().grid_size


class Detector(object):
    def __init__(self, weights, threshold):
        self.model = model()
        self.threshold = threshold
        self.model.load_weights(weights)

    def detect(self, image):
        ori_image = image
        height, width, _ = ori_image.shape
        image = cv2.resize(ori_image, (416, 416))
        img = image / 255.0
        img = np.expand_dims(img, axis=0)
        grid_output = self.model.predict(img)
        grid_output = grid_output[0]
        output = (grid_output > self.threshold).astype(int)

        """ Finding bounding box """
        prediction = np.where(output > self.threshold)
        row_wise = prediction[0]
        col_wise = prediction[1]
        try:
            x1 = min(col_wise) * grid_size
            y1 = min(row_wise) * grid_size
            x2 = (max(col_wise) + 1) * grid_size
            y2 = (max(row_wise) + 1) * grid_size
            """ size conversion """
            x1 = int(x1 / 416 * width)
            y1 = int(y1 / 416 * height)
            x2 = int(x2 / 416 * width)
            y2 = int(y2 / 416 * height)
            return (x1, y1), (x2, y2)

        except ValueError:
            print('NO Hand Detected')
            return None, None
