import cv2
import numpy as np


def visualize(image, prob, key):
    index = 0

    color = [(120, 20, 240), (240, 55, 210), (240, 55, 140), (240, 75, 55), (170, 240, 55)]
    for c, p in enumerate(prob):
        if p > 0.5:
            print(key[index])
            image = cv2.circle(image, (int(key[index]), int(key[index + 1])), radius=5, color=color[c], thickness=-2)
        index = index + 2

    cv2.imshow("Press 'Esc' to CLOSE", image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return


directory = 'train/'
dataset = 'train'

x_full = np.load(directory + dataset + '_x.npy')
y_prob_full = np.load(directory + dataset + '_y_prob.npy')
y_keys_full = np.load(directory + dataset + '_y_keys.npy')

# dataset info
print(dataset, ':')
print(x_full.shape, x_full.dtype)
print(y_prob_full.shape, y_prob_full.dtype)
print(y_keys_full.shape, y_keys_full.dtype)

sample_number = 10

x_sample = x_full[sample_number]
y_prob_sample = y_prob_full[sample_number]
y_keys_sample = y_keys_full[sample_number]

visualize(image=x_sample,
          prob=y_prob_sample,
          key=y_keys_sample)
