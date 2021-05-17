import numpy as np


def label_generator(directory, dtype='train', sample=0):
    x_full = np.load(directory + dtype + '/' + dtype + '_x.npy')
    y_prob_full = np.load(directory + dtype + '/' + dtype + '_y_prob.npy')
    y_keys_full = np.load(directory + dtype + '/' + dtype + '_y_keys.npy')

    image = x_full[sample]
    probability = y_prob_full[sample]
    keypoints = y_keys_full[sample]
    return image, probability, keypoints
