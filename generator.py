import random
import numpy as np
from visualize import visualize
from preprocess.augmentation import augment


def batch_indices(batch_size=None, dataset_size=None):
    index_a = list(range(0, dataset_size, batch_size))
    index_b = list(range(batch_size, dataset_size, batch_size))
    index_b.append(dataset_size)
    indices = list(zip(index_a, index_b))
    return indices


def train_generator(batch_size, is_augment=True, viz=False):
    if is_augment:
        batch_size = int(batch_size / 2)

    # load dataset
    directory = 'dataset/train/'

    train_x_full = np.load(directory + 'train_x.npy')
    train_y_prob_full = np.load(directory + 'train_y_prob.npy')
    train_y_keys_full = np.load(directory + 'train_y_keys.npy')

    dataset_size = train_y_prob_full.shape[0]
    indices = batch_indices(batch_size=batch_size, dataset_size=dataset_size)
    print('Training Dataset Size: {0}'.format(dataset_size))

    while True:

        for index in indices:
            # load from dataset
            train_x = train_x_full[index[0]:index[1]]
            train_y_prob = train_y_prob_full[index[0]:index[1]]
            train_y_keys = train_y_keys_full[index[0]:index[1]]

            if viz:
                visualize(train_x[-1], train_y_prob[-1], train_y_keys[-1])

            # augment dataset and append to the batch
            train_x_aug, train_y_keys_aug = augment(train_x, train_y_prob, train_y_keys)
            train_x = np.append(train_x, train_x_aug, axis=0)
            train_y_prob = np.append(train_y_prob, train_y_prob, axis=0)
            train_y_keys = np.append(train_y_keys, train_y_keys_aug, axis=0)

            if viz:
                visualize(train_x[-1], train_y_prob[-1], train_y_keys[-1])

            # normalizing the image and the keypoints
            train_x = train_x / 255.0
            train_y_prob = np.squeeze(train_y_prob)
            train_y_keys = train_y_keys / 128.0

            # creating ensembles of the keypoints
            train_y_keys = np.reshape(train_y_keys, (train_y_keys.shape[0], 1, 10))
            train_y_keys = np.repeat(train_y_keys, 10, axis=1)

            # random shuffling over the batch
            seed = random.randint(0, 1000)
            np.random.seed(seed)
            np.random.shuffle(train_x)
            np.random.seed(seed)
            np.random.shuffle(train_y_prob)
            np.random.seed(seed)
            np.random.shuffle(train_y_keys)

            train_y = [train_y_prob, train_y_keys]
            yield train_x, train_y


def valid_generator(batch_size, viz=False):
    directory = 'dataset/valid/'

    valid_x_full = np.load(directory + 'valid_x.npy')
    valid_y_prob_full = np.load(directory + 'valid_y_prob.npy')
    valid_y_keys_full = np.load(directory + 'valid_y_keys.npy')

    dataset_size = valid_y_prob_full.shape[0]
    indices = batch_indices(batch_size=batch_size, dataset_size=dataset_size)
    print('Validation  Dataset Size: {0}'.format(dataset_size))

    while True:

        for index in indices:
            # load from dataset
            valid_x = valid_x_full[index[0]:index[1]]
            valid_y_prob = valid_y_prob_full[index[0]:index[1]]
            valid_y_keys = valid_y_keys_full[index[0]:index[1]]

            if viz:
                visualize(valid_x[-1], valid_y_prob[-1], valid_y_keys[-1])

            # normalizing the image and the keypoints
            valid_x = valid_x / 255.0
            valid_y_prob = np.squeeze(valid_y_prob)
            valid_y_keys = valid_y_keys / 128.0

            # creating ensembles of the keypoints
            valid_y_keys = np.reshape(valid_y_keys, (valid_y_keys.shape[0], 1, 10))
            valid_y_keys = np.repeat(valid_y_keys, 10, axis=1)

            valid_y = [valid_y_prob, valid_y_keys]
            yield valid_x, valid_y


if __name__ == '__main__':
    gen = valid_generator(batch_size=32, viz=True)
    x_batch, y_batch = next(gen)

    print(x_batch.shape)
    print(y_batch[0].shape)
    print(y_batch[1].shape)

    visualize(x_batch[0] * 255., y_batch[0][0], y_batch[1][0][0] * 128.)
