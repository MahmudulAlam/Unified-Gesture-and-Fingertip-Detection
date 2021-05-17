import numpy as np

keys = np.load('test/test_y_keys.npy')
crop = np.load('test/crop_info.npy')
key_list = []


for key, crop_info in zip(keys, crop):
    top_x, top_y = crop_info[0], crop_info[1]
    height, width = crop_info[2], crop_info[3]

    # normalize keypoints for the cropped image
    for i in range(0, len(key), 2):
        key[i] = (key[i] - top_x) / width * 128
        key[i + 1] = (key[i + 1] - top_y) / height * 128

    key_list.append(key)

key_list = np.asarray(key_list, dtype=np.float32)
np.save('test_y_keys.npy', key_list)
