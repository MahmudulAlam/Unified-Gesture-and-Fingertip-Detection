import cv2
import time
import numpy as np
from statistics import mean
from hand_detector.detector import YOLO
from unified_detector import Fingertips

images = np.load('../dataset/test/images.npy')
test_x = np.load('../dataset/test/test_x.npy')
test_y_prob = np.load('../dataset/test/test_y_prob.npy')
test_y_keys = np.load('../dataset/test/test_y_keys.npy')
crop_info = np.load('../dataset/test/crop_info.npy')

hand_model = YOLO(weights='../weights/yolo.h5', threshold=0.5)
fingertips = Fingertips(weights='../weights/fingertip.h5')

# classification
ground_truth_class = np.array([0, 0, 0, 0, 0, 0, 0, 0])
prediction_class = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# regression
fingertip_err = np.array([0, 0, 0, 0, 0, 0, 0, 0])
avg_time = 0
iteration = 0
conf_mat = np.zeros(shape=(8, 8))
pr_prob_per_yolo = []  # prediction of probability performance using yolo
pr_pos_per_yolo = []  # prediction of position performance using yolo

for n_image, (info, image, cropped_image, gt_prob, gt_pos) in enumerate(zip(crop_info, images, test_x,
                                                                            test_y_prob, test_y_keys), 1):
    print('Images: ', n_image)

    tl = [info[0], info[1]]
    height, width = info[2], info[3]

    top_left, bottom_right = hand_model.detect(image)
    if top_left or bottom_right is not None:
        x1, y1, x2, y2 = int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])
        cropped_image = image[y1:y2, x1:x2]
        height, width, _ = cropped_image.shape

        """ Predictions """
        tic = time.time()
        prob, pos = fingertips.classify(image=cropped_image)
        pos = np.mean(pos, 0)

        """ Post processing """
        threshold = 0.5
        prob = np.asarray([(p >= threshold) * 1.0 for p in prob])

        for i in range(0, len(gt_pos), 2):
            gt_pos[i] = gt_pos[i] * width / 128. + tl[0]
            gt_pos[i + 1] = gt_pos[i + 1] * height / 128. + tl[1]

        for i in range(0, len(pos), 2):
            pos[i] = pos[i] * width + top_left[0]
            pos[i + 1] = pos[i + 1] * height + top_left[1]

        toc = time.time()
        avg_time = avg_time + (toc - tic)
        iteration = iteration + 1

        """ Calculations """
        # classification
        gt_cls = fingertips.class_finder(prob=gt_prob)
        pred_cls = fingertips.class_finder(prob=prob)
        ground_truth_class[gt_cls] = ground_truth_class[gt_cls] + 1

        if gt_cls == pred_cls:
            prediction_class[pred_cls] = prediction_class[pred_cls] + 1

            # Regression
            squared_diff = np.square(gt_pos - pos)
            error = 0
            for i in range(0, 5):
                if prob[i] == 1:
                    error = error + np.sqrt(squared_diff[2 * i] + squared_diff[2 * i + 1])
            error = error / sum(prob)
            fingertip_err[pred_cls] = fingertip_err[pred_cls] + error

        conf_mat[gt_cls, pred_cls] = conf_mat[gt_cls, pred_cls] + 1
        pr_prob_per_yolo.append(prob)
        pr_pos_per_yolo.append(pos)

        """ Drawing finger tips """
        index = 0
        color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
        for c, p in enumerate(prob):
            if p == 1:
                image = cv2.circle(image, (int(pos[index]), int(pos[index + 1])), radius=12,
                                   color=color[c], thickness=-2)
            index = index + 2

    cv2.imshow('', image)
    cv2.waitKey(0)
    # cv2.imwrite('output_perform/' + image_name, image)

accuracy = prediction_class / ground_truth_class
accuracy = accuracy * 100
accuracy = np.round(accuracy, 2)
avg_time = avg_time / iteration
fingertip_err = fingertip_err / prediction_class
fingertip_err = np.round(fingertip_err, 4)
np.save('../data/conf_mat_yolo.npy', conf_mat)

print(prediction_class)
print(ground_truth_class)

print('Accuracy: ', accuracy, '%')
print('Fingertip detection error: ', fingertip_err, ' pixels')
print('Mean Error: ', mean(fingertip_err), ' pixels')
print('Total Iteration: ', iteration)
print('Mean Execution Time: ', round(avg_time, 4))

pr_prob_per_yolo = np.asarray(pr_prob_per_yolo)
pr_pos_per_yolo = np.asarray(pr_pos_per_yolo)
np.save('../data/yolo/pr_prob_per_yolo.npy', pr_prob_per_yolo)
np.save('../data/yolo/pr_pos_per_yolo.npy', pr_pos_per_yolo)
