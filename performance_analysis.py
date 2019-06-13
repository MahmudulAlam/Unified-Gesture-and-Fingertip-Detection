import numpy as np

""" Ground Truth Outputs """
gt_probability = np.load('data/performance/gt_prob_per.npy')

""" Predicted Outputs """
pr_probability = np.load('data/performance/pr_prob_per.npy')


def class_finder(prob):
    cls = None
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


class_number = [0, 1, 2, 3, 4, 5, 6, 7]
accuracy = [[], [], [], [], [], [], [], []]
precision = [[], [], [], [], [], [], [], []]
recall = [[], [], [], [], [], [], [], []]
f1_score = [[], [], [], [], [], [], [], []]

threshold_values = [0.5]
for threshold in threshold_values:
    print('Threshold: ', threshold)

    true_positive = [0, 0, 0, 0, 0, 0, 0, 0]
    false_positive = [0, 0, 0, 0, 0, 0, 0, 0]
    false_negative = [0, 0, 0, 0, 0, 0, 0, 0]
    true_negative = [0, 0, 0, 0, 0, 0, 0, 0]

    for id in class_number:
        for gt_prob, pr_prob in zip(gt_probability, pr_probability):
            prob = np.asarray([(p >= threshold) * 1.0 for p in pr_prob])
            gt_class = class_finder(gt_prob)
            pr_class = class_finder(prob)

            if gt_class == pr_class == id:
                true_positive[id] = true_positive[id] + 1
            else:
                if gt_class == id:
                    false_negative[id] = false_negative[id] + 1
                elif pr_class == id:
                    false_positive[id] = false_positive[id] + 1
                else:
                    true_negative[id] = true_negative[id] + 1

    print('True positive:', true_positive)
    print('False positive:', false_positive)
    print('False Negative:', false_negative)
    print('True Negative:', true_negative)

    for k in range(0, len(class_number)):
        try:
            p = true_positive[k] / (true_positive[k] + false_positive[k])
        except ZeroDivisionError:
            p = 0
        try:
            r = true_positive[k] / (true_positive[k] + false_negative[k])
        except ZeroDivisionError:
            r = 0
        precision[k].append(round(p, 6))
        recall[k].append(round(r, 6))
        f1 = 2 * ((p * r) / (p + r))
        f1_score[k].append(round(f1, 6))

    true_positive = np.array(true_positive)
    true_negative = np.array(true_negative)
    false_positive = np.array(false_positive)
    false_negative = np.array(false_negative)

    a = (true_positive + true_negative)
    b = (true_positive + true_negative + false_positive + false_negative)
    accuracy = a / b
    accuracy = accuracy * 100
    accuracy = np.round(accuracy, 4)

print('')
print('Accuracy =', accuracy)
print('precision =', precision)
print('recall =', recall)
print('F1 Score  =', f1_score)
