import cv2
from preprocess.datagen import label_generator


def visualize(image, prob, key):
    index = 0
    color = [(15, 15, 240), (15, 240, 155), (240, 155, 15), (240, 15, 155), (240, 15, 240)]
    for c, p in enumerate(prob):
        if p > 0.5:
            image = cv2.circle(image, (int(key[index]), int(key[index + 1])), radius=5,
                               color=color[c], thickness=-2)
        index = index + 2

    cv2.imshow('Visualize', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_name = 'ChuangyeguFirstfloor_Single_Five_color_514.jpg'
    image, probability, position = label_generator(image_name=image_name, type='Test')
    visualize(image, probability, position)
