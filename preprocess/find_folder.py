""" Finding the folder name from image name """
from preprocess.flag import Finger


def find_folder(image_name):
    name = image_name.split('_')
    if 'One' in name:
        folder_name = 'SingleOne'
    elif 'Two' in name:
        folder_name = 'SingleTwo'
    elif 'Three' in name:
        folder_name = 'SingleThree'
    elif 'Four' in name:
        folder_name = 'SingleFour'
    elif 'Five' in name:
        folder_name = 'SingleFive'
    elif 'Six' in name:
        folder_name = 'SingleSix'
    elif 'Seven' in name:
        folder_name = 'SingleSeven'
    elif 'Eight' in name:
        folder_name = 'SingleEight'
    elif 'Nine' in name:
        folder_name = 'SingleNine'
    elif 'Good' in name:
        folder_name = 'SingleGood'
    elif 'Bad' in name:
        folder_name = 'SingleBad'
    else:
        folder_name = 'SingleNone'
    return folder_name


def finger_type(image_name):
    name = image_name.split('_')
    finger = None
    if 'One' in name:
        finger = Finger().SingleOne()
    elif 'Two' in name:
        finger = Finger().SingleTwo()
    elif 'Three' in name:
        finger = Finger().SingleThree()
    elif 'Four' in name:
        finger = Finger().SingleFour()
    elif 'Five' in name:
        finger = Finger().SingleFive()
    elif 'Six' in name:
        finger = Finger().SingleSix()
    elif 'Seven' in name:
        finger = Finger().SingleSeven()
    elif 'Eight' in name:
        finger = Finger().SingleEight()
    elif 'Nine' in name:
        finger = Finger().SingleNine()
    elif 'Good' in name:
        finger = Finger().SingleGood()
    elif 'Bad' in name:
        finger = Finger().SingleBad()
    return finger
