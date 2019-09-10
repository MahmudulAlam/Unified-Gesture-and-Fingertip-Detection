""" Considering the presence of the finger in a dataset as True else False """


class Finger:
    def __int__(self):
        pass

    def SingleOne(self):
        thumb = False
        index = True
        middle = False
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky

    def SingleTwo(self):
        thumb = False
        index = True
        middle = True
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky

    def SingleThree(self):
        thumb = False
        index = True
        middle = True
        ring = True
        pinky = False
        return thumb, index, middle, ring, pinky

    def SingleFour(self):
        thumb = False
        index = True
        middle = True
        ring = True
        pinky = True
        return thumb, index, middle, ring, pinky

    def SingleFive(self):
        thumb = True
        index = True
        middle = True
        ring = True
        pinky = True
        return thumb, index, middle, ring, pinky

    def SingleSix(self):
        thumb = True
        index = False
        middle = False
        ring = False
        pinky = True
        return thumb, index, middle, ring, pinky

    def SingleSeven(self):
        thumb = True
        index = True
        middle = False
        ring = False
        pinky = True
        return thumb, index, middle, ring, pinky

    def SingleEight(self):
        thumb = True
        index = True
        middle = False
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky

    def SingleNine(self):
        thumb = False
        index = True
        middle = False
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky

    def SingleGood(self):
        thumb = True
        index = False
        middle = False
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky

    def SingleBad(self):
        thumb = True
        index = False
        middle = False
        ring = False
        pinky = False
        return thumb, index, middle, ring, pinky
