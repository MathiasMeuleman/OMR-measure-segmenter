import cv2
import numpy as np


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def compare_measure_bounding_boxes(self, other):
    """Compares bounding boxes of two measures and returns which one should come first"""
    if self['ulx'] >= other['ulx'] and self['uly'] >= other['uly']:
        return +1  # self after other
    elif self['ulx'] < other['ulx'] and self['uly'] < other['uly']:
        return -1  # other after self
    else:
        overlap_y = min(self['lry'] - other['uly'], other['lry'] - self['uly']) \
                    / min(self['lry'] - self['uly'], other['lry'] - other['uly'])
        if overlap_y >= 0.5:
            if self['ulx'] < other['ulx']:
                return -1
            else:
                return 1
        else:
            if self['ulx'] < other['ulx']:
                return 1
            else:
                return -1


def preprocess(_img):
    """
    Takes a numpy array of (w x h x 3) representing the image.
    Binarizes the image (color 2 grayscale) and rotates the image such that its smallest rectangular bounding box
    is aligned correctly.
    :param _img:
    :return:
    """
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _img = cv2.bitwise_not(_img)
    thres = cv2.threshold(_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thres > 0))
    _area = cv2.minAreaRect(coords)
    area = ((_area[0][1], _area[0][0]), (_area[1][1], _area[1][0]), _area[2])

    angle = area[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = _img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

