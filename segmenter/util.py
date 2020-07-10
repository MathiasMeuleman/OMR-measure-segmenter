import math

from PIL import Image
from scipy import ndimage

import cv2
import numpy as np

from segmenter.image_util import resize_img


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


def preprocess_minrect(_img):
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


def preprocess(img):
    """
    Preprocess the image in a few steps:
      - Use Otsu thresholding to binarize the image
      - Invert the image, for easier processing
      - Rotate the image to a correct alignment. The median angle for detected Hough lines is taken as the rotation angle
    :param img: The original image, as NumPy array
    :return: The binarized, inverted, rotated image, as NumPy image
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)[1]
    img_bw = cv2.bitwise_not(img_bw)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        cv2.line(img_gray, (x1, y1), (x2, y2), (255, 0, 0), 3)
    median_angle = np.median(angles)
    rotated = ndimage.rotate(img_bw, median_angle)
    return rotated
