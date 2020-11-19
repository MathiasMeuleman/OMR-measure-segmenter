from pathlib import Path
from posixpath import join

import numpy as np
import cv2
import math
from PIL import Image, ImageDraw

root_dir = Path(__file__).parent.parent.absolute()
data_dir = join(root_dir, 'data')


def overlay_segments(page, img_source_dir=join(data_dir, 'ppm-600'), measures_source_dir=join(data_dir, 'ppm-600-segments')):
    image = Image.open(join(img_source_dir, 'transcript-{}.png'.format(page)))
    image = resize_img(image)
    measures = np.load(join(measures_source_dir, 'transcript-{}.npy'.format(page)))
    for measure in measures:
        draw = ImageDraw.Draw(image)
        ulx, uly, lrx, lry = measure
        draw.rectangle([ulx * image.size[0], uly * image.size[1], lrx * image.size[0], lry * image.size[1]], outline='red', fill=None)
        del draw
    image.save(join(root_dir, 'tmp/Mahler_Symphony_1/CNN-segmented-img/page-{}.png'.format(page)))


def split_image(page, img_source_dir=join(data_dir, 'ppm-600'), measures_source_dir=join(data_dir, 'ppm-600-segments'), save_dir=join(data_dir, 'ppm-600-measures')):
    image = Image.open(join(img_source_dir, 'transcript-{}.png'.format(page)))
    measures = np.load(join(measures_source_dir, 'transcript-{}.npy'.format(page)))
    for i, measure in enumerate(measures):
        measure_img = image.crop(measure * np.array((image.width, image.height, image.width, image.height)))
        measure_img.save(join(save_dir, 'measure-{}-{}.png'.format(page, i)))


def resize_img(_img, maxdims=(1000, 700)):
    """
    Resize a given image. Image can be either a Pillow Image, or a NumPy array. Resizing is done automatically such
    that the entire image fits inside the given maxdims box, keeping aspect ratio intact
    :param _img:
    :param maxdims:
    :return:
    """
    try:
        # If NumPy array, create Pillow Image
        img = Image.fromarray(_img)
    except TypeError:
        # Else image must already be a Pillow Image
        img = _img
    ratio = max(img.size[1] / maxdims[0], img.size[0] / maxdims[1])
    image = img.resize((int(img.size[0] / ratio), int(img.size[1] / ratio)), Image.ANTIALIAS)
    return image


def find_image_rotation(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []
    for [[x1, y1, x2, y2]] in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    return np.median(angles)
