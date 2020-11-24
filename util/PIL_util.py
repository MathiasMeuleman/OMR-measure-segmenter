import numpy as np
from PIL import Image


def binary_array_to_image(array):
    return Image.fromarray(np.uint8(array * 255), 'L')


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
