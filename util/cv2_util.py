import cv2
import numpy as np


def show_cv2_image(images, names, wait_for_input=True):
    resize = 13
    for (img, name) in zip(images, names):
        dims = (int(img.shape[1] * resize/100), int(img.shape[0] * resize/100))
        img = cv2.resize(img, dims)
        cv2.imshow(name, img)
    if wait_for_input:
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()


def draw_rad_lines(img, lines):
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x1 = 100
        x2 = img.shape[1] - 100
        y1 = int((rho - x1*a) / b)
        y2 = int((rho - x2*a) / b)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('lines.png', img)


def rad_to_rotation(rad):
    return rad * 180/np.pi - 90


def correct_image_rotation(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    # Hough Transform for line detection, with precision for \ro and \theta at 1 pixel and 180 degrees, respectively
    # Theta is bound for horizontal lines (90 degrees ~= 1.57 rads)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, int(img.shape[1] * 0.15), min_theta=1.45, max_theta=1.69)
    lines = lines.reshape((lines.shape[0], 2))
    median_rad_angle = np.median(lines[:, 1])
    median_angle = rad_to_rotation(median_rad_angle)

    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected_img = cv2.warpAffine(img, rot_matrix, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    return corrected_img, median_angle


def invert_and_threshold(img):
    inverted = cv2.bitwise_not(img)
    blurred = cv2.GaussianBlur(inverted, (9, 9), 0)
    _, bw = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def find_horizontal_lines(img):
    """
    Assumes inverted white on black binary image. Use `invert_and_threshold` to obtain it.
    :param img:
    :return:
    """
    line_img = np.copy(img)
    cols = line_img.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // 30, 1))
    line_img = cv2.erode(line_img, kernel)
    line_img = cv2.dilate(line_img, kernel)

    return line_img


def find_vertical_lines(img):
    """
    Assumes inverted white on black binary image. Use `invert_and_threshold` to obtain it.
    :param img:
    :return:
    """
    line_img = np.copy(img)
    rows = line_img.shape[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 30))
    line_img = cv2.morphologyEx(line_img, cv2.MORPH_OPEN, kernel)

    return line_img


def find_vertical_lines_subtracted(img):
    """
    Assumes inverted white on black binary image. Use `invert_and_threshold` to obtain it.
    :param img:
    :return:
    """
    line_img = np.copy(img)
    rows = line_img.shape[0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // 30))

    hor_lines = find_horizontal_lines(line_img)
    line_img = cv2.subtract(line_img, hor_lines)
    line_img = cv2.GaussianBlur(line_img, (11, 11), 0)
    line_img = cv2.morphologyEx(line_img, cv2.MORPH_OPEN, kernel)

    return line_img
