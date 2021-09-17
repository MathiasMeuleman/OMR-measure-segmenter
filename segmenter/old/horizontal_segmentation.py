import cv2
import numpy as np
import matplotlib.pyplot as plt
from util.dirs import data_dir


def auto_skew_image(_img, draw=False):
    print("Deskewing image...")
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

    _img = cv2.bitwise_not(_img)
    thres = cv2.threshold(_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thres > 0))

    # edges = cv2.Canny(bw_img, 100, 200)
    # show_cv2_image([edges], ['edges'], True)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/80, 100, 100, 10)
    # line_img = overlay_lines(_img.copy(), lines)
    # coords = np.column_stack(np.where(line_img > 0))

    _area = cv2.minAreaRect(coords)
    area = ((_area[0][1], _area[0][0]), (_area[1][1], _area[1][0]), _area[2])
    # if draw:
    #     box = np.int0(cv2.boxPoints(area))
    #     cv2.drawContours(_img, [box], 0, (255, 0, 0), 3)
    #     show_cv2_image([_img], ['box'])

    angle = area[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = _img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, M


def load_and_deskew(page):
    path = data_dir / 'Beethoven_Septett/ppm-300'
    input_path = path / 'transcript-{}.png'.format(page)
    img = cv2.imread(input_path)
    deskewed, M = auto_skew_image(img.copy(), draw=False)
    cv2.imwrite(path / 'transcript-{}-deskewed.png'.format(page), deskewed)
    return deskewed


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def horizontal(pages):
    f, axes = plt.subplots(3, 1, figsize=(14, 17))
    for i, idx in enumerate(pages):
        page = 76 + idx
        print("Processing page {}".format(page))
        img = load_and_deskew(page)
        profile = img.sum(axis=1) / 255
        peakranges = consecutive(np.where(profile > profile.mean() + profile.std())[0], stepsize=60)
        peaks = np.asarray([np.round(np.median(peak)) for peak in peakranges])
        cuts = (peaks[1:] + peaks[:-1]) / 2
        cuts = np.roll(np.append(cuts, (cuts[-1] + (cuts[-1] - cuts[-2]), cuts[0] - (cuts[1] - cuts[0]))), 1)
        axes[i].bar(range(len(profile)), profile, 1, alpha=0.2, color='b')
        # for peak in peaks:
        #     axes[i].axvline(peak, color='r')
        for cut in cuts:
            axes[i].axvline(cut, color='r')
        axes[i].axhline(profile.mean() + profile.std())
        axes[i].set_title("Page {}".format(page))
    plt.show()


if __name__ == "__main__":
    for k in range(5):
        horizontal([k * 3, k * 3 + 1, k * 3 + 2])

