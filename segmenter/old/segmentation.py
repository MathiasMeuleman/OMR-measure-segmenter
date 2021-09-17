import io
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu

from segmenter.old.image_util import resize_img
from util.cv2_util import preprocess
from util.dirs import data_dir


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


def show_PIL_image(image):
    resize = 13
    dims = (int(image.size[1] * resize/100), int(image.size[0] * resize/100))
    image = image.resize(dims, Image.ANTIALIAS)
    image.show()


def overlay_lines(_img, lines):
    line_img = np.zeros((_img.shape[0], _img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), [255, 0, 0], 3)
    _img = cv2.addWeighted(_img, 0.8, line_img, 1.0, 0.0)
    # show_cv2_image([_img], ['lines'], True)
    # cv2.imwrite('lines_78.png', _img)
    return cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)


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
    if draw:
        box = np.int0(cv2.boxPoints(area))
        cv2.drawContours(_img, [box], 0, (255, 0, 0), 3)
        show_cv2_image([_img], ['box'])

    angle = area[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    print(angle)
    (h, w) = _img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def get_img_from_fig(fig, size):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, pad_inches=0, bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    _img = cv2.imdecode(img_arr, 1)
    _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _img = cv2.resize(_img, size)

    return _img


def plot_intensity_histograms(_img, hor_hist, hor_thres, vert_hist, vert_thres):
    print("Plotting intensity histograms...")
    hor_fig = plt.figure()
    plt.axis('off')
    ax = plt.gca()
    ax.set_ylim(_img.shape[0], 0)
    plt.barh(range(len(hor_hist)), hor_hist, 1)
    plt.axvline(hor_thres, color='r')
    hor_data = get_img_from_fig(hor_fig, (1000, _img.shape[0]))

    vert_fig = plt.figure()
    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim(0, _img.shape[1])
    plt.bar(range(len(vert_hist)), vert_hist, 1)
    plt.axhline(vert_thres, color='r')
    vert_data = get_img_from_fig(vert_fig, (_img.shape[1], 1000))

    plt.close('all')
    return hor_data, vert_data


def calculate_intensity_histograms(_img):
    print("Get intensity histograms")
    hor_hist = _img.sum(1) / 255
    hor_thres = threshold_otsu(hor_hist)
    vert_hist = _img.sum(0) / 255
    vert_thres = threshold_otsu(vert_hist)
    return hor_hist, vert_hist, hor_thres, vert_thres


def show_img_with_hist(_img, horizontal_hist_plt, vertical_hist_plt, name):
    top = np.concatenate((np.zeros((1000, 1000), dtype=np.uint8) + 255, vertical_hist_plt), axis=1)
    bottom = np.concatenate((horizontal_hist_plt, _img), axis=1)
    img_with_hists = np.concatenate((top, bottom), axis=0)
    cv2.imwrite('data/intensities/{}.png'.format(name), img_with_hists)
    show_cv2_image([img_with_hists], ["Intensity histograms"], True)


def vertical_segmentation(_img, vert_hist, vthres):
    print("Executing vertical segmentation...")
    stepsize = 5
    vcut_candidates = np.argwhere(vert_hist > vthres).flatten()
    vcut_regions = np.split(vcut_candidates, np.where(np.diff(vcut_candidates) > stepsize)[0] + 1)
    vcuts = np.array(list(map(lambda region: int(median(region)), vcut_regions)))

    bbs = []
    for i in range(len(vcuts)):
        if i + 1 < len(vcuts):
            bbs.append([(vcuts[i]-50, ((i+1)%2)*50+10), (vcuts[i+1]+50, _img.shape[0] - (i%2)*50-10)])

    return bbs


def horizontal_segmentation(_img, hor_hist, hthres):
    print("Executing horizontal segmentation...")
    peak_stepsize = 5
    hcut_peak_candidates = np.argwhere(hor_hist > hthres).flatten()
    hcut_peak_regions = np.split(hcut_peak_candidates, np.where(np.diff(hcut_peak_candidates) > peak_stepsize)[0] + 1)
    hcut_peaks = np.array(list(map(lambda region: int(median(region)), hcut_peak_regions)))

    stepsize = int(np.average(np.diff(hcut_peaks)))
    hcut_regions = np.split(hcut_peaks, np.where(np.diff(hcut_peaks) > stepsize)[0] + 1)

    minima = [0]
    for i in range(len(hcut_regions)):
        if i + 1 < len(hcut_regions):
            boundaries = (hcut_regions[i][-1], hcut_regions[i+1][0])
            minima.append(boundaries[0] + np.argmin(hor_hist[boundaries[0]:boundaries[1]]))
    minima.append(_img.shape[0])

    bbs = []
    for i in range(len(minima)):
        if i + 1 < len(minima):
            bbs.append([(((i+1)%2)*50+10, minima[i]), (_img.shape[1] - (i%2)*50-10, minima[i+1])])
    print(bbs)
    return bbs, minima


def segment_measures(pages):
    for idx in pages:
        page = 76 + idx
        path = data_dir / 'Beethoven_Septett/ppm-300/transcript-{}.png'.format(page)
        img = cv2.imread(path)
        rotated = auto_skew_image(img.copy(), draw=False)

        # print(img.shape)
        # margins = np.array([np.max(np.where(img[i,:,1] != 255)) for i in np.arange(1000, 6000, 500)])
        # print(margins)
        # rotated_margins = np.array([np.max(np.where(rotated[i,:] != 255)) for i in np.arange(1000, 6000, 500)])
        # # rotated_margins = np.array([np.max(np.nonzero(rotated[i,:])) for i in np.arange(1000, 6000, 500)])
        # print(rotated_margins)

        # show_cv2_image([img, rotated], ['original', 'deskewed'], False)
        hor_hist, vert_hist, hthres, vthres = calculate_intensity_histograms(rotated)
        hor_data, vert_data = plot_intensity_histograms(rotated, hor_hist, hthres, vert_hist, vthres)
        show_img_with_hist(rotated, hor_data, vert_data, 'intensity_hists_{}'.format(page))
        # vertical_bbs = vertical_segmentation(rotated, vert_hist, vthres)
        # horizontal_bbs, minima = horizontal_segmentation(rotated, hor_hist, hthres)

        # img = Image.open(path)
        # for line in minima:
            # draw = ImageDraw.Draw(img)
            # draw.line((0, line, img.size[1], line), fill='blue', width=5)
            # del draw
        # show_PIL_image(img)
        # for box in vertical_bbs:
        #     draw = ImageDraw.Draw(img)
        #     draw.rectangle(box, outline='red', fill=None, width=5)
        #     del draw
        # for box in horizontal_bbs:
        #     draw = ImageDraw.Draw(img)
        #     draw.rectangle(box, outline='blue', fill=None, width=5)
        #     del draw
        # print('Saving page {}...'.format(page))
        # show_PIL_image(img)
        # img.save('segmented_page_v2_{}.png'.format(page))


def detect_peaks(image):
    image = preprocess_2(image)
    hor_hist, vert_hist, hthres, vthres = calculate_intensity_histograms(image)
    hor_data, vert_data = plot_intensity_histograms(image, hor_hist, hthres, vert_hist, vthres)
    show_img_with_hist(image, hor_data, vert_data, 'intensity_hists_{}'.format(page))
    profile = image.sum(axis=0) / 255
    peaks, props = find_peaks(profile,  distance=200, prominence=500)
    peaks = peaks / image.shape[1]
    image = resize_img(image)
    for peak in peaks:
        draw = ImageDraw.Draw(image)
        draw.line((peak * image.size[0], 0, peak * image.size[0], image.size[1]), fill='red')
        del draw
    image.show()


def detect_systems(image):
    profile = image.sum(axis=1) / 255
    zeros = np.diff(profile == 0).nonzero()[0]


if __name__ == "__main__":
    pages = [1, 76]
    for page in pages:
        img = Image.open(data_dir / 'ppm-600/transcript-{}.png'.format(page))
        img = preprocess(np.asarray(img))
        detect_systems(img)
