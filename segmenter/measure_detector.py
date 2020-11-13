import sys
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as im
import scipy.signal as sig
from PIL import Image, ImageDraw, ImageOps
from matplotlib import patches
from segmenter.util import get_hough_angle, get_minrect_angle

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)

############################################
# CLASSES
System = namedtuple("System", ["top", "bottom", "start", "end", "v_profile", "h_profile"])
Block = namedtuple("Block", ["start", "end", "system"])
Measure = namedtuple("Measure", ["top", "bottom", "block"])


############################################
# METHODS

# This method does some pre-processing on the pages
def open_and_preprocess(path):
    original = Image.fromarray(plt.imread(path)[:, :, 0]).convert('L')  # Convert to "luminance" (single-channel greyscale)
    img = ImageOps.autocontrast(original)                             # Does some automatic histogram stretching to enhance contrast
    img = ImageOps.invert(img)                                   # Inverts the image so we get white on black
    img = img.point(lambda x: x > 50)                           # Threshold on value 50, this will convert the range into [0, 1] (floats though!)
    return np.asarray(img)


# This method will determine whether the given page image contains scores, it is used for finding the first page in a score PDF in case it has a title-page etc.
# It uses FFT and peak-finding to identify such pages, not sure if this works for any type of score.
#
# TODO: Maybe we don't need this step, and can cleverly use system/block detection
# ideas:
# - Use the fact that systems should be evenly distributed (after artifact filtering) --> what about pages with a single system (orchestra sheets)?
# - Use the fact that the horizontal mean of the page should have peaks that are evenly distributed --> likely more robust
def page_has_score(img, N=100, plot=False, peaks_needed = 1):
    h, w = np.shape(img)
    
    # Perform FFT along N evenly spaced columns and collect their magnitudes
    # TODO: N should probably depend on the page width
    cols = []
    for i in np.linspace(0, w-1, N):
        col = img[:, int(i)]
        mag = np.abs(np.fft.fft(col))[1:len(col)//2]  # ignore first big value
        if max(mag) > 0:
            magnorm = mag/max(mag)
            cols += [magnorm]
            
    # Take mean of magnitudes and smooth a little bit, then perform peak detection
    # TODO: Check if smoothing is redundant
    cols = np.mean(cols, axis=0)
    smooth_factor = max(3, int(h / 500))
    cols = im.gaussian_filter1d(cols, smooth_factor, mode='nearest')
    # TODO: Use minimum peak distance?
    peaks = sig.find_peaks(cols, prominence=0.05, width=(int(h / 144), int(h/48)))
    
    # Plotting
    if plot:
        plt.figure()
        plt.plot(cols)
        for p in peaks[0]:
            plt.plot(p, cols[p], 'o')
    
    # Pages with scores on them usually generate three distinct peaks
    return len(peaks[0]) > peaks_needed


# This method will use the previous method to automatically find the first page, when given the list of (properly ordered) paths to the page images.
def find_first_page(paths):
    for i, p in enumerate(paths):
        img = open_and_preprocess(p)
        if page_has_score(img):
            return i


# TODO These assumptions are not correct
# This method will find the systems in the score.
# Note that "system" here indicates a sequential element: some orchestral scores have two separate sections on a page that are not sequential, this method will not produce the correct results in that case!
# To fully adapt this to orchestral scores, you can probably make the assumption that there is just a single line per page, and work from there.
def find_systems_in_score(img, plot=False):
    h, w = np.shape(img)

    # Here we will use binary-propagation to fill the systems, making them fully solid. We can then use the mean across the horizontal axis to find where there is "mass" on the vertical axis.
    img_solid = im.binary_fill_holes(img)                                   # Binary propagation, which usually fills up the systems (if they have no holes)
    mean_solid_systems = np.mean(img_solid, axis=1)                         # Thresholded horizontal mean after solidification, this gives a 1D image
    labels, count = im.measurements.label(im.binary_opening(mean_solid_systems > 0.2, iterations=int(w / 137.25)))  # This is a threshold, opening and label operation in one system. Note that we are actually labelling a 1D image here!

    systems = []
    for i in range(1, count + 1):
        # Using our labels we can mask out the area of interest on the vertical slice
        mask = (labels == i)
        current = mask * mean_solid_systems

        # Find top and bottom of system (wherever the value is first non-zero)
        t = np.min(np.where(current > 0))
        b = np.max(np.where(current > 0))

        system = System(
            top=t,
            bottom=b,
            start=0,
            end=w,
            v_profile=np.mean(img[t:b, :], axis=1),
            h_profile=np.mean(img[t:b, :], axis=0)
        )
        systems.append(system)

    if plot:
        plt.figure()
        plt.plot(labels)
        plt.plot(mean_solid_systems)
        for system in systems:
            plt.plot(system.top, 0, '|', color='green', markersize=15)
            plt.plot(system.bottom, 0, '|', color='red', markersize=15)
        plt.show()
    
    return systems


def modified_zscore(data):
    """
    Calculate the modified z-score of a 1 dimensional numpy array
    Reference:
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    median = np.median(data)
    deviation = data - median
    mad = np.median(np.abs(deviation))
    return 0.6745 * deviation / mad


# Here we do some fancy peak detection to detect all the systems. Check scipi docs for info on find_peaks, the chosen parameters I found seem to work best.
# It is better to make this a bit more sensitive and then filter out bad blocks in a post-processing step, by walking along the system (currently the method does this).
def find_blocks_in_system(img, system, plot=False):
    h, w = np.shape(img)
    # This definition assumes a maximum amount of 20 measures per system, spread out with equal distance.
    min_block_width = int(w / 25)
    min_heigth = np.mean(system.h_profile) + 2*np.std(system.h_profile)
    # TODO: Maybe we can do this in 1 go?
    m_peaks = sig.find_peaks(system.h_profile, distance=min_block_width, height=min_heigth, prominence=0.2)[0]                              # Thin block lines
    t_peaks = sig.find_peaks(system.h_profile, distance=min_block_width, height=0.8, width=int(w / 274.5), prominence=0.4, wlen=int(w / 274.5))[0]       # Thick block lines (when a section ends, there's a double block line)

    # Since we did two separate peak detection operations, we should filter peaks that ended up too close to each other, note this is not needed if we just do a single peak detection pass
    peaks = list(m_peaks)
    for t_peak in t_peaks:
        if min(np.abs(m_peaks - t_peak)) > min_block_width:
            peaks.append(t_peak)
    block_splits = sorted(peaks)

    # Filter out outliers by means of modified z-scores
    zscores = modified_zscore(system.h_profile[block_splits])
    print(zscores)
    block_splits = np.asarray(block_splits)[np.abs(zscores) < 10.0]

    if plot:
        plt.plot(system.h_profile)
        plt.axhline(np.mean(system.h_profile) + 3*np.std(system.h_profile))
        for peak in block_splits:
            plt.plot(peak, system.h_profile[peak], 'x', color='red')
        plt.show()
        plt.figure()
        plt.hist(system.h_profile)
        plt.show()

    # Some plumbing to make sure the splits align correctly
    # if block_splits:
        # Add start and end of system to splits. If these are close to the first and last actual splits, the access blocks will be removed later in `discard_sparse_blocks`
        # if block_splits[0] > 0:
        #     block_splits.insert(0, 0)
        # if block_splits[-1] < system.end - system.start:
        #     block_splits.append(system.end - system.start)
    #     # If the first split is too close to the start of the system, the first split should start at the system start
    #     if block_splits[0] - system.start < min_block_width:
    #         block_splits[0] = 0
    #     # If the last split is too close to the end of the system, the last split should end at the system end
    #     if system.end - block_splits[-1] < min_block_width:
    #         block_splits[-1] = system.end - system.start
    # # This else is really a fail-safe: if we did not find any blocks at all, we just assume the entire system is a single block, with just a start/end point equal to the system start/end
    # else:
    #     block_splits += [system.start, system.end]

    blocks = []
    for i in range(len(block_splits) - 1):
        blocks.append(Block(block_splits[i], block_splits[i + 1], system))
    return blocks


def find_measure_split_intersect(peak, midpoint, profile):
    # The split is made at a point of low mass (so as few intersections with mass as possible).
    # A small margin is allowed, to find a balance between cutting in the middle and cutting through less mass.
    region_min = np.min(profile)
    boundary_candidates = np.where(profile <= region_min * 1.25)[0]
    # Use index closest to the original midpoint, to bias towards the center between two bars
    measure_split = boundary_candidates[(np.abs(boundary_candidates - (midpoint - peak))).argmin()]
    return peak + measure_split


def find_measure_split_region(peak, profile, plot=False):
    # Find the longest region with intensities below a certain threshold
    region_splits = np.where(np.diff(profile < 0.05) == 1)[0]
    region_idx = (region_splits[1:] - region_splits[:-1]).argmax()
    # Split the measures at the middle of the retrieved region
    measure_split = int(np.mean([region_splits[region_idx], region_splits[region_idx + 1]]))
    return peak + measure_split


def find_measures_in_system(img, system, blocks, method='region'):
    h, w = np.shape(img)
    min_measure_dist = int(h / 30)

    # First we find peaks over the entire system to find the middle between each two consecutive bars
    peaks = sig.find_peaks(system.v_profile, distance=min_measure_dist, height=0.2, prominence=0.2)[0]  # Find peaks in vertical profile, which indicate the bar lines.
    midpoints = np.asarray(peaks[:-1] + np.round(np.diff(peaks) / 2), dtype='int')  # Get the midpoints between the detected bar lines, which will be used as the starting point for getting the Measures.
    measures = []
    for j, block in enumerate(blocks):
        # Slice out the profile for this block only
        block_profile = np.mean(img[system.top:system.bottom, system.start + block.start:system.start + block.end], axis=1)
        # The measure splits are relative to the current block, start with 0 to include the top
        measure_splits = [0]
        for i in range(len(peaks) - 1):
            # Slice out the profile between two peaks (the part in between bars)
            region_profile = block_profile[peaks[i]:peaks[i + 1]]
            if method == 'intersect':
                measure_splits.append(find_measure_split_intersect(peaks[i], midpoints[i], region_profile))
            elif method == 'region':
                measure_splits.append(find_measure_split_region(peaks[i], region_profile, plot=j < 1))
        measure_splits.append(system.bottom - system.top)

        for i in range(len(measure_splits) - 1):
            measures.append(Measure(measure_splits[i], measure_splits[i + 1], block))
    return measures


# This will get rid of systems that are too short in height, usually these are text elements and other non-score parts
def discard_thin_systems(systems):
    heights = [abs(system.top - system.bottom) for system in systems]
    return [system for system in systems if abs(system.top - system.bottom) > 0.5 * max(heights)]


# Here we discard detected blocks that are too sparse. This may be a bit sketchy, as it looks at image density, so it could accidentally discard actual score elements that are just empty.
# It is mainly intended to remove elements before or after systems that are not blocks, such as textual elements, in case left/right bound detection of lines messed up.
def discard_sparse_blocks(img, blocks):
    new_blocks = []
    for block in blocks:
        snip = img[block.system.top:block.system.bottom, block.system.start + block.start:block.system.start + block.end]
        snip = im.sobel(snip, axis=0) > 0.5
        density = np.sum(snip)/np.size(snip)
        if density > 0.07:
            new_blocks.append(block)
    return new_blocks


# Gives a sorted list of pages. As long as the pages are numbered incrementally separated with a "-", this will work fine.
def get_sorted_page_paths(page_path):
    paths = [p for p in Path(page_path).iterdir() if p.is_file()]
    return [str(p.resolve()) for p in sorted(paths, key=lambda p: int(str(p.stem).split("-")[1]))]


def plot_measures(img, measures):
    plt.figure(figsize=(20, 20))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    for measure in measures:
        start = (measure.block.system.start + measure.block.start, measure.block.system.top + measure.top)
        rect = patches.Rectangle(start, measure.block.end - measure.block.start, measure.bottom - measure.top,
                                 linewidth=6, edgecolor='r', facecolor='r', alpha=0.2)
        plt.gca().add_patch(rect)
    plt.show()


def create_img_with_measure(path, measures):
    max_height = 950
    img = Image.open(path)
    for measure in measures:
        draw = ImageDraw.Draw(img)
        start = (measure.block.system.start + measure.block.start, measure.block.system.top + measure.top)
        end = (start[0] + (measure.block.end - measure.block.start), start[1] + (measure.bottom - measure.top))
        draw.rectangle((start, end), outline='red', fill=None, width=5)
        del draw
    scale = max_height / img.size[1]
    img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    return img


def show_measures(path, measures):
    img = create_img_with_measure(path, measures)
    img.show()


def save_measure_img(path, measures, name):
    img = create_img_with_measure(path, measures)
    img.save(name, "PNG")


def main():
    page_path = r"../data/Mahler_Symphony_1/ppm-600"
    plot = False
    show = False
    save = True

    paths = get_sorted_page_paths(page_path)

    # first_page = find_first_page(paths)
    first_page = 0
    print(f"Scores start at page {paths[first_page]}")
    for i in range(first_page, len(paths)):
        path = paths[i]
        print(f"Processing page {i+1} ({path})")
        img = open_and_preprocess(path)
        systems = find_systems_in_score(img)
        # lines = discard_thin_lines(lines)
        all_measures = []
        for system in systems:
            blocks = find_blocks_in_system(img, system)
            blocks = discard_sparse_blocks(img, blocks)
            measures = find_measures_in_system(img, system, blocks, method='region')
            all_measures += measures

        if plot:
            plot_measures(img, all_measures)
        if show:
            show_measures(path, all_measures)
        if save:
            save_measure_img(path, all_measures, str((Path(r"../tmp/output") / path.split("/")[-1]).resolve()))


if __name__ == "__main__":
    main()
