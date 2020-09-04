from PIL import Image, ImageOps
from collections import namedtuple
from matplotlib import patches
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.ndimage as im
import sys
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
    img = Image.fromarray(plt.imread(path)[:, :, 0]).convert('L')  # Convert to "luminance" (single-channel greyscale)
    img = ImageOps.autocontrast(img)                             # Does some automatic histogram stretching to enhance contrast
    img = ImageOps.invert(img)                                   # Inverts the image so we get white on black
    img = img.point(lambda x: x > 50)                           # Threshold on value 50, this will convert the range into [0, 1] (floats though!)
    
    return np.array(img) # float (range 0-1) single-channel image


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
        # We need rough values for these first to determine the left and right bounds of the system, we will correct them later to tighter values
        t = np.min(np.where(current > 0))
        b = np.max(np.where(current > 0))
        
        # Taking these vertical bounds from the 2D image, we obtain a cut-out version of a system by slicing.
        # Then looking at the horizontal profile (so averaged over the vertical axis), we get a 1D image which we can use for left and right starting points
        snip = img_solid[t:b, :]
        solid_h_profile = np.mean(snip, axis=0) > 0.1
        l = np.min(np.where(solid_h_profile))
        r = np.max(np.where(solid_h_profile))
        
        # Here we correct the top and bottom values to tighter bounds (the score systems themselves to be precise, without the notes sticking out)
        scale = w/abs(r-l)  # This scale value is used to filter out rubbish that is too short to be a system. Probably it is better to do this in post-processing entirely though... Feel free to remove?
        try:
            t_corr = np.min(np.where(current*scale > 0.85))
            b_corr = np.max(np.where(current*scale > 0.85))

            system = System(
                top=t_corr,
                bottom=b_corr,
                start=l,
                end=r,
                v_profile=np.mean(img[t_corr:b_corr, l:r], axis=1),
                h_profile=np.mean(img[t_corr:b_corr, l:r], axis=0)
            )
            systems.append(system)
        except:
            # Just a place-holder to indicate that we don't care when a system does not get detected (just so that it continues)
            print("Can't find top and bottom bounds within threshold for system ", i)
        
    # Plotting code, plots some of the intermediate images that are used, enable by setting the "plot" keyword to True in the method call
    if plot:
        plt.figure()
        plt.plot(labels)
        plt.plot(mean_solid_systems)
        for system in systems:
            plt.plot(system.top, 0, '|', color='green', markersize=15)
            plt.plot(system.bottom, 0, '|', color='red', markersize=15)
    
    return systems
        

# Here we do some fancy peak detection to detect all the systems. Check scipi docs for info on find_peaks, the chosen parameters I found seem to work best.
# It is better to make this a bit more sensitive and then filter out bad blocks in a post-processing step, by walking along the system (currently the method does this).
def find_blocks_in_system(img, system, plot=False):
    h, w = np.shape(img)
    # TODO: Verify what the actual minimum possible width for a block is
    min_block_width = int(w / 55)
    sy = im.gaussian_filter(im.sobel(img[system.top:system.bottom, :], axis=1, mode='nearest'), 2)                                                               # Obtain vertical edges via sobel filter
    # TODO: Maybe we can do this in 1 go?
    m_peaks = sig.find_peaks(system.h_profile, distance=min_block_width, height=0.4, prominence=0.3, wlen=int(w / 100))[0]                              # Thin block lines
    t_peaks = sig.find_peaks(system.h_profile, distance=min_block_width, height=0.8, width=int(w / 274.5), prominence=0.4, wlen=int(w / 274.5))[0]       # Thick block lines (when a section ends, there's a double block line)

    # Since we did two separate peak detection operations, we should filter peaks that ended up too close to each other, note this is not needed if we just do a single peak detection pass
    peaks = list(m_peaks)
    for t_peak in t_peaks:
        if min(np.abs(m_peaks - t_peak)) > min_block_width:
            peaks.append(t_peak)
    block_splits = sorted(peaks)

    # Some plumbing to make sure the splits align correctly
    if block_splits:
        # Add start and end of system to splits. If these are close to the first and last actual splits, the access blocks will be removed later in `discard_sparse_blocks`
        if block_splits[0] > 0:
            block_splits.insert(0, 0)
        if block_splits[-1] < system.end - system.start:
            block_splits.append(system.end - system.start)
    #     # If the first split is too close to the start of the system, the first split should start at the system start
    #     if block_splits[0] - system.start < min_block_width:
    #         block_splits[0] = 0
    #     # If the last split is too close to the end of the system, the last split should end at the system end
    #     if system.end - block_splits[-1] < min_block_width:
    #         block_splits[-1] = system.end - system.start
    # # This else is really a fail-safe: if we did not find any blocks at all, we just assume the entire system is a single block, with just a start/end point equal to the system start/end
    else:
        block_splits += [system.start, system.end]

    blocks = []
    for i in range(len(block_splits) - 1):
        blocks.append(Block(block_splits[i], block_splits[i + 1], system))
    return blocks


# TODO: See if splitting through middle of largest stretch of (almost) no mass works better.
def find_measures_in_system(img, system, blocks):
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
        for i, midpoint in enumerate(midpoints):
            # Slice out the profile between two peaks (the part in between bars)
            region_profile = block_profile[peaks[i]:peaks[i+1]]

            # The split is made at a point of low mass (so as few intersections with mass as possible).
            # A small margin is allowed, to find a balans between cutting in the middle and cutting through less mass.
            region_min = np.min(region_profile)
            boundary_candidates = np.where(region_profile <= region_min * 1.25)[0]
            # Use index closest to the original midpoint, to bias towards the center between two bars
            measure_split = boundary_candidates[(np.abs(boundary_candidates - (midpoint - peaks[i]))).argmin()]
            measure_splits.append(peaks[i] + measure_split)
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


def main():
    page_path = r"../tmp/test"
    paths = get_sorted_page_paths(page_path)

    first_page = find_first_page(paths)
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
            measures = find_measures_in_system(img, system, blocks)
            all_measures += measures

        # This creates plots of the score pages, with block highlighting
        plt.figure(figsize=(20, 20))
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        for measure in all_measures:
            start = (measure.block.system.start + measure.block.start, measure.block.system.top + measure.top)
            rect = patches.Rectangle(start, measure.block.end - measure.block.start, measure.bottom - measure.top, linewidth=6, edgecolor='r', facecolor='r', alpha=0.2)
            plt.gca().add_patch(rect)
        plt.show()


if __name__ == "__main__":
    main()
