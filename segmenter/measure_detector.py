from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as im
import scipy.signal as sig
import sys

from util.cv2_util import correct_image_rotation, find_horizontal_lines, find_vertical_lines, invert_and_threshold
from util.PIL_util import resize_img

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=sys.maxsize)

Page = namedtuple("Page", ["height", "width", "rotation", "systems"])
System = namedtuple("System", ["ulx", "uly", "lrx", "lry", "v_profile", "h_profile", "staff_boundaries", "measures"])
Measure = namedtuple("Measure", ["ulx", "uly", "lrx", "lry", "staffs"])
Staff = namedtuple("Staff", ["ulx", "uly", "lrx", "lry"])


def contiguous_regions(condition):
    """
    Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    See: https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


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
    mad = np.median(np.abs(deviation)) + 1e-6  # Small epsilon to avoid division by 0
    return 0.6745 * deviation / mad


class MeasureDetector:

    def __init__(self, path):
        self.path = path
        self.page = None
        self.original = self.rotated = self.bw = None
        self.v_lines = None
        self.rotation = 0
        self._img_max = 255

    def open_and_preprocess(self):
        """
        Load the image and perform some preprocessing. These steps are:
        - Find the rotation of the image and correct for it.
        - Invert and threshold the image.
        """
        original = cv2.imread(self.path)
        rotated, rotation = correct_image_rotation(original)
        bw = invert_and_threshold(rotated)
        v_lines = find_vertical_lines(bw)
        self.original = original
        self.rotated = rotated
        self.bw = bw
        self.rotation = rotation
        self.v_lines = v_lines

    def find_system_staff_boundaries_peaks(self, v_profile):
        # Find peaks in vertical profile, which indicate the bar lines.
        peaks = sig.find_peaks(v_profile, height=0.3 * self._img_max, prominence=0.1)[0]
        gaps = np.diff(peaks)

        median_gap = np.median(gaps)
        staff_split_indices = np.where(gaps > 2*median_gap)[0]
        staff_split_indices = np.append(staff_split_indices, gaps.shape[0])
        staff_boundaries = []
        cur_start = 0
        if len(staff_split_indices) == 0:
            return []

        for staff_split_idx in staff_split_indices:
            staff_boundary = [peaks[cur_start], peaks[staff_split_idx]]
            cur_start = staff_split_idx + 1
            staff_boundaries.append(staff_boundary)
        return staff_boundaries

    def find_system_staff_boundaries_lines(self, sub_img, plot=False):
        line_img = find_horizontal_lines(sub_img)
        v_profile = np.mean(line_img, axis=1)
        peaks = sig.find_peaks(v_profile, height=0.3 * self._img_max)[0]
        gaps = np.diff(peaks)

        if plot:
            plt.figure()
            plt.plot(v_profile)
            for peak in peaks:
                plt.plot(peak, v_profile[peak], 'x', color='red')
            plt.title('Vertical profile of system with detected line peaks')
            plt.show()

        staff_split_indices = np.where(gaps > 2 * gaps.mean())[0]
        staff_split_indices = np.append(staff_split_indices, gaps.shape[0])
        staff_boundaries = []
        cur_start = 0
        if len(staff_split_indices) == 0:
            return []

        for staff_split_idx in staff_split_indices:
            staff_boundary = [peaks[cur_start], peaks[staff_split_idx]]
            cur_start = staff_split_idx + 1
            staff_boundaries.append(staff_boundary)
        return staff_boundaries

    def find_system_staff_boundaries(self, sub_img, method='peaks', plot=False):
        v_profile = np.mean(sub_img, axis=1)
        if method == 'peaks':
            staff_boundaries = self.find_system_staff_boundaries_peaks(v_profile)
        else:
            staff_boundaries = self.find_system_staff_boundaries_lines(sub_img, plot=plot)
        if plot:
            plt.figure()
            plt.plot(v_profile)
            for bound in staff_boundaries:
                plt.axvline(bound[0], color='green')
                plt.axvline(bound[1], color='green')
            plt.title("Vertical profile of system with staff boundaries")
            plt.show()

        return staff_boundaries

    def find_systems_in_page(self, plot=False):
        img = self.bw
        h, w = np.shape(img)

        # Use binary propagation to make systems solid, so that we can apply some noise reduction.
        img_solid = im.binary_fill_holes(img)
        mean_solid_systems = np.mean(img_solid, axis=0)
        v_regions = contiguous_regions(mean_solid_systems > 0.15)
        lx, rx = v_regions[np.argmax(np.diff(v_regions))]
        lx = max(0, int(lx - (rx - lx) * 0.01))
        rx = min(w, int(rx + (rx - lx) * 0.01))
        noise_mask = np.ones(w, dtype=bool)
        noise_mask[lx:rx] = False
        v_lines = self.v_lines
        v_lines[:, noise_mask] = 0

        h_profile = np.mean(v_lines, axis=1)
        kernel = (1/9) * np.ones(9, dtype=np.float)
        smoothed_h_profile = np.convolve(h_profile, kernel)[4:-4]
        system_boundaries = contiguous_regions(smoothed_h_profile > 0.0)
        # Ignore system if it is too small in height (< 5% page height). Mainly this happens with text on pages.
        system_boundaries = system_boundaries[np.diff(system_boundaries).reshape(-1) / h > 0.05]

        for i in range(system_boundaries.shape[0]):
            if i >= system_boundaries.shape[0] - 1:
                half_diff = np.diff(system_boundaries[i]) * 0.05
            else:
                half_diff = (system_boundaries[i + 1][0] - system_boundaries[i][1]) // 2
            if i == 0:
                system_boundaries[i][0] = max(0, system_boundaries[i][0] - half_diff)
            system_boundaries[i][1] = min(h, system_boundaries[i][1] + half_diff)
            if i < system_boundaries.shape[0] - 1:
                system_boundaries[i + 1][0] -= half_diff

        if plot:
            plt.figure()
            plt.plot(smoothed_h_profile)
            for system_boundary in system_boundaries:
                plt.axvline(system_boundary[0], color='green')
                plt.axvline(system_boundary[1], color='green')
            plt.title('Horizontal profile with detected systems')
            plt.show()

        systems = []
        for uly, lry in system_boundaries:

            # Find left and right border of system as the largest active region
            snippet = np.mean(img[uly:lry, :], axis=0)
            regions = contiguous_regions(snippet > 0.4)
            ulx, lrx = regions[np.argmax(np.diff(regions).flatten())]
            # Add 1 percent margin on both sides to improve peak detection at the edges of the system h_profile
            ulx = max(0, int(ulx - (lrx - ulx) * 0.01))
            lrx = min(w, int(lrx + (lrx - ulx) * 0.01))

            system = System(
                ulx=ulx,
                uly=uly,
                lrx=lrx,
                lry=lry,
                v_profile=np.mean(img[uly:lry, ulx:lrx], axis=1),
                h_profile=np.mean(img[uly:lry, ulx:lrx], axis=0),
                staff_boundaries=[],
                measures=[]
            )
            systems.append(system)

        return systems

    def find_measures_in_system(self, system, plot=False):
        img = self.bw
        h, w = np.shape(img)

        h_profile = np.mean(self.v_lines[system.uly:system.lry, system.ulx:system.lrx], axis=0)
        mean, std = np.mean(h_profile), np.std(h_profile)

        # Take a relatively small min_width to also find measure lines in measures (for e.g. a pickup or anacrusis)
        min_block_width = int(w / 50)
        min_height = mean + 2*std
        peaks = sig.find_peaks(h_profile, distance=min_block_width, height=min_height, prominence=0.15)[0]
        measure_split_candidates = sorted(peaks)

        # Filter out outliers by means of modified z-scores
        zscores = modified_zscore(h_profile[measure_split_candidates])
        # Use only candidate peaks if their modified z-score is below a given threshold or if their height is at least 3 standard deviations over the mean
        measure_splits = np.asarray(measure_split_candidates)[(np.abs(zscores) < 15.0) | (h_profile[measure_split_candidates] > mean + 3*std)]
        if measure_splits.shape[0] > 0 and measure_splits[-1] < (h_profile.shape[0] - 2*min_block_width):
            measure_splits = np.append(measure_splits, h_profile.shape[0])

        if plot:
            plt.figure()
            plt.plot(h_profile, color='green')
            plt.axhline(mean + 2*std)
            for split in peaks:
                plt.plot(split, h_profile[split], 'x', color='red')
            plt.title("Horizontal profile of system without staffs")
            plt.show()

        measures = []
        for i in range(len(measure_splits) - 1):
            measures.append(Measure(
                ulx=system.ulx + measure_splits[i],
                uly=system.uly,
                lrx=system.ulx + measure_splits[i + 1],
                lry=system.lry,
                staffs=[]
            ))
        return system._replace(measures=measures)

    def add_staff_boundaries(self, system, method='peaks', plot=False):
        img = self.bw
        ulx = system.measures[0].ulx
        lrx = system.measures[len(system.measures) - 1].lrx
        # Find staff boundaries for system
        staff_boundaries = self.find_system_staff_boundaries(img[system.uly:system.lry, ulx:lrx], method, plot)
        return system._replace(staff_boundaries=staff_boundaries)

    def find_staff_split_intersect(self, profile, plot=False):
        # The split is made at a point of low mass (so as few intersections with mass as possible).
        # A small margin is allowed, to find a balance between cutting in the middle and cutting through less mass.
        region_min = np.min(profile)
        midpoint = int(profile.shape[0] / 2)
        boundary_candidates = np.where(profile <= region_min * 1.25)[0]
        # Use index closest to the original midpoint, to bias towards the center between two bars
        staff_split = boundary_candidates[(np.abs(boundary_candidates - midpoint)).argmin()]

        if plot:
            plt.figure()
            plt.plot(profile)
            plt.axvline(staff_split, color='red')
            plt.title("Intersect split profile with split")
            plt.show()

        return staff_split

    def find_staff_split_region(self, profile, plot=False):
        # Find the longest region with intensities below a certain threshold
        region_min = np.min(profile)
        region_splits = contiguous_regions(profile <= 2 * region_min)
        # Fallback to 'intersect' method when no region is found
        if region_splits.shape[0] == 0:
            return self.find_staff_split_intersect(profile)
        region_idx = np.argmax(np.diff(region_splits).flatten())
        # Split the measures at the middle of the retrieved region
        staff_split = int(np.mean(region_splits[region_idx]))

        if plot:
            plt.figure()
            plt.plot(profile)
            plt.axhline(region_min, color='green')
            plt.axhline(2 * region_min)
            plt.axvline(staff_split, color='red')
            plt.title("Region split profile with split and region minimum")
            plt.show()

        return staff_split

    def add_staffs_to_system(self, system, method='region'):
        img = self.bw
        populated_measures = []
        for measure in system.measures:
            # Slice out the profile for this measure only
            measure_profile = np.mean(img[measure.uly:measure.lry, measure.ulx:measure.lrx], axis=1)
            # The measure splits are relative to the current measure, start with 0 to include the top
            staff_splits = []
            for i in range(len(system.staff_boundaries) - 1):
                # Slice out the profile between two peaks (the part in between bars)
                region_profile = measure_profile[system.staff_boundaries[i][1]:system.staff_boundaries[i + 1][0]]
                if method == 'intersect':
                    staff_split = self.find_staff_split_intersect(region_profile)
                elif method == 'region':
                    staff_split = self.find_staff_split_region(region_profile, plot=False)
                else:
                    staff_split = int(region_profile.shape[0] / 2)
                staff_splits.append(staff_split + system.staff_boundaries[i][1])

            staff_splits.insert(0, 0)
            staff_splits.append(system.lry - system.uly)

            staffs = []
            for j in range(len(staff_splits) - 1):
                staffs.append(Staff(
                    ulx=measure.ulx,
                    uly=measure.uly + staff_splits[j],
                    lrx=measure.lrx,
                    lry=measure.uly + staff_splits[j + 1],
                ))
            populated_measure = measure._replace(staffs=staffs)
            populated_measures.append(populated_measure)
        return system._replace(measures=populated_measures)

    def detect(self, sys_method='lines', measure_method='region', plot=False):
        self.open_and_preprocess()
        height, width = self.bw.shape
        systems = self.find_systems_in_page(plot=plot)
        populated_systems = []
        for system in systems:
            system = self.find_measures_in_system(system, plot)
            system = self.add_staff_boundaries(system, method=sys_method, plot=plot)
            system = self.add_staffs_to_system(system, method=measure_method)
            populated_systems.append(system)
        page = Page(height=height, width=width, rotation=self.rotation, systems=populated_systems)
        self.page = page
        return self
