import numpy as np
from PIL import Image, ImageOps
from statistics import median
from tqdm import tqdm

from score_analysis.score_image import ScoreImage
from score_analysis.skeleton_extractor import SkeletonExtractor
from util.dirs import data_dir


class BarlineDetector:

    def __init__(self, image, len_threshold=0):
        self.score_image = image if isinstance(image, ScoreImage) else ScoreImage(image)
        self.image = self.score_image.image
        self.staffline_height = self.score_image.staffline_height
        self.staffspace_height = self.score_image.staffspace_height

        if len_threshold == 0:
            # Default length threshold is set at the height of two staffs of 5 stafflines each.
            staff_height = 5 * self.staffspace_height + 5 * self.staffline_height
            self.len_threshold = staff_height * 2.5
        else:
            self.len_threshold = len_threshold

    class SegmentGroup:
        """
        Helper data class for BarlineSegments groups. Groups are formed when `barline.x_values[0]` is within
        a `staffspace_height` margin.
        """
        def __init__(self, group_distance):
            self.group_distance = group_distance
            self.x = 0
            self.min_x = 0
            self.max_x = 0
            self.max_length = 0
            self.segments = []

        def reaches(self, segment):
            return len(self.segments) == 0 or \
                   (self.min_x - self.group_distance <= segment.x_values[0] <= self.max_x + self.group_distance)

        def add_segment(self, segment):
            first = len(self.segments) == 0
            self.segments.append(segment)
            if first or segment.x_values[0] < self.min_x:
                self.min_x = segment.x_values[0]
            if first or segment.x_values[0] > self.max_x:
                self.max_x = segment.x_values[0]
            if len(segment.x_values) > self.max_length:
                self.x = segment.x_values[0]
                self.max_length = len(segment.x_values)

    def detect_barlines(self, debug=0):
        """
        Detect barlines on the ScoreImage. The following steps are taken:
         1) First, a skeleton is extracted in the vertical direction, yielding one-pixel thin vertical runs
            of the ScoreImage.
         2) The skeletons are filtered by length to barline candidates. Default threshold is set at two staff heights,
            with each staff consisting of 5 stafflines.
         3) The barline candidates are grouped together into systems. The assumption made here is that at least one
            of the barlines overlaps with all barline candidates belonging to one system,
            spanning (approximately) the entire height of the system. From these selected barline candidates,
            boundaries of a system in the vertical direction are determined.
         4) In each system the candidate barlines are grouped into barline groups. Candidate barlines are grouped
            together when they are within 2 * staffspace_height of horizontal distance of each other. From each group,
            overlap in the horizontal direction is then removed by keeping the longest of the overlapping segments.
            This yields a Barline with segments that do not overlap horizontally, and fall in neighboring columns,
            meaning they are likely to portray one single full barline.
         5) Filter out false positives from the barline groups. The used assumption here is that most barlines will
            consist of approximately the same amount of black pixels throughout the systems, although they may be
            interrupted and consist of multiple segments. Any barline groups that have less than half the median length
            are discarded.
         6) These subsets of BarlineSegments are each used to find a prediction for a single uninterrupted line
            from the top to the bottom of the vertical system boundaries. By interpolating the parts in between segments
            that are unknown, any possible curves are followed as closely as possible. All vertical runs from the
            original skeleton list for which at least 80% of the pixels fall into a margin of 2 * staffline_space of the
            corresponding predicted pixel are selected as segments for the barline. These are again filtered to remove
            horizontally overlapping segments. This yields the final Barline.
        :returns A list of Systems, each containing a list of Barlines, each containing a list of BarlineSegments.
        """
        # Step 1: Extract skeletons
        extractor = SkeletonExtractor(self.image, 'vertical')
        skeleton_list = extractor.get_skeleton_list()

        # Step 2: Filter skeletons by threshold length
        candidate_lines = [BarlineSegment(line[0], line[1]) for line in skeleton_list
                           if len(line[1]) >= self.len_threshold]
        candidate_lines.sort(key=lambda l: (l.y, len(l.x_values)))

        # Step 3: Group candidate lines into system groups based on overlapping y values.
        label = 1
        system_groups = []
        # Using the Barline class here is not entirely correct, since the horizontal direction is not considered,
        # but exactly fits the purpose of grouping BarlineSegments.
        current_candidate = Barline()
        current_candidate.label = label
        for segment in candidate_lines:
            if segment.label > 0:
                continue
            if len(current_candidate.segments) > 0 \
                    and (segment.y + len(segment.x_values) < current_candidate.min_y
                         or segment.y > current_candidate.max_y):
                # End of overlaps: start with new Barline
                system_groups.append(current_candidate)
                label += 1
                current_candidate = Barline()
                current_candidate.label = label
            segment.label = label
            current_candidate.add_segment(segment)
        system_groups.append(current_candidate)

        label = 1
        systems = []
        group_distance = 2 * self.staffspace_height
        for system_group in system_groups:
            system_group.segments.sort(key=lambda s: s.x_values[0])

            # Step 4: Group segments further into barline groups. Overlap in the horizontal direction is eliminated by
            # keeping the longest of the overlapping segments.
            grouped_segments = []
            current_group = self.SegmentGroup(group_distance=group_distance)
            for segment in system_group.segments:
                if not current_group.reaches(segment):
                    grouped_segments.append(current_group)
                    current_group = self.SegmentGroup(group_distance=group_distance)
                current_group.add_segment(segment)
            grouped_segments.append(current_group)
            barlines = []
            for group in grouped_segments:
                barline = Barline()
                barline.label = label
                group.segments.sort(key=lambda s: len(s.x_values), reverse=True)
                for segment in group.segments:
                    overlaps = False
                    for barline_segment in barline.segments:
                        if barline_segment.y <= segment.y <= barline_segment.y + len(barline_segment.x_values) or \
                                barline_segment.y <= segment.y + len(segment.x_values) <= barline_segment.y + len(barline_segment.x_values):
                            overlaps = True
                    if not overlaps:
                        barline.add_segment(segment)
                label += 1
                barlines.append(barline)

            # Step 5: Filter false positives from the barline groups
            median_length = median([sum([len(s.x_values) for s in barline.segments]) for barline in barlines])
            barlines = [b for b in barlines if sum([len(s.x_values) for s in b.segments]) > 0.5 * median_length]

            # Step 6: Predict a barline spanning the entire height of the system group. All skeletons that are close to
            # this prediction are selected as being part of the final Barline.
            margin = 2 * self.staffline_height
            system = System()
            for barline in barlines:
                barline.interpolate(system_group.min_y, system_group.max_y)
                for skeleton in [skeleton for skeleton in skeleton_list if system_group.min_y <= skeleton[0] <= system_group.max_y]:
                    margin_count = 0
                    start_idx = skeleton[0] - system_group.min_y
                    stop_idx = min(len(barline.prediction) - 1, skeleton[0] - system_group.min_y + len(skeleton[1]))
                    if stop_idx - start_idx == 0:
                        continue
                    for i in range(start_idx, stop_idx):
                        if abs(barline.prediction[i] - skeleton[1][i - start_idx]) <= margin:
                            margin_count += 1
                    if margin_count / (stop_idx - start_idx) > 0.8:
                        segment = BarlineSegment(skeleton[0], skeleton[1])
                        segment.label = label
                        barline.add_segment(segment)
                barline.segments.sort(key=lambda s: len(s.x_values), reverse=True)
                new_barline = Barline()
                for segment in barline.segments:
                    overlaps = False
                    for barline_segment in new_barline.segments:
                        if barline_segment.y <= segment.y <= barline_segment.y + len(barline_segment.x_values) or \
                                barline_segment.y <= segment.y + len(segment.x_values) <= barline_segment.y + len(barline_segment.x_values):
                            overlaps = True
                    if not overlaps:
                        new_barline.add_segment(segment)
                # Transfer barline predictions to new barline
                new_barline.prediction = barline.prediction
                new_barline.prediction_y_start = barline.prediction_y_start
                new_barline.label = barline.label
                system.add_barline(new_barline)
                label += 1
            systems.append(system)

        if debug > 0:
            extractor.skeleton_list_to_image()
            ImageOps.invert(extractor.skeleton_image).save('skeleton.png')

            candidate_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for segment in candidate_lines:
                for row in range(len(segment.x_values)):
                    candidate_img[segment.y + row, segment.x_values[row]-2:segment.x_values[row]+2] = (255, 0, 0)
            Image.fromarray(candidate_img).save('candidate_barlines.png')

            grouped_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for barline in system_groups:
                color = (255, barline.label * 50, 0)
                for segment in barline.segments:
                    for row in range(len(segment.x_values)):
                        grouped_img[segment.y + row, segment.x_values[row]-2:segment.x_values[row]+2] = color
            Image.fromarray(grouped_img).save('grouped_candidate_barlines.png')

            prediction_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for system in systems:
                for barline in system.barlines:
                    color = (150 + (31 * barline.label) % 106,
                             150 + (111 * (barline.label + 1)) % 106,
                             150 + (201 * (barline.label + 2)) % 106)
                    for row in range(len(barline.prediction)):
                        prediction_img[barline.prediction_y_start + row, barline.prediction[row]-2:barline.prediction[row]+2] = color
            Image.fromarray(prediction_img).save('barline_predictions.png')

            barline_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for system in systems:
                for barline in system.barlines:
                    color = (150 + (31 * barline.label) % 106,
                             150 + (111 * (barline.label + 1)) % 106,
                             150 + (201 * (barline.label + 2)) % 106)
                    for segment in barline.segments:
                        for row in range(len(segment.x_values)):
                            barline_img[segment.y + row, segment.x_values[row]-2:segment.x_values[row]+2] = color
            Image.fromarray(barline_img).save('barlines.png')
        return systems


class BarlineSegment:
    """
    Data class that represents a barline segment.
    """
    def __init__(self, y, x_values):
        self.y = y
        self.x_values = x_values
        self.label = 0


class Barline:
    """
    Helper class that represents a Barline. A Barline consists of several BarlineSegments.
    """
    def __init__(self):
        self.label = 0
        self.segments = []
        self.min_y = 0
        self.max_y = 0
        self.prediction = []
        self.prediction_y_start = 0

    def add_segment(self, segment):
        first = len(self.segments) == 0
        self.segments.append(segment)
        if first or segment.y < self.min_y:
            self.min_y = segment.y
        if first or segment.y + len(segment.x_values) > self.max_y:
            self.max_y = segment.y + len(segment.x_values)

    def remove_segment(self, segment):
        self.segments.remove(segment)
        min_y = self.segments[0].y
        max_y = 0
        for s in self.segments:
            if s.y < min_y:
                min_y = s.y
            if s.y + len(s.x_values) > max_y:
                max_y = s.y + len(s.x_values)
        self.min_y = min_y
        self.max_y = max_y

    def interpolate(self, start, end):
        """
        Find a prediction of the entire barline by interpolating missing sections between segments.
        :param start: The top of the interpolated y region
        :param end: The bottom of the interpolated y region
        """
        self.prediction_y_start = start
        self.segments.sort(key=lambda s: s.y)
        prediction = [0] * (end - start)
        for segment in self.segments:
            for i in range(segment.y - start, segment.y + len(segment.x_values) - start):
                prediction[i] = segment.x_values[i - (segment.y - start)]

        missing_start = start
        segment_idx = 0
        prev_segment = None
        while missing_start < end:
            next_segment = self.segments[segment_idx] if segment_idx < len(self.segments) else None
            missing_end = end if next_segment is None else next_segment.y
            if prev_segment is None:
                x_diff = next_segment.x_values[-1] - next_segment.x_values[0]
                y_diff = len(next_segment.x_values)
            elif next_segment is None:
                x_diff = prev_segment.x_values[-1] - prev_segment.x_values[0]
                y_diff = len(prev_segment.x_values)
            else:
                x_diff = next_segment.x_values[0] - prev_segment.x_values[-1]
                y_diff = next_segment.y - (prev_segment.y + len(prev_segment.x_values))
            tan_a = x_diff / y_diff
            for i in range(missing_start, missing_end):
                if prev_segment is None:
                    prediction[i - start] = next_segment.x_values[0] - int(tan_a * (i - missing_start))
                else:
                    prediction[i - start] = prev_segment.x_values[-1] + int(tan_a * (i - missing_start))
            prev_segment = next_segment
            missing_start = end if prev_segment is None else prev_segment.y + len(prev_segment.x_values)
            segment_idx += 1
        self.prediction = prediction


class System:

    def __init__(self):
        self.barlines = []
        self.min_y = 0
        self.max_y = 0

    def add_barline(self, barline):
        first = len(self.barlines) == 0
        self.barlines.append(barline)
        if first or barline.min_y < self.min_y:
            self.min_y = barline.min_y
        if first or barline.max_y > self.max_y:
            self.max_y = barline.max_y


if __name__ == '__main__':
    for image_path in tqdm((data_dir / 'sample' / 'pages').iterdir()):
        image = Image.open(image_path)
        detector = BarlineDetector(image)
        systems = detector.detect_barlines(debug=0)
        barline_img = np.array(image.convert('RGB'))
        for system in systems:
            for barline in system.barlines:
                color = (150 + (31 * barline.label) % 106,
                         150 + (111 * (barline.label + 1)) % 106,
                         150 + (201 * (barline.label + 2)) % 106)
                for segment in barline.segments:
                    for row in range(len(segment.x_values)):
                        barline_img[segment.y + row, segment.x_values[row] - 2:segment.x_values[row] + 2] = color
        Image.fromarray(barline_img).save(data_dir / 'sample' / 'barlines' / image_path.name)
