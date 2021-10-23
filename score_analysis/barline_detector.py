import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from score_analysis.score_image import ScoreImage
from score_analysis.skeleton_extractor import SkeletonExtractor
from util.dirs import data_dir
from util.score_draw import ScoreDraw


class BarlineDetector:

    def __init__(self, image, len_threshold=0, output_path=None):
        self.score_image = image if isinstance(image, ScoreImage) else ScoreImage(image)
        self.image = self.score_image.image
        self.staffline_height = self.score_image.staffline_height
        self.staffspace_height = self.score_image.staffspace_height
        self.output_path = Path(output_path) if output_path is not None else None

        if len_threshold == 0:
            # Default length threshold is set at the height of 2 staffs of 5 stafflines each.
            staff_height = 5 * self.staffspace_height + 5 * self.staffline_height
            self.len_threshold = staff_height * 2.5
        else:
            self.len_threshold = len_threshold

    def overlap(self, segment, start, end):
        if isinstance(segment, BarlineSegment):
            y = segment.y
            x_len = len(segment.x_values)
        elif isinstance(segment, list) and len(segment) == 2:
            y = segment[0]
            x_len = len(segment[1])
        else:
            raise ValueError('Segment not of type BarlineSegment or Skeleton')
        return start <= y <= end or start <= y + x_len <= end

    def segment_overlaps_group(self, grouped_segments, segment):
        return any([self.overlap(s, segment.y, segment.y + len(segment.x_values)) or
                    self.overlap(segment, s.y, s.y + len(s.x_values)) for s in grouped_segments.segments])

    def group_to_segment_angle(self, grouped_segments, segment):
        if len(grouped_segments.segments) == 0:
            raise ValueError('Group has no segments')
        if self.segment_overlaps_group(grouped_segments, segment):
            raise NotImplementedError('Group to segment angle not supported for segments that overlaps group')
        i = 0
        while i < len(grouped_segments.segments) - 1 and grouped_segments.segments[i].y < segment.y:
            i += 1
        closest_segment = grouped_segments.segments[i]
        if i < len(grouped_segments.segments) - 1 and segment.y - (closest_segment.y + len(closest_segment.x_values)) > grouped_segments.segments[i + 1].y - (segment.y + len(segment.x_values)):
            closest_segment = grouped_segments.segments[i + 1]
        if segment.y < closest_segment.y:
            return (closest_segment.x_values[0] - segment.x_values[-1]) / (closest_segment.y - (segment.y + len(segment.x_values)))
        else:
            return (closest_segment.x_values[-1] - segment.x_values[0]) / (segment.y - (closest_segment.y + len(closest_segment.x_values)))

    class BarlineCandidateGroup:
        """
        Helper data class for BarlineCandidate groups.
        """
        def __init__(self):
            self.label = 0
            self.min_y = 0
            self.max_y = 0
            self.longest_segment = None
            self.segments = []

        def add_segment(self, segment):
            first = len(self.segments) == 0
            self.segments.append(segment)
            if self.longest_segment is None or len(segment.x_values) > len(self.longest_segment.x_values):
                self.longest_segment = segment
            if first or segment.y < self.min_y:
                self.min_y = segment.y
            if first or segment.y + len(segment.x_values) > self.max_y:
                self.max_y = segment.y + len(segment.x_values)

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
            together when they don't overlap horizontally and approximately lie on a common line. An imaginary line is
            drawn in between the candidate and the segment of the group that lies closest to it.
            The candidate barline is then added to the group when both:
             1) The angle of the candidate is approximately the same as the average angle of all segments in the group.
             2) The angle of the imaginary line is approximately the same as the average angle of the group.
            Otherwise the a new group is started and the candidate is added to the new group.
            This yields a Barline with segments that do not overlap horizontally and lie on an approximate line,
            meaning they are likely to portray one single full barline.
         5) Remove Barlines that are close to another Barline. The longest Barline is kept. This is done to prevent
            double detection of barlines that consist of multiple parallel lines or very think lines.
         6) The Barlines are each used to find a prediction for a single uninterrupted line from the top to the bottom
            of the vertical system boundaries. By interpolating the parts in between segments that are unknown,
            small possible curves are followed as closely as possible. All vertical runs from the original skeleton list
            for which at least 80% of the pixels fall into a margin of 2 * staffline_space of the corresponding
            predicted pixel are selected as segments for the barline. These are again filtered to remove horizontally
            overlapping segments. The remaining segments constitute the enitire Barline.
         7) A final filtering step is done that removes Barlines that don't span most of the vertical height of the
            system or don't fill at least 50% of pixels in between their vertical limits. Since the longest segment,
            referenced in Step 3, can be larger than the other Barlines when they are the first Barline of the system,
            the filtering is based on the second longest segment when its min and max y position don't exceed that
            of the longest segment.
        :returns A list of Systems, each containing a list of Barlines, each containing a list of BarlineSegments.
        """
        # Step 1: Extract skeletons
        extractor = SkeletonExtractor(self.image, 'vertical')
        skeleton_list = extractor.get_skeleton_list(window=5)

        # Step 2: Filter skeletons by threshold length
        candidate_lines = [BarlineSegment(line[0], line[1]) for line in skeleton_list
                           if len(line[1]) >= self.len_threshold]
        candidate_lines.sort(key=lambda l: (l.y, len(l.x_values)))

        # Step 3: Group candidate lines into system groups based on overlapping y values.
        label = 1
        system_groups = []
        current_candidate = self.BarlineCandidateGroup()
        current_candidate.label = label
        for segment in candidate_lines:
            if segment.label > 0:
                continue
            longest = current_candidate.longest_segment
            if longest is not None and segment.y > longest.y + len(longest.x_values):
                # End of overlaps: start with new Barline
                system_groups.append(current_candidate)
                label += 1
                current_candidate = self.BarlineCandidateGroup()
                current_candidate.label = label
            segment.label = label
            current_candidate.add_segment(segment)
        system_groups.append(current_candidate)

        label = 1
        systems = []
        for system_group in system_groups:
            system_group.segments.sort(key=lambda s: s.x_values[0])

            # Step 4: Group segments further into barline groups. Groupings are made based on segment orientation.
            barlines = []
            current_barline = Barline()
            current_barline.label = label
            for i, segment in enumerate(system_group.segments):
                if len(current_barline.segments) > 0:
                    if self.segment_overlaps_group(current_barline, segment) \
                            or abs(self.group_to_segment_angle(current_barline, segment) - current_barline.angle) > 0.05 \
                            or abs(segment.angle - current_barline.angle) > 0.05:
                        barlines.append(current_barline)
                        label += 1
                        current_barline = Barline()
                        current_barline.label = label
                segment.label = label
                current_barline.add_segment(segment)
            barlines.append(current_barline)

            # Step 5: Remove grouped segments that are close together. The group with the largest summed length is kept.
            remove_barlines = []
            current_barline = barlines[0]
            current_x = current_barline.segments[0].x_values[0]
            for i in range(1, len(barlines)):
                barline = barlines[i]
                barline_x = barline.segments[0].x_values[0]
                if abs(current_x - barline_x) < 5 * self.staffspace_height:
                    if sum([len(s.x_values) for s in current_barline.segments]) > sum(
                            [len(s.x_values) for s in barline.segments]):
                        remove_barlines.append(barline)
                    else:
                        remove_barlines.append(current_barline)
                        current_barline = barline
                    current_x = barline_x
                else:
                    current_barline = barline
                    current_x = barline.segments[0].x_values[0]
            for barline in remove_barlines:
                barlines.remove(barline)

            # Step 6: Predict a barline spanning the entire height of the system group. All skeletons that are close to
            # this prediction are selected as being part of the final Barline.
            margin = 2 * self.staffline_height
            system = System()
            for barline in barlines:
                barline.interpolate(system_group.min_y, system_group.max_y)
                for skeleton in [skeleton for skeleton in skeleton_list if self.overlap(skeleton, system_group.min_y, system_group.max_y)]:
                    margin_count = 0
                    start_idx = max(0, skeleton[0] - system_group.min_y)
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

            # Step 7: Filter out false Barlines that don't span the entire system height, modulo a margin.
            system.barlines.sort(key=lambda b: sum([len(s.x_values) for s in b.segments]), reverse=True)

            remove_barlines = []
            min_y = system.barlines[0].min_y
            max_y = system.barlines[0].max_y
            if len(system.barlines) > 1:
                if system.barlines[1].min_y > min_y:
                    min_y = system.barlines[1].min_y
                if system.barlines[1].max_y < max_y:
                    max_y = system.barlines[1].max_y
            for barline in system.barlines:
                if barline.min_y - min_y > 5 * self.staffspace_height or \
                        max_y - barline.max_y > 5 * self.staffspace_height or \
                        sum([len(s.x_values) for s in barline.segments]) < 0.5 * (barline.max_y - barline.min_y):
                    remove_barlines.append(barline)
            for barline in remove_barlines:
                system.barlines.remove(barline)

            # Sort barlines and their segments from left to right
            for barline in system.barlines:
                barline.segments.sort(key=lambda s: s.y)
            system.barlines.sort(key=lambda b: b.segments[0].x_values[0])

            # Avoid false positives in systems. Systems with only one barline cannot exist in music.
            if len(system.barlines) > 1:
                systems.append(system)

        if debug > 0:
            extractor.skeleton_list_to_image()
            ImageOps.invert(extractor.skeleton_image).save('skeleton.png')

            candidate_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for segment in candidate_lines:
                for row in range(len(segment.x_values)):
                    candidate_img[segment.y + row, segment.x_values[row] - 2:segment.x_values[row] + 2] = (255, 0, 0)
            Image.fromarray(candidate_img).save('candidate_barlines.png')

            grouped_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for barline in system_groups:
                color = (255, barline.label * 50, 0)
                for segment in barline.segments:
                    for row in range(len(segment.x_values)):
                        grouped_img[segment.y + row, segment.x_values[row] - 2:segment.x_values[row] + 2] = color
            Image.fromarray(grouped_img).save('grouped_candidate_barlines.png')

            prediction_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for system in systems:
                for barline in system.barlines:
                    color = (150 + (31 * barline.label) % 106,
                             150 + (111 * (barline.label + 1)) % 106,
                             150 + (201 * (barline.label + 2)) % 106)
                    for row in range(len(barline.prediction)):
                        prediction_img[barline.prediction_y_start + row,
                        barline.prediction[row] - 2:barline.prediction[row] + 2] = color
            Image.fromarray(prediction_img).save('barline_predictions.png')

            barline_img = np.array(ImageOps.invert(extractor.skeleton_image).convert('RGB'))
            for system in systems:
                for barline in system.barlines:
                    color = (150 + (31 * barline.label) % 106,
                             150 + (111 * (barline.label + 1)) % 106,
                             150 + (201 * (barline.label + 2)) % 106)
                    for segment in barline.segments:
                        for row in range(len(segment.x_values)):
                            barline_img[segment.y + row, segment.x_values[row] - 4:segment.x_values[row] + 4] = color
            Image.fromarray(barline_img).save('barlines.png')

        if self.output_path is not None:
            with open(self.output_path, 'w') as f:
                json.dump({'systems': [system.to_json() for system in systems]}, f, sort_keys=True, indent=2)
        return systems


class BarlineSegment:
    """
    Data class that represents a barline segment.
    """

    def __init__(self, y, x_values):
        self.y = y
        self.x_values = x_values
        self.angle = (x_values[0] - x_values[-1]) / len(x_values)
        self.label = 0

    @staticmethod
    def from_json(json_data):
        segment = BarlineSegment(json_data['y'], json_data['x_values'])
        return segment

    def to_json(self):
        return {'y': self.y, 'x_values': self.x_values}


class Barline:
    """
    Helper class that represents a Barline. A Barline consists of several BarlineSegments.
    """

    def __init__(self):
        self.label = 0
        self.segments = []
        self.min_y = 0
        self.max_y = 0
        self.angle = 0.0
        self.prediction = []
        self.prediction_y_start = 0

    def add_segment(self, segment):
        first = len(self.segments) == 0
        self.segments.append(segment)
        if first or segment.y < self.min_y:
            self.min_y = segment.y
        if first or segment.y + len(segment.x_values) > self.max_y:
            self.max_y = segment.y + len(segment.x_values)
        self.angle = sum([s.angle for s in self.segments]) / len(self.segments)

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
        if len(self.segments) > 0:
            self.angle = sum([s.angle for s in self.segments]) / len(self.segments)
        else:
            self.angle = 0.0

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

    def get_average(self):
        coords = []
        for segment in self.segments:
            coords.extend(segment.x_values)
        return int(sum(coords) / len(coords))

    @staticmethod
    def from_json(json_data):
        barline = Barline()
        barline.min_y = json_data['top']
        barline.max_y = json_data['bottom']
        barline.segments = [BarlineSegment.from_json(segment) for segment in json_data['segments']]
        return barline

    def to_json(self):
        return {
            'segments': [segment.to_json() for segment in self.segments],
            'top': self.min_y,
            'bottom': self.max_y,
        }


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

    @staticmethod
    def from_json(json_data):
        system = System()
        system.min_y = json_data['top']
        system.max_y = json_data['bottom']
        system.barlines = [Barline.from_json(barline) for barline in json_data['barlines']]
        return system

    def to_json(self):
        return {
            'barlines': [barline.to_json() for barline in self.barlines],
            'top': self.min_y,
            'bottom': self.max_y,
        }


if __name__ == '__main__':
    barline_paths = data_dir / 'sample' / 'barlines'
    barline_paths.mkdir(parents=True, exist_ok=True)
    overlay_paths = data_dir / 'sample' / 'barline_overlays'
    overlay_paths.mkdir(parents=True, exist_ok=True)
    for image_path in tqdm((data_dir / 'sample' / 'pages').iterdir()):
        image = Image.open(image_path)
        detector = BarlineDetector(image, output_path=barline_paths / (image_path.stem + '.json'))
        systems = detector.detect_barlines(debug=0)
        score_draw = ScoreDraw(image)
        barline_img = score_draw.draw_systems(systems)
        barline_img.save(data_dir / image_path.name)
