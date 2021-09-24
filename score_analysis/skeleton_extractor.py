import numpy as np
from queue import SimpleQueue
from PIL import Image

from score_analysis.score_image import ScoreImage
from util.dirs import data_dir


class SkeletonExtractor:

    def __init__(self, image, direction):
        if direction not in ['vertical', 'horizontal']:
            raise ValueError('Value of parameter "orientation" should be either "vertical" or "horizontal"')
        self.direction = direction
        self.score_image = image if isinstance(image, ScoreImage) else ScoreImage(image)
        self.image = self.score_image.wb_image
        self.np_image = np.asarray(self.image)
        self.black_value = self.np_image.max()
        self.white_value = 0
        self.staffline_height = self.score_image.staffline_height
        self.staffspace_height = self.score_image.staffspace_height
        self.black_runs = None
        self.skeleton_list = None
        self.skeleton_image = None

    def extract_black_runs(self, window=3, blackness=60):
        """
        Extract horizontal black runs. A 1D window of size `window` * staffspace_height
        is layed over each pixel. That pixel is set to active if the contents of the
        window is over `blackness` percent black.
        :param window: The window size is defined as `window` * staffspace_height.
        :param blackness: The blackness threshold in percentage (0-100).
        """
        threshold = blackness / 100.
        np_image = self.np_image if self.direction == 'horizontal' else self.np_image.T
        dest_img = np.full(np_image.shape, self.white_value, dtype=np.uint8)
        window_size = round(self.staffspace_height * window)
        half_window = window_size // 2
        rows, cols = np_image.shape
        for i in range(rows):
            image_row = np_image[i, :]
            queue_window = SimpleQueue()
            w_sum = 0

            idx = 0
            # Left padding
            while idx <= half_window:
                queue_window.put(self.white_value)
                w_sum += self.white_value
                idx += 1
            window_center = 0
            window_end = 0
            # Fill window until window center is at first image row pixel
            while idx < window_size:
                queue_window.put(image_row[window_end])
                w_sum += image_row[window_end]
                idx += 1
                window_end += 1

            first = True
            while window_center < image_row.shape[0]:
                # Remove first pixel and push next pixel, or padding if outside image row boundary
                w_sum -= queue_window.get()
                if idx >= image_row.shape[0]:
                    queue_window.put(self.white_value)
                    w_sum += self.white_value
                else:
                    queue_window.put(image_row[window_end])
                    w_sum += image_row[window_end]

                w_avg = w_sum / window_size
                if w_avg / 255 >= threshold:
                    dest_img[i, window_center] = self.black_value
                    if first:
                        # The first black pixel usually lies on a black path.
                        # Rescue black pixels that preceed the black pixel that flipped the threshold condition.
                        first = False
                        col_back = window_center - 1
                        while True:
                            if col_back < 0 or image_row[col_back] != self.black_value:
                                break
                            dest_img[i, col_back] = self.black_value
                            col_back -= 1
                else:
                    # Rescue black pixels on the path that lie after the pixel that flipped the threshold condition.
                    # If `first` was set to `False`, we are on a black path
                    if not first:
                        if image_row[window_center] == self.black_value:
                            dest_img[i, window_center] = self.black_value
                        else:
                            # When we encounter a white pixel, end the black path
                            first = True

                idx += 1
                window_center += 1
                window_end += 1

        self.black_runs = dest_img

    @staticmethod
    def interpolate(y_values, start_pos):
        """
        Interpolate y_values from start_pos on. Interpolation is done in-place,
        along the angle between the point at start_pos and the last point in y_values
        :param y_values: The y values for the line
        :param start_pos: The index at which to start the interpolation
        """
        y_values_size = len(y_values)
        start_middle = y_values[start_pos]
        current_middle = y_values[-1]

        # Compute the angle
        y_diff = current_middle - start_middle
        x_diff = y_values_size - start_pos
        tan_a = y_diff / x_diff

        # Interpolate middle values along the computed angle
        for i in range(y_values_size - start_pos):
            new_y_value = int(i * tan_a + start_middle)
            y_values[i + start_pos] = new_y_value

    def label_pixels(self, image, top, middle, bottom, col, label):
        """
        Label all pixels in the image between top and bottom,
        limited to 2 * staffline_height, with label `label`.
        """
        decision_plus = min(image.shape[0] - 1, middle + 2 * self.staffline_height)
        decision_minus = max(0, middle - 2 * self.staffline_height)

        b = min(bottom, decision_plus)
        t = max(top, decision_minus)

        for i in range(t, b + 1):
            image[i, col] = label

    def down(self, image, col, row, middle_prev, neighbor):
        had_neighbor = neighbor is not None
        nearest_distance = image.shape[0]
        bottom = row + 1
        if col >= image.shape[1] - 1:
            while bottom < image.shape[0] and image[bottom, col] == self.black_value:
                bottom += 1
        elif had_neighbor:
            nearest_distance = abs(middle_prev - neighbor)
            while bottom < image.shape[0] and image[bottom, col] == self.black_value:
                if image[bottom, col + 1] == self.black_value:
                    distance = abs(middle_prev - bottom)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        neighbor = bottom
                bottom += 1
        else:
            while bottom < image.shape[0] and image[bottom, col] == self.black_value:
                if image[bottom, col + 1] == self.black_value:
                    if had_neighbor:
                        distance = abs(middle_prev - bottom)
                        if distance < nearest_distance:
                            nearest_distance = distance
                            neighbor = bottom
                    else:
                        nearest_distance = abs(middle_prev - bottom)
                        neighbor = bottom
                bottom += 1

        return bottom, neighbor

    def up(self, image, col, row, middle_prev):
        neighbor = None
        nearest_distance = image.shape[0]
        last_col = True
        if col < image.shape[1]:
            last_col = False
            if middle_prev != 0 and image[middle_prev, col + 1] == self.black_value:
                neighbor = middle_prev

        top = row
        if neighbor is not None or last_col:
            while top > 0 and image[top, col] == self.black_value:
                top -= 1
        else:
            while top > 0 and image[top, col] == self.black_value:
                if image[top, col + 1] == self.black_value:
                    if neighbor is not None:
                        distance = abs(middle_prev - top)
                        if distance < nearest_distance:
                            nearest_distance = distance
                            neighbor = top
                    else:
                        nearest_distance = abs(middle_prev - top)
                        neighbor = top
                top -= 1
        return top, neighbor

    def get_vertical_segment(self, image, col, row, middle_prev):
        """
        Find the vertical segment belonging to pixel (row, col) in the image.
        The segment is vertically bound by the `top` and `bottom` return values.
        Additionally, the column to the right of `col` is scanned for an active pixel
        closest to `middle_prev`, which will be returned as `neighbor`.
        """
        last_col = col + 1 >= image.shape[1]
        top = bottom = row
        neighbor = None
        nearest_distance = image.shape[0]
        while bottom < image.shape[0] and image[bottom, col] == self.black_value:
            if not last_col and image[bottom, col + 1] == self.black_value:
                distance = abs(middle_prev - bottom)
                if distance < nearest_distance:
                    nearest_distance = distance
                    neighbor = bottom
            bottom += 1
        while top > 0 and image[top, col] == self.black_value:
            if not last_col and image[top, col + 1] == self.black_value:
                distance = abs(middle_prev - top)
                if distance < nearest_distance:
                    nearest_distance = distance
                    neighbor = top
            top -= 1
        return top, bottom, neighbor

    @staticmethod
    def compute_middle(top, bottom):
        return top + (bottom - top) // 2

    def get_middle(self, top, middle_prev, bottom):
        guessed = False
        wall = False
        staffheight_tolerance = 0.75 * self.staffline_height

        if middle_prev == 0:
            middle = SkeletonExtractor.compute_middle(top, bottom)
        elif top <= middle_prev <= bottom:
            if bottom - top < 2 * staffheight_tolerance:
                middle = SkeletonExtractor.compute_middle(top, bottom)
            elif bottom <= middle_prev + staffheight_tolerance:
                middle = max(0, bottom - staffheight_tolerance)
            else:
                diff = max(0, middle_prev - staffheight_tolerance)
                if top >= diff:
                    middle = top + staffheight_tolerance
                else:
                    guessed = True
                    middle = middle_prev
        else:
            wall = True
            if middle_prev < top:
                if bottom - top >= 2 * staffheight_tolerance:
                    middle = top + staffheight_tolerance
                else:
                    middle = SkeletonExtractor.compute_middle(top, bottom)
            elif middle_prev > bottom:
                if bottom - top >= 2 * staffheight_tolerance:
                    middle = bottom - staffheight_tolerance
                else:
                    middle = SkeletonExtractor.compute_middle(top, bottom)
            else:
                middle = SkeletonExtractor.compute_middle(top, bottom)
        return int(middle), guessed, wall

    def vertical_thinning(self, image):
        skeleton_list = []
        label = 1
        for col in range(image.shape[1]):
            for row in range(image.shape[0]):
                # This pixel has been scanned before, or is white
                if image[row, col] != self.black_value:
                    continue
                y_values = []
                line = [col, y_values]

                # Middle is still undefined at start of a new line
                middle = 0
                guessed_prev = False
                start_pos = 0

                current_col = col
                current_row = row

                label += 1

                while True:
                    # top, neighbor = self.up(image, current_col, current_row, middle)
                    # bottom, neighbor = self.down(image, current_col, current_row, middle, neighbor)
                    top, bottom, neighbor = self.get_vertical_segment(image, current_col, current_row, middle)
                    middle, guessed, wall = self.get_middle(top, middle, bottom)

                    y_values.append(middle)

                    # -------------
                    # Special cases
                    # -------------

                    y_values_size = len(y_values)
                    if wall:
                        if y_values_size < 6 * self.staffline_height + 1:
                            start_pos = 0
                        else:
                            start_pos = y_values_size - 1 - 6 * self.staffline_height

                        self.interpolate(y_values, start_pos)
                        guessed = False

                    elif guessed and not guessed_prev:
                        start_pos = y_values_size - 1
                    elif not guessed and guessed_prev:
                        self.interpolate(y_values, start_pos)

                    guessed_prev = guessed

                    # Label all pixels in this section
                    self.label_pixels(image, top, middle, bottom, current_col, label)

                    # Update current col and row
                    current_row = neighbor
                    current_col += 1

                    if neighbor is None:
                        break

                skeleton_list.append(line)
        skeleton_list.sort(key=lambda s: (s[1][0], s[0]))
        self.skeleton_list = skeleton_list

    def get_skeleton_list(self, window=3, blackness=60):
        if self.skeleton_list is None:
            self.extract_black_runs(window, blackness)
            self.vertical_thinning(self.black_runs)
        return self.skeleton_list

    def get_skeleton_image(self, original, skeleton_list):
        original_img = np.array(original)
        image = np.full(original_img.shape, self.white_value, dtype=np.uint8)
        if self.direction == 'horizontal':
            for (x, y_values) in skeleton_list:
                for col in range(len(y_values)):
                    image[y_values[col], x + col] = self.black_value
        else:
            for (y, x_values) in skeleton_list:
                for row in range(len(x_values)):
                    image[y + row, x_values[row]] = self.black_value
        return Image.fromarray(image)

    def skeleton_list_to_image(self):
        self.skeleton_image = self.get_skeleton_image(self.np_image, self.skeleton_list)


if __name__ == '__main__':
    sample_dir = data_dir / 'sample' / 'pages'
    image_path = sample_dir / 'page_1.png'
    image = Image.open(image_path).convert('L')
    tracker = SkeletonExtractor(image, 'horizontal')
    skeleton_list = tracker.get_skeleton_list()
