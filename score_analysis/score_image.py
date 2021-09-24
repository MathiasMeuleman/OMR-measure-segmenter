import numpy as np
from collections import Counter
from PIL import Image
from skimage.filters import threshold_otsu


class ScoreImage:

    def __init__(self, image):
        self.image = Image.fromarray(np.asarray(image)).convert('L')

        # Thresholded and inverted image
        threshold = threshold_otsu(np.asarray(self.image))
        self.wb_image = self.image.point(lambda p: p <= threshold and 255)

        # Set staffline and staffspace measurements
        measurements = self.get_staff_measurements()
        self.staffline_height = measurements[0]
        self.staffspace_height = measurements[1]

    def rle(self, inarray):
        """
        run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: The found run lengths
        """
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return None, None, None
        else:
            y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element position
            z = np.diff(np.append(-1, i))  # run lengths
            return z

    def get_staff_measurements(self, black_value=255):
        """
        Get the staffline height and staffspace height from the given image.
        Calculation is done through run length encoding. The most frequent black run
        is used as the staffline height, the most frequent white run is used as the staffspace height.
        Assumes a black and white image. The value of black pixels can be passed.
        :return: The tuple (staffline_height, staffspace_height)
        """
        img = np.array(self.wb_image)
        rows, cols = img.shape
        black_runs = []
        white_runs = []
        for i in range(cols):
            col = img[:, i]
            runlengths = self.rle(col)
            starts_black = col[0] == black_value
            if starts_black:
                black_runs.extend(runlengths[0:len(runlengths):2])
                white_runs.extend(runlengths[1:len(runlengths):2])
            else:
                black_runs.extend(runlengths[1:len(runlengths):2])
                white_runs.extend(runlengths[0:len(runlengths):2])
        staffline_height = max(black_runs, key=Counter(black_runs).get)
        staffspace_height = max(white_runs, key=Counter(white_runs).get)
        return staffline_height, staffspace_height
