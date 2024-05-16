import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from score_analysis.measure_clusterer import MeasureImage
from util.dirs import musicdata_dir


def draw_profile(profile):
    plt.plot(profile, 'k')
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.show()
    plt.savefig('profile_plot.png')


part = musicdata_dir / 'beethoven_symphony_1' / 'part_1'

image = Image.open(part / 'measure_images' / 'page_3' / 'system_1_measure_85.png')
measure = MeasureImage(None, image, 0, 0, 0, 0)
profile_img = np.ones((measure.image.height, measure.image.width)) * 255
for i in range(measure.profile.size):
    s = (profile_img.shape[0] - 1 - measure.profile[i])
    profile_img[s-5:s+5, i] = 0
profile_img = Image.fromarray(profile_img)
draw_profile(measure.profile)

res = Image.new('L', (image.width, image.height + profile_img.height + 20), 255)
res.paste(image, (0, 0))
res.paste(profile_img, (0, image.height + 20))
res.save('measure_profile.png')
