import numpy as np
from PIL import Image, ImageColor, ImageDraw


class ScoreDraw:

    def __init__(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError('Expected PIL Image object')
        self.image = image.convert('RGB')

    colors = ['red', 'orange', 'green', 'blue', 'purple']

    def draw_staffs(self, staffs):
        draw = ImageDraw.Draw(self.image, mode='RGBA')
        for i, staff in enumerate(staffs):
            color = ImageColor.getrgb(self.colors[i % len(self.colors)])
            for line in staff.stafflines:
                draw.line(((line.start, line.y), (line.end, line.y)), fill=color, width=5)
        del draw
        return self.image

    def draw_systems(self, systems):
        barline_img = np.array(self.image)
        for system in systems:
            for i, barline in enumerate(system.barlines):
                color = ImageColor.getrgb(self.colors[i % len(self.colors)])
                for segment in barline.segments:
                    for row in range(len(segment.x_values)):
                        barline_img[segment.y + row, segment.x_values[row] - 2:segment.x_values[row] + 2] = color
        return Image.fromarray(barline_img)

    def draw_measures(self, measures):
        draw = ImageDraw.Draw(self.image)
        for measure in measures:
            color = ImageColor.getrgb(self.colors[measure.system % len(self.colors)])
            draw.rectangle(((measure.start, measure.top), (measure.end, measure.bottom)), outline=color, width=5)
        del draw
        return self.image
