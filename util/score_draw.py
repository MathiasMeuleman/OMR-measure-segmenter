import numpy as np
from PIL import Image, ImageColor, ImageDraw


class ScoreDraw:

    def __init__(self, image):
        if not isinstance(image, Image.Image):
            raise ValueError('Expected PIL Image object')
        self.image = image.convert('RGB')

    colors = ['red', 'orange', 'green', 'blue', 'purple']

    def draw_staffs(self, staffs):
        image = self.image.copy()
        draw = ImageDraw.Draw(image, mode='RGBA')
        for i, staff in enumerate(staffs):
            color = ImageColor.getrgb(self.colors[i % len(self.colors)])
            for line in staff.stafflines:
                draw.line(((line.start, line.y), (line.end, line.y)), fill=color, width=5)
        del draw
        return image

    def draw_systems(self, systems):
        image = self.image.copy()
        barline_img = np.array(image)
        for system in systems:
            for i, barline in enumerate(system.barlines):
                color = ImageColor.getrgb(self.colors[i % len(self.colors)])
                for segment in barline.segments:
                    for row in range(len(segment.x_values)):
                        barline_img[segment.y + row, segment.x_values[row] - 4:segment.x_values[row] + 4] = color
        return Image.fromarray(barline_img)

    def draw_measures(self, measures):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        for measure in measures:
            color = ImageColor.getrgb(self.colors[measure.system % len(self.colors)])
            draw.rectangle(((measure.start, measure.top), (measure.end, measure.bottom)), outline=color, width=5)
        del draw
        return image

    def draw_measures_bb(self, measures):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        for measure in measures:
            color = ImageColor.getrgb(self.colors[measure.system % len(self.colors)])
            draw.rectangle(((measure.bb[0], measure.bb[1]), (measure.bb[2], measure.bb[3])), outline=color, width=5)
        del draw
        return image
