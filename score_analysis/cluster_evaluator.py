import json
from itertools import groupby

import math
import numpy as np
from PIL import Image, ImageOps
from music21 import converter
from operator import attrgetter, itemgetter
from sklearn.metrics.cluster import homogeneity_completeness_v_measure

from score_analysis.measure_clusterer import MeasureImage
from score_analysis.measure_detector import Measure
from score_analysis.score_mapping import ScoreMappingPage
from util.dirs import get_score_dirs, get_musicdata_scores
from util.table_generator import generate_table, TableColumn


class ClusterMeasure(MeasureImage):

    def __init__(self, measure, image, page, system, bar, staff, part, score_measures):
        super(ClusterMeasure, self).__init__(measure, image, page, system, bar, staff)
        self.idx = -1
        self.label = -1
        self.part = part
        self.score_measures = score_measures
        self.rhythm = None

    def __str__(self):
        return 'Idx: {}\tPart: {}\tPage: {}\tSystem: {}\tBar: {}\tStaff: {}'.format(self.idx, self.part, self.page, self.system, self.bar, self.staff)


class ClusterEvaluator:

    def __init__(self, score_dirs):
        self.measures_path = score_dirs['measures']
        self.measure_images_path = score_dirs['measure_images']
        self.labels_path = score_dirs['labels']
        self.medoids_path = score_dirs['medoids']
        self.score_path = score_dirs['score_path']
        self.score_mapping_path = score_dirs['score_mapping']
        self.part_order_path = score_dirs['parts_order']
        self.labels = None
        self.sub_labels = None
        self.medoids = None
        self.sub_medoids = None
        self.measures = None
        self.score_mapping = None
        self.score = None
        self.parts = None

    def get_measure(self, page, system, bar, staff):
        return next((m for m in self.measures if m.page == page and m.system == system and m.bar == bar and m.staff == staff))

    def load_measures(self):
        print('Loading measures...')
        self.measures = []
        page_offset = 0
        for i, (measures_path, images_path, parts, score_mapping) in enumerate(zip(self.measures_path, self.measure_images_path, self.parts, self.score_mapping)):
            if not measures_path.exists():
                raise FileNotFoundError('No measures folder found at: {}'.format(self.measures_path))
            if not images_path.exists():
                raise FileNotFoundError('No images folder found at: {}'.format(self.measure_images_path))
            part_nr = i + 1
            for images_path in sorted(images_path.iterdir()):
                page_path = measures_path / (images_path.stem + '.json')
                if not images_path.exists():
                    raise FileNotFoundError('No image found at: {}'.format(images_path))
                page_nr = int(page_path.stem.split('_')[1])
                with open(page_path) as f:
                    measures = [Measure.from_json(json_data) for json_data in json.load(f)['measures']]
                measures.sort(key=attrgetter('system', 'start', 'top'))
                systems = [list(v) for k, v in groupby(measures, key=attrgetter('system'))]
                for j, system in enumerate(systems):
                    system_nr = j + 1
                    bar = 0
                    staff = 0
                    prev_top = system[0].top + 1
                    for k, measure in enumerate(system):
                        if measure.top < prev_top:
                            bar += 1
                            staff = 1
                        else:
                            staff += 1
                        prev_top = measure.top
                        image = Image.open(images_path / 'system_{}_measure_{}.png'.format(j + 1, k + 1))
                        score_measures = []
                        mapping_page = next((p for p in score_mapping if p.pagenumber == page_nr))
                        mapping_system = next((s for s in mapping_page.systems if s.systemnumber == system_nr))
                        if staff > len(mapping_system.staffs):
                            continue
                        mapping_staff = next((s for s in mapping_system.staffs if s.staffnumber == staff))
                        measure_idx = mapping_system.measurestart + bar - 1
                        staff_parts = [part for part in parts if part.id in [p.id for p in mapping_staff.parts]]
                        for part in staff_parts:
                            score_measures.append(part.measure(measure_idx, collect=[], indicesNotNumbers=True))
                        self.measures.append(ClusterMeasure(measure, image, page_offset + page_nr, system_nr, bar, staff, part_nr, score_measures))
            # Use measures_path dir length here because images_path dir length can have missing pages.
            page_offset += len(list(measures_path.iterdir()))
        self.measures.sort(key=attrgetter('page', 'system', 'bar', 'staff'))
        for j, measure in enumerate(self.measures):
            measure.idx = j
            measure.label = self.labels[j]

    def load_score_mappings(self):
        score_mappings = []
        for path in self.score_mapping_path:
            with open(path) as f:
                score_mappings.append([ScoreMappingPage.from_json(json_data) for json_data in json.load(f)['pages']])
        self.score_mapping = score_mappings

    def load_scores(self):
        parts = []
        for path in self.score_path:
            parts.append(list(converter.parse(path).recurse(classFilter='Part')))
        self.parts = parts

    def load(self):
        self.labels = np.load(self.labels_path).tolist()
        self.medoids = np.load(self.medoids_path).tolist()
        self.load_score_mappings()
        self.load_scores()
        self.load_measures()

    def encode_rhythms(self):
        ppq = 24  # pulses per quarter
        current_time_signature = self.measures[0].score_measures[0].timeSignature
        if current_time_signature is None:
            current_time_signature = self.measures[0].score_measures[0].getTimeSignatures(returnDefault=False).timeSignature
        if current_time_signature is None:
            raise ValueError('Could not find first time signature')

        for measure in self.measures:
            if measure.score_measures[0].timeSignature is not None:
                current_time_signature = measure.score_measures[0].timeSignature
            rhythms = []
            for score_measure in measure.score_measures:
                rhythm = []
                position = 0.0
                for note in score_measure.notesAndRests:
                    length = round(current_time_signature.denominator / 4 * ppq * note.quarterLength, 1)
                    rhythm.append((position, length, not note.isRest))
                    position += length
                rhythms.append(rhythm)
            combined_rhythm = rhythms[0]
            if len(rhythms) > 1:
                for rhythm in rhythms[1:]:
                    if rhythm == combined_rhythm:
                        continue
                    combined_rhythm.extend(rhythm)
                    combined_rhythm.sort(key=itemgetter(0, 1, 2))
            measure.rhythm = combined_rhythm

    def generate_measures_image(self, measures):
        border_width = 4
        image_width = sum(m.image.width for m in measures[0:10]) + 10 * border_width
        image = Image.new('L', (image_width, image_width), 255)

        cur_x = 0
        cur_y = 0
        largest_height = 0

        for measure in measures:
            measure_image = ImageOps.expand(measure.image, border=border_width)
            if cur_x + measure_image.width >= image.width:
                cur_x = 0
                cur_y += largest_height
                largest_height = 0
            if cur_y + measure_image.height >= image.height:
                image = ImageOps.pad(image, (image.width, image.height + image_width), color=255, centering=(0, 0))
            image.paste(measure_image, (cur_x, cur_y))
            cur_x += measure_image.width
            largest_height = max(largest_height, measure_image.height)

        image = image.crop((0, 0, image.width, cur_y + largest_height))
        return image

    def combine_medoids(self, clustered_measures):
        medoids = []
        for cluster in clustered_measures:
            medoids.append(self.measures[self.medoids[cluster[0].label]])

        medoid_rhythms = [m.rhythm for m in medoids]
        same_medoids = []
        for i in range(len(medoid_rhythms)):
            if medoid_rhythms[i] in medoid_rhythms[0:i]:
                idx = medoid_rhythms.index(medoid_rhythms[i])
                same_medoids.append((i, idx))
        for (source, target) in same_medoids:
            clustered_measures[target].extend(clustered_measures[source])
        return [clustered_measures[i] for i in range(len(clustered_measures)) if i not in [s for (s, _) in same_medoids]]

    def homogeneity_completeness_v_measure(self, measures):
        pred_labels = [m.label for m in measures]
        true_labels = [m.true_label for m in measures]
        homogeinity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, pred_labels)
        return homogeinity, completeness, v_measure

    def evaluate_clusters(self, combine_same_medoids=False):
        clustered_measures = [list(v) for k, v in groupby(sorted(self.measures, key=attrgetter('label')), key=attrgetter('label'))]
        rhythm_measures = [list(v) for k, v in groupby(sorted(self.measures, key=attrgetter('rhythm')), key=attrgetter('rhythm'))]
        rhythm_measures.sort(key=len)
        for i, group in enumerate(rhythm_measures):
            for measure in group:
                measure.true_label = i

        if combine_same_medoids:
            clustered_measures = self.combine_medoids(clustered_measures)

        result_message = self.labels_path.parent.name
        homogeneity, completeness, v_measure = self.homogeneity_completeness_v_measure([m for g in rhythm_measures for m in g])
        result_message += '\nHomogenity: {}\tCompleteness: {}'.format(homogeneity, completeness)

        total_correct_measures = 0
        total_sub_correct_measures = 0
        for cluster in clustered_measures:
            label = cluster[0].label
            medoid = self.measures[self.medoids[label]]
            correct_measures = [m for m in cluster if m.rhythm == medoid.rhythm]
            total_correct_measures += len(correct_measures)
            result_message += '\n\n\t{}'.format(label)
            result_message += '\tSame as medoid: {}/{}\t({}%)'.format(len(correct_measures), len(cluster), round(len(correct_measures) / len(cluster) * 100, 2))

            sorted_measures = sorted(cluster, key=attrgetter('rhythm'))
            grouped_measures = [list(v) for k, v in groupby(sorted_measures, key=attrgetter('rhythm'))]
            grouped_measures.sort(key=len, reverse=True)

            result_message += '\n\t\tSymbolic similarities:'
            for group in grouped_measures:
                result_message += '\n\t\t\t{} ({}%)\t{}\t{}'.format(len(group), round(len(group) / len(cluster) * 100, 2), 'x' if group[0].rhythm == medoid.rhythm else '', group[0].rhythm)

        total_measures = len(self.measures)
        result_message += '\ntotal avg {}/{} ({}%)'.format(total_correct_measures, total_measures, round(total_correct_measures / total_measures * 100, 2))
        result_message += '\ntotal sub avg {}/{} ({}%)'.format(total_sub_correct_measures, total_measures, round(total_sub_correct_measures / total_measures * 100, 2))
        with open(self.labels_path.parent / 'cluster_evaluation.txt', 'w') as f:
            f.write(result_message)
        return homogeneity, completeness, v_measure


if __name__ == '__main__':
    # score = musicdata_dir / 'bach_brandenburg_concerto_5_part_1'
    # score = musicdata_dir / 'beethoven_symphony_4'
    datapoints = []
    columns = [TableColumn('Homogeneity', 'homogeneity', 'r'), TableColumn('Completeness', 'completeness', 'r'), TableColumn('V measure', 'v_measure', 'r')]
    for score in get_musicdata_scores(follow_parts=False):
        print(score.name)
        dirs = get_score_dirs(score)
        evaluator = ClusterEvaluator(dirs)
        evaluator.load()
        evaluator.encode_rhythms()
        h, c, v = evaluator.evaluate_clusters()
        datapoints.append({'score': score.name, 'homogeneity': round(h, 3), 'completeness': round(c, 3), 'v_measure': round(v, 3)})

    print(generate_table(columns, datapoints))
