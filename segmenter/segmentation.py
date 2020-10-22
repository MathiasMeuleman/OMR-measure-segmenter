import warnings
from functools import cmp_to_key
from posixpath import join

import numpy as np
import tensorflow as tf
from PIL import Image

from segmenter.image_util import data_dir, overlay_segments, root_dir
from segmenter.util import consecutive, compare_measure_bounding_boxes, preprocess

warnings.filterwarnings('ignore')

graph_initialized = False
sess = None


def initialize_graph():
    global graph_initialized, sess
    detection_graph = tf.Graph()
    detection_graph.as_default()
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('segmenter/model.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session()
    graph_initialized = True


def infer(image: np.ndarray):
    if not graph_initialized:
        initialize_graph()
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections',
        'detection_boxes',
        'detection_scores',
        'detection_classes'
    ]:
        tensor_name = key + ':0'

        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict


def filter_overlapping_segments(measures):
    """
    Results from the Pacha segmenter can sometimes include a "measure" that overlaps a number of other measures.
    This is filtered out here.
    NOTE! This method assumes the measures to be sorted according to the Pacha measure sorting method.
    :param measures:
    :return:
    """
    measures = np.asarray(measures)
    measures_ulx = np.array([measure['ulx'] for measure in measures])
    measures_uly = np.array([measure['uly'] for measure in measures])
    measures_lrx = np.array([measure['lrx'] for measure in measures])
    measures_lry = np.array([measure['lry'] for measure in measures])
    passed = []
    for i, measure in enumerate(measures):
        overlap_hor = (measures_ulx > measure['ulx']) & (measures_lrx < measure['lrx'])
        overlap_vert = ((measure['uly'] < measures_uly) & (measure['lry'] > (measures_uly + (measures_lry - measures_uly) / 2))) |\
                       ((measure['lry'] > measures_lry) & (measure['uly'] < (measures_uly + (measures_lry - measures_uly) / 2)))
        in_measures = measures[overlap_hor & overlap_vert]
        if len(in_measures) == 0:
            passed.append(measure)
    return passed


def slice_horizontal(image: np.ndarray, vertical_measures):
    image = preprocess(image)
    profile = image.sum(axis=1) / 255
    # Determine the mid-points in between bars, use them as candidate cutpoints.
    peakranges = consecutive(np.where(profile > profile.mean() + profile.std())[0], stepsize=60)
    peaks = np.array([np.round(np.median(peak)) for peak in peakranges])
    cuts = np.round((peaks[1:] + peaks[:-1]) / 2).astype('int32')
    deviation = int(np.round(np.min(np.diff(cuts)) / 4))

    measures = []
    for measure in vertical_measures:
        # Tweak the candidate cutpoints for each vertical segment, to minimize the risk of cut-offs
        vertical_profile = image[:, int(measure['ulx'] * image.shape[1]):int(measure['lrx'] * image.shape[1])].sum(axis=1) / 255
        horizontal_cuts = []
        for cut in cuts:
            region_profile = vertical_profile[cut - deviation:cut + deviation]
            # Cut at index with smallest cut-off value (least amount of black pixels intersected).
            min_cutoff_val = np.min(region_profile)
            candidate_horizontal_cuts = np.where(region_profile < np.ceil(min_cutoff_val) + 3)[0]
            # Use index closest to the original cut point, to bias towards the center between two bars
            horizontal_cut_idx = candidate_horizontal_cuts[(np.abs(candidate_horizontal_cuts - deviation)).argmin()]
            horizontal_cuts.append(cut - deviation + horizontal_cut_idx)

        horizontal_cuts = np.asarray(horizontal_cuts)
        horizontal_cuts = horizontal_cuts / image.shape[0]

        cut_positions = horizontal_cuts[(horizontal_cuts > measure['uly']) & (horizontal_cuts < measure['lry'])].tolist()
        cut_positions.insert(0, measure['uly'])
        cut_positions.insert(len(cut_positions), measure['lry'])
        for i in range(len(cut_positions)):
            if i + 1 < len(cut_positions):
                measures.append({'ulx': measure['ulx'], 'uly': cut_positions[i], 'lrx': measure['lrx'], 'lry': cut_positions[i+1]})

    return measures


def segment_image(image: np.array):
    (image_width, image_height, _) = image.shape

    output_dict = infer(image)

    measures = []

    for idx in range(output_dict['num_detections']):
        if output_dict['detection_classes'][idx] == 1 and output_dict['detection_scores'][idx] > 0.5:
            y1, x1, y2, x2 = output_dict['detection_boxes'][idx]

            measures.append({
                'ulx': x1,
                'uly': y1,
                'lrx': x2,
                'lry': y2
            })
        else:
            break

    measures.sort(key=cmp_to_key(compare_measure_bounding_boxes))
    # measures = filter_overlapping_segments(measures)
    # measures = slice_horizontal(image, measures)

    return {'measures': measures}


def measures_to_numpy(measures):
    """
    Transform the measures to a numpy format which can easily be saved. This format is of the form
    [ulx, uly, lrx, lry] for each measure, indicating the rectangular bounding box for that measure.
    Note that the values indicate the fraction of the page (width/height), instead of the absolute (x/y).
    :param measures:
    :return:
    """
    measures_np = np.zeros((len(measures), 4))
    for i, measure in enumerate(measures):
        measures_np[i, :] = [measure['ulx'], measure['uly'], measure['lrx'], measure['lry']]
    return measures_np


def segment(page, img_source_dir=join(data_dir, 'ppm-600'), measure_output_dir=join(data_dir, 'ppm-600-segments')):
    """
    Segments a given page into measures. The individual measure boundaries are stored in a npz file
    :return:
    """
    path = join(img_source_dir, 'transcript-{}.png'.format(page))
    image = Image.open(path)
    segmented = segment_image(np.asarray(image))
    measures = measures_to_numpy(segmented['measures'])
    np.save(join(measure_output_dir, 'transcript-{}.npy'.format(page)), measures)


def main():
    img_dir = join(data_dir, 'Mahler_Symphony_1/ppm-600')
    measure_dir = join(root_dir, 'tmp/Mahler_Symphony_1/ppm-600-segments')
    for page in range(1, 30):
        segment(page, img_source_dir=img_dir, measure_output_dir=measure_dir)
        overlay_segments(page, img_source_dir=img_dir, measures_source_dir=measure_dir)


if __name__ == "__main__":
    main()
