from pathlib import Path

root_dir = Path(__file__).parents[1].absolute()
data_dir = root_dir / 'data'
eval_dir = root_dir / 'evaluation'
tmp_dir = root_dir / 'tmp'

segmenter_data_dir = Path(__file__).parents[2].absolute() / 'OMR-measure-segmenter-data'
musicdata_dir = segmenter_data_dir / 'musicdata'
stafffinder_testset_dir = segmenter_data_dir / 'stafffinder-testset'

measures_dir = 'measures'
measure_images_dir = 'measure_images'
score_mapping_path = 'score_mapping.json'
part_order_path = 'order.txt'
labels_path = 'cluster_labels.npy'
medoids_path = 'cluster_medoids.npy'

MUSIC_DATA_EXTENSIONS = ['.mxl', '.xml', '.musicxml']


def page_sort_key(path):
    return int(path.stem.split('_')[1])


def get_musicdata_scores(follow_parts=True, sort_reverse=False):
    scores = sorted([score for score in musicdata_dir.iterdir() if score.is_dir()], reverse=sort_reverse)
    combined = []
    for score in scores:
        parts = sorted([d for d in score.iterdir() if d.is_dir() and 'part_' in d.name])
        if follow_parts and len(parts) > 0:
            combined.extend(parts)
        else:
            combined.append(score)
    return combined


def get_parts(score):
    return sorted([part for part in score.iterdir() if part.is_dir() and part.name.startswith('part_')])


def get_score_path(score):
    music_path = next((f for f in score.iterdir() if f.suffix in MUSIC_DATA_EXTENSIONS), None)
    if music_path is None or not music_path.is_file():
        music_path = score / 'stage2s'
        if not music_path.is_dir():
            raise FileNotFoundError('Could not find supported music data in directory ' + str(score))
    return music_path


def get_score_dirs(score):
    dirs = {}
    parts = get_parts(score)
    if len(parts) > 0:
        dirs['measures'] = [score / part.name / measures_dir for part in parts]
        dirs['measure_images'] = [score / part.name / measure_images_dir for part in parts]
        dirs['score_mapping'] = [score / part.name / score_mapping_path for part in parts]
        dirs['part_order'] = [score / part.name / part_order_path for part in parts]
        dirs['score_path'] = [get_score_path(part) for part in parts]
    else:
        dirs['measures'] = [score / measures_dir]
        dirs['measure_images'] = [score / measure_images_dir]
        dirs['score_mapping'] = [score / score_mapping_path]
        dirs['parts_order'] = [score / part_order_path]
        dirs['score_path'] = get_score_path(score)
    dirs['labels'] = score / labels_path
    dirs['medoids'] = score / medoids_path
    return dirs
