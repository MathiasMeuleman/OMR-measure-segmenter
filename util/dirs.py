from pathlib import Path

root_dir = Path(__file__).parents[1].absolute()
data_dir = root_dir / 'data'
eval_dir = root_dir / 'evaluation'
tmp_dir = root_dir / 'tmp'

segmenter_data_dir = Path(__file__).parents[2].absolute() / 'OMR-measure-segmenter-data'
musicdata_dir = segmenter_data_dir / 'musicdata'
stafffinder_testset_dir = segmenter_data_dir / 'stafffinder-testset'


def page_sort_key(path):
    return int(path.stem.split('_')[1])


def get_musicdata_scores(follow_parts=True):
    scores = sorted([score for score in musicdata_dir.iterdir() if score.is_dir()])
    combined = []
    for score in scores:
        parts = sorted([d for d in score.iterdir() if d.is_dir() and 'part_' in d.name])
        if follow_parts and len(parts) > 0:
            combined.extend(parts)
        else:
            combined.append(score)
    return combined
