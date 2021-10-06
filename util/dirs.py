from pathlib import Path

root_dir = Path(__file__).parents[1].absolute()
data_dir = root_dir / 'data'
eval_dir = root_dir / 'evaluation'
tmp_dir = root_dir / 'tmp'

segmenter_data_dir = Path(__file__).parents[2].absolute() / 'OMR-measure-segmenter-data'
musicdata_dir = segmenter_data_dir / 'musicdata'
stafffinder_testset_dir = segmenter_data_dir / 'stafffinder-testset'
