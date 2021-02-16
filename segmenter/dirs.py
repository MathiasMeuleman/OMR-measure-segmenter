from os.path import join
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent.absolute()
data_dir = join(root_dir, 'data')
eval_dir = join(root_dir, 'evaluation')
tmp_dir = join(root_dir, 'tmp')
