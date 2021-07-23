from staff_finder_py2.image_deformer import ImageDeformer
from pathlib import Path

if __name__ == '__main__':
    testset_dir = Path(__file__).absolute().parent.parent.parent / 'OMR-measure-segmenter-data/testset'
    infile = testset_dir / 'historic/rossi.png'
    nostaves = testset_dir / 'historic/rossi-nostaff.png'
    deformer = ImageDeformer(infile, nostaves, save_dir=testset_dir)
    deformer.deform_image(False)
