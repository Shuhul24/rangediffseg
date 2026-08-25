from .nuscenes import NuScenesLidarSeg
from .range_view_loader import RangeViewDataset
from .semantic_kitti import SemanticKITTI


def build_parser(name, root, split, settings):
    """
    Build the point-cloud parser for a dataset name and a split.

    Args:
        name     : 'SemanticKitti' or 'nuScenes'
        root     : dataset root directory
        split    : 'train', 'val' or 'test'
        settings : the Option object (for the sequence lists / nuScenes version)
    """
    if name == 'SemanticKitti':
        sequences = {'train': settings.train_seqs,
                     'val': settings.val_seqs,
                     'test': settings.test_seqs}[split]
        return SemanticKITTI(root, sequences, has_label=(split != 'test'))

    if name == 'nuScenes':
        version = settings.version
        if split == 'test':
            version = 'v1.0-test'
        return NuScenesLidarSeg(
            root, version=version, split=split,
            has_label=(split != 'test'),
            info_path=settings.info_path,
            index_cache_dir=settings.index_cache_dir)

    raise ValueError(f'invalid dataset: {name}')
