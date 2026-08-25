"""
nuScenes-lidarseg parser, mirroring the SemanticKITTI one so that the rest of
the pipeline (projection, range-view loader, training, evaluation) is unchanged.

Adapted from RangeViT (github.com/valeoai/rangevit), which merges the 32 general
nuScenes classes into the 16 + 1 classes of the lidarseg challenge.

Scans live at
    <root>/samples/LIDAR_TOP/*.pcd.bin        (N x 5 float32: x, y, z, intensity, ring)
and per-point labels at
    <root>/lidarseg/<version>/<token>_lidarseg.bin   (N uint8)

The list of keyframes per split can come from three places, tried in order:
  1. `info_path`  -- RangeViT's `nuscenes_lidar_n_label_data_info.json`, giving
     exactly the samples RangeViT trains on;
  2. a cached index written by a previous run;
  3. the `nuscenes-devkit`, which owns the official train/val scene split.
"""

import json
import os

import numpy as np


class NuScenesLidarSeg:
    NUM_CLASSES = 17  # 16 semantic classes + 1 ignore/unlabelled (index 0)

    # Merges the 32 general nuScenes classes into the 16 challenge classes.
    GENERAL_TO_SEG = {
        0: 0,  1: 0,  2: 7,  3: 7,  4: 7,  5: 0,  6: 7,  7: 0,
        8: 0,  9: 1, 10: 0, 11: 0, 12: 8, 13: 0, 14: 2, 15: 3,
        16: 3, 17: 4, 18: 5, 19: 0, 20: 0, 21: 6, 22: 9, 23: 10,
        24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 0, 30: 16, 31: 0,
    }

    CLASS_NAMES = [
        'ignore', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation',
    ]

    def __init__(self, root: str, version: str = 'v1.0-trainval', split: str = 'train',
                 has_label: bool = True, info_path: str = None,
                 index_cache_dir: str = './cache'):
        assert version in ('v1.0-trainval', 'v1.0-mini', 'v1.0-test')
        assert split in ('train', 'val', 'test')

        self.root = root
        self.version = version
        self.split = split
        self.has_label = has_label and split != 'test'

        lidar_files, label_files = self._build_index(info_path, index_cache_dir)

        self.files = []
        for i, lidar in enumerate(lidar_files):
            # 'samples/LIDAR_TOP/n015-...__LIDAR_TOP__1532402927647951.pcd.bin'
            stem = os.path.basename(lidar).replace('.pcd.bin', '')
            label = label_files[i] if i < len(label_files) else None
            token = (os.path.basename(label).replace('_lidarseg.bin', '')
                     if label else stem)
            self.files.append({
                'seq': self.split,     # nuScenes has no sequence folders
                'stem': token,         # sample_data token, used when exporting
                'scan': os.path.join(root, lidar),
                'label': os.path.join(root, label) if label else None,
            })

        # Fast remapping of the raw uint8 labels
        self.lut = np.zeros(256, dtype=np.int32)
        for raw, mapped in self.GENERAL_TO_SEG.items():
            self.lut[raw] = mapped

        # Our training labels already are the challenge classes, so exporting is
        # the identity; kept for interface parity with the SemanticKITTI parser.
        self.inv_lut = np.arange(self.NUM_CLASSES, dtype=np.uint32)

        self.mapped_cls_name = list(self.CLASS_NAMES)
        # RangeViT does not frequency-weight the nuScenes focal loss.
        self.cls_freq = None

        print(f'nuScenes: {version} - {split} #samples: {len(self.files)}')

    # ------------------------------------------------------------------ index

    def _build_index(self, info_path, index_cache_dir):
        if info_path is not None:
            return self._index_from_info_file(info_path)

        cache_path = None
        if index_cache_dir:
            cache_path = os.path.join(
                index_cache_dir, f'nuscenes_{self.version}_{self.split}.json')
            if os.path.isfile(cache_path):
                print(f'Loading the nuScenes index from {cache_path}')
                with open(cache_path) as f:
                    cached = json.load(f)
                return cached['lidar'], cached['label']

        lidar_files, label_files = self._index_from_devkit()

        if cache_path is not None:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump({'lidar': lidar_files, 'label': label_files}, f)
                print(f'Cached the nuScenes index to {cache_path}')
            except OSError as e:
                print(f'Could not cache the nuScenes index ({e}); rebuilding it next time.')

        return lidar_files, label_files

    def _index_from_info_file(self, info_path):
        """RangeViT's `nuscenes_lidar_n_label_data_info.json`."""
        with open(info_path) as f:
            info = json.load(f)
        if self.version not in info or self.split not in info[self.version]:
            raise KeyError(f'{info_path} has no entry for {self.version}/{self.split}')
        lidar_files, label_files = info[self.version][self.split]
        return lidar_files, label_files

    def _index_from_devkit(self):
        """Official keyframe list and train/val scene split, via nuscenes-devkit."""
        try:
            from nuscenes import NuScenes as NuScenesDevkit
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as e:
            raise ImportError(
                'Building the nuScenes index needs the nuScenes devkit '
                '(`pip install nuscenes-devkit`). Alternatively, point '
                'dataset.info_path at RangeViT\'s '
                '`nuscenes_lidar_n_label_data_info.json`.') from e

        print(f'Building the nuScenes index for {self.version}/{self.split} '
              f'(this reads the metadata tables and takes a while)...')
        nusc = NuScenesDevkit(version=self.version, dataroot=self.root, verbose=False)
        scene_names = set(create_splits_scenes()[self.split])

        lidar_files, label_files = [], []
        for sample in nusc.sample:
            scene = nusc.get('scene', sample['scene_token'])
            if scene['name'] not in scene_names:
                continue
            sd_token = sample['data']['LIDAR_TOP']
            lidar_files.append(nusc.get('sample_data', sd_token)['filename'])
            if self.has_label:
                label_files.append(nusc.get('lidarseg', sd_token)['filename'])

        return lidar_files, label_files

    # ------------------------------------------------------------------ items

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            points : (N, 4) float32 [x, y, z, intensity]
            labels : (N,)   int32   training class indices [0, NUM_CLASSES)
        """
        entry = self.files[idx]

        # nuScenes scans are N x 5: x, y, z, intensity, ring index
        points = np.fromfile(entry['scan'], dtype=np.float32).reshape(-1, 5)[:, :4]

        labels = np.zeros(len(points), dtype=np.int32)
        if self.has_label and entry['label'] is not None and os.path.exists(entry['label']):
            raw = np.fromfile(entry['label'], dtype=np.uint8)
            labels = self.lut[raw]

        return points, labels
