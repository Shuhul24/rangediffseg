"""
SemanticKITTI dataset parser: reads binary scan (.bin) and label (.label) files.
Adapted from RangeViT (github.com/valeoai/rangevit).

Dataset structure expected:
    <root>/sequences/<seq>/velodyne/*.bin
    <root>/sequences/<seq>/labels/*.label
"""

import os

import numpy as np


class SemanticKITTI:
    NUM_CLASSES = 20  # 19 semantic classes + 1 ignore/unlabelled (index 0)

    # Maps raw SemanticKITTI labels -> 0-based training labels
    LABEL_MAP = {
        0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
        30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
        51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
        99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
    }

    # Maps training labels back to raw SemanticKITTI labels (learning_map_inv),
    # used when writing predictions for the benchmark test server.
    INV_LABEL_MAP = {
        0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40,
        10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72,
        18: 80, 19: 81,
    }

    # Name of each training class (index 0 is ignored during evaluation)
    CLASS_NAMES = [
        'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
        'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
        'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
        'pole', 'traffic-sign',
    ]

    # Point frequency of every raw label over the training split, from the
    # `content` field of the official semantic-kitti.yaml. Used to weight the
    # focal loss, exactly as RangeViT does.
    CONTENT = {
        0: 0.018889854628292943,   1: 0.0002937197336781505,
        10: 0.040818519255974316,  11: 0.00016609538710764618,
        13: 2.7879693665067774e-05, 15: 0.00039838616015114444,
        16: 0.0,                   18: 0.0020633612104619787,
        20: 0.0016218197275284021, 30: 0.00017698551338515307,
        31: 1.1065903904919655e-08, 32: 5.532951952459828e-09,
        40: 0.1987493871255525,    44: 0.014717169549888214,
        48: 0.14392298360372,      49: 0.0039048553037472045,
        50: 0.1326861944777486,    51: 0.0723592229456223,
        52: 0.002395131480328884,  60: 4.7084144280367186e-05,
        70: 0.26681502148037506,   71: 0.006035012012626033,
        72: 0.07814222006271769,   80: 0.002855498193863172,
        81: 0.0006155958086189918, 99: 0.009923127583046915,
        252: 0.001789309418528068, 253: 0.00012709999297008662,
        254: 0.00016059776092534436, 255: 3.745553104802113e-05,
        256: 0.0,                  257: 0.00011351574470342855,
        258: 0.00010157861367183268, 259: 4.3840131989471124e-05,
    }

    def __init__(self, root: str, sequences: list, has_label: bool = True):
        """
        Args:
            root      : path to the SemanticKITTI dataset root
            sequences : list of sequence IDs, e.g. ['00', '01', '08']
            has_label : False for the test sequences (11-21), which ship no labels
        """
        self.has_label = has_label
        self.files = []
        for seq in sequences:
            seq = f'{int(seq):02d}' if not isinstance(seq, str) else seq
            velo_dir = os.path.join(root, 'sequences', seq, 'velodyne')
            label_dir = os.path.join(root, 'sequences', seq, 'labels')
            if not os.path.isdir(velo_dir):
                raise FileNotFoundError(f'Sequence directory not found: {velo_dir}')
            for fname in sorted(os.listdir(velo_dir)):
                if not fname.endswith('.bin'):
                    continue
                stem = os.path.splitext(fname)[0]
                self.files.append({
                    'seq': seq,
                    'stem': stem,
                    'scan': os.path.join(velo_dir, fname),
                    'label': os.path.join(label_dir, stem + '.label'),
                })

        # Build a 65536-entry LUT for fast label remapping
        self.lut = np.zeros(65536, dtype=np.int32)
        for raw, mapped in self.LABEL_MAP.items():
            self.lut[raw] = mapped

        # Inverse LUT for exporting predictions in the original label space
        self.inv_lut = np.zeros(self.NUM_CLASSES, dtype=np.uint32)
        for mapped, raw in self.INV_LABEL_MAP.items():
            self.inv_lut[mapped] = raw

        self.mapped_cls_name = list(self.CLASS_NAMES)
        self.cls_freq = self._class_frequencies()

    def _class_frequencies(self):
        """Aggregate the raw-label frequencies into the 20 training classes."""
        freq = np.zeros(self.NUM_CLASSES, dtype=np.float64)
        for raw, content in self.CONTENT.items():
            freq[self.LABEL_MAP[raw]] += content
        return freq

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            points : (N, 4) float32 [x, y, z, intensity]
            labels : (N,)   int32   training class indices [0, NUM_CLASSES)
        """
        entry = self.files[idx]

        # Load binary scan: N points x 4 floats (x, y, z, intensity)
        points = np.fromfile(entry['scan'], dtype=np.float32).reshape(-1, 4)

        labels = np.zeros(len(points), dtype=np.int32)
        if self.has_label and os.path.exists(entry['label']):
            raw = np.fromfile(entry['label'], dtype=np.int32)
            sem_raw = raw & 0xFFFF          # lower 16 bits = semantic class
            labels = self.lut[sem_raw]

        return points, labels
