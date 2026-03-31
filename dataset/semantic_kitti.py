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

    # Maps raw SemanticKITTI labels → 0-based training labels
    LABEL_MAP = {
        0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
        30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
        51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
        99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
    }

    def __init__(self, root: str, sequences: list):
        """
        Args:
            root      : path to the SemanticKITTI dataset root
            sequences : list of sequence IDs, e.g. ['00', '01', '08']
        """
        self.files = []
        for seq in sequences:
            velo_dir  = os.path.join(root, 'sequences', seq, 'velodyne')
            label_dir = os.path.join(root, 'sequences', seq, 'labels')
            for fname in sorted(os.listdir(velo_dir)):
                stem = os.path.splitext(fname)[0]
                self.files.append({
                    'scan':  os.path.join(velo_dir, fname),
                    'label': os.path.join(label_dir, stem + '.label'),
                })

        # Build a 65536-entry LUT for fast label remapping
        self.lut = np.zeros(65536, dtype=np.int32)
        for raw, mapped in self.LABEL_MAP.items():
            self.lut[raw] = mapped

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Returns:
            points : (N, 4) float32 [x, y, z, intensity]
            labels : (N,)   int32   training class indices [0, NUM_CLASSES)
        """
        entry = self.files[idx]

        # Load binary scan: N points × 4 floats (x, y, z, intensity)
        points = np.fromfile(entry['scan'], dtype=np.float32).reshape(-1, 4)

        labels = np.zeros(len(points), dtype=np.int32)
        if os.path.exists(entry['label']):
            raw     = np.fromfile(entry['label'], dtype=np.int32)
            sem_raw = raw & 0xFFFF          # lower 16 bits = semantic class
            labels  = self.lut[sem_raw]

        return points, labels
