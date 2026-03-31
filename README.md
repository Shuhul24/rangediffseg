# RangeDiffSeg

**3D LiDAR Semantic Segmentation via Denoising Diffusion Probabilistic Models on Range-View Images**

RangeDiffSeg combines two ideas:
- **Range-view projection** from [RangeViT](https://github.com/valeoai/rangevit): 3D LiDAR point clouds are projected onto a compact 2D range image (H × W), segmented, then mapped back to per-point 3D labels.
- **DDPM segmentation** from [SegDiff](https://github.com/tomeramit/SegDiff): a timestep-conditioned U-Net iteratively denoises a noisy segmentation map, conditioned on the range image, to produce the final per-pixel class predictions.

```
3D Point Cloud
      │
      ▼  Spherical projection
 Range Image (64 × 512, 5 ch)          ←─ conditioning
      │                                         │
      ▼                                         │
 Noisy Seg Map ──► U-Net (DDPM) ──► Pred Seg Map
                                         │
                                         ▼  Back-projection
                                   Per-point 3D Labels
```

---

## Repository Layout

```
rangediffseg/
├── dataset/
│   ├── projection.py        # Spherical 3D → 2D projection & back-projection indices
│   ├── augmentor.py         # Random 3D augmentations (flip, translate, rotate, scale)
│   ├── semantic_kitti.py    # SemanticKITTI binary file parser & label remapping
│   └── range_view_loader.py # PyTorch Dataset: projects scans, returns (seg, img, mask)
├── diffusion/
│   ├── nn.py                # Sinusoidal timestep embedding, GroupNorm, zero_module
│   ├── gaussian_diffusion.py# DDPM: noise schedules, q_sample, p_sample_loop, MSE loss
│   └── unet.py              # Timestep-conditioned U-Net (ResBlocks + bottleneck attention)
├── train.py                 # Training entry point
├── inference.py             # Evaluation entry point (DDPM sampling + mIoU)
├── config.yaml              # All hyperparameters — primary file to edit
└── requirements.txt
```

---

## 1. Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers: `torch`, `torchvision`, `numpy`, `scipy`, `pyyaml`, `tqdm`.

> Tested with Python 3.9, PyTorch 1.12+, CUDA 11.6.

---

## 2. Data Preparation

### 2.1 Download SemanticKITTI

1. Download the **velodyne point clouds** and **labels** from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html#download).
2. Unzip everything into a single root directory.

### 2.2 Expected directory structure

The parser reads binary `.bin` scan files and `.label` annotation files.
Your dataset directory **must** follow this exact layout:

```
/your/dataset/root/
└── sequences/
    ├── 00/
    │   ├── velodyne/          # raw LiDAR scans
    │   │   ├── 000000.bin
    │   │   ├── 000001.bin
    │   │   └── ...
    │   └── labels/            # per-point semantic labels
    │       ├── 000000.label
    │       ├── 000001.label
    │       └── ...
    ├── 01/
    │   ├── velodyne/
    │   └── labels/
    │   ...
    └── 21/
        └── velodyne/          # test sequences have no labels
```

> **Note:** Test sequences (11–21) contain only `velodyne/` folders (no labels).
> The standard split is: train = 00–07, 09–10 · val = 08 · test = 11–21.

### 2.3 Label encoding

SemanticKITTI raw labels are automatically remapped to 20 training classes
(19 semantic + 1 ignore/unlabelled at index 0) inside `dataset/semantic_kitti.py`.
No manual pre-processing is required.

---

## 3. Configuration — What to Change

All user-facing settings live in **`config.yaml`**.
The only **required** change before running anything is the dataset path.

### 3.1 Required change

```yaml
dataset:
  root: /your/dataset/root        # ← set this to your SemanticKITTI root
```

### 3.2 Optional changes

| Section | Key | Default | When to change |
|---|---|---|---|
| `dataset` | `proj_h` / `proj_w` | `64` / `512` | Match your LiDAR (e.g. 32-beam → `proj_h: 32`) |
| `dataset` | `train_seqs` / `val_seqs` | standard split | Custom splits or quick experiments |
| `diffusion` | `num_timesteps` | `100` | More steps (e.g. 1000) for higher quality; fewer for speed |
| `diffusion` | `schedule` | `cosine` | `linear` for the original DDPM schedule |
| `model` | `base_ch` | `64` | Reduce (e.g. `32`) to cut GPU memory; increase for capacity |
| `model` | `ch_mults` | `[1,2,4,4]` | Shallower (`[1,2,4]`) or deeper (`[1,2,4,8]`) U-Net |
| `model` | `use_mid_attn` | `true` | Set `false` to save memory during prototyping |
| `training` | `batch_size` | `4` | Adjust to fit your GPU VRAM |
| `training` | `lr` | `2e-4` | Standard AdamW range: `1e-4` – `5e-4` |
| `training` | `max_steps` | `300000` | Reduce for quick experiments |
| `training` | `log_dir` | `./checkpoints` | Where checkpoints are saved |

### 3.3 Augmentation

All 3D augmentation parameters (flip probability, translation range, rotation
range, scale range) are under `dataset.augmentation` in `config.yaml`.
Set any probability to `0.0` to disable that augmentation entirely.

---

## 4. Training

```bash
python train.py --config config.yaml
```

**Resume from a checkpoint:**

```bash
python train.py --config config.yaml --resume checkpoints/ckpt_0050000.pt
```

### What happens during training

1. Each scan is loaded, randomly augmented in 3D, then projected to a
   64 × 512 range image with 5 channels `[range, x, y, z, intensity]`.
2. The ground-truth label map is one-hot encoded and scaled to `[-1, 1]`.
3. A random timestep `t ~ Uniform(0, T)` is sampled per scan.
4. The clean segmentation is noised to `x_t` via the forward diffusion process.
5. The U-Net predicts the clean segmentation `x_0` from `(x_t, t, range_image)`.
6. MSE loss is computed only over valid (non-empty) LiDAR pixels.
7. Checkpoints are saved to `log_dir` every `save_every` steps.

### Monitoring

Training loss is printed every `log_every` steps:

```
step   5000 / 300000  loss=0.0842
step  10000 / 300000  loss=0.0731
...
```

---

## 5. Evaluation / Testing

```bash
python inference.py --config config.yaml --checkpoint checkpoints/ckpt_0300000.pt
```

**Evaluate on a different split** (e.g. test):

```bash
python inference.py --config config.yaml \
                    --checkpoint checkpoints/ckpt_0300000.pt \
                    --split test
```

### What happens during evaluation

1. For each scan, the full DDPM reverse process runs: starting from Gaussian
   noise, the U-Net iteratively denoises for `T` steps, conditioned on the
   range image.
2. The final `(num_classes, H, W)` output is argmax-ed to get 2D pixel labels.
3. Each 3D point is mapped back to its projected pixel using cached
   `(uproj_x, uproj_y)` indices — giving a per-point 3D label.
4. mIoU and per-class IoU are computed over all scans (class 0 ignored).

### Expected output

```
Evaluating on 4071 scans...

mIoU (classes 1-19): XX.XX%
  class  1: XX.XX%   # car
  class  2: XX.XX%   # bicycle
  ...
  class 19: XX.XX%   # vegetation
```

> **Note:** The benchmark metric for SemanticKITTI is **mIoU over 19 classes**
> (class 0 = unlabelled is always excluded). This matches the official leaderboard.

---

## 6. Key Design Choices

| Choice | Detail |
|---|---|
| **x₀-prediction** | U-Net directly predicts the clean segmentation (not the noise ε). Cleaner for discrete argmax decoding. |
| **Conditioning by concatenation** | Range image is concatenated channel-wise with the noisy segmentation as U-Net input — simple and effective. |
| **Masked MSE loss** | Loss is averaged only over valid LiDAR pixels; empty range-image cells (no projected point) are ignored. |
| **Cosine noise schedule** | Preferred over linear for fewer diffusion steps (T=100). Switch to `linear` for T≥1000. |
| **Back-projection** | Each point stores its projected pixel (u, v) at dataset load time; inference uses a single NumPy gather — no KNN needed. |

---

## 7. Citations

If you use this work, please also cite the two projects it builds on:

```bibtex
@article{segdiff2021,
  title   = {SegDiff: Image Segmentation with Diffusion Probabilistic Models},
  author  = {Amit, Tomer and Nachmani, Eliya and Toker, Tal and Wolf, Lior},
  journal = {arXiv:2112.00390},
  year    = {2021}
}

@inproceedings{rangevit2023,
  title     = {RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving},
  author    = {Ando, Angelika and Gidaris, Spyros and Bursuc, Andrei and Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  booktitle = {CVPR},
  year      = {2023}
}
```
