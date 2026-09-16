# RangeDiffSeg / RangeDiT

**LiDAR semantic segmentation on range-view images with an off-the-shelf Diffusion Transformer (DiT)**

Supports **SemanticKITTI** (`config.yaml`) and **nuScenes-lidarseg** (`config_nusc.yaml`).

This repository applies the [RangeViT](https://github.com/valeoai/rangevit) recipe to the
[Diffusion Transformer](https://github.com/facebookresearch/DiT) instead of a plain ViT.

RangeViT takes an off-the-shelf ImageNet ViT, keeps the transformer trunk, and learns only what is
specific to LiDAR range images — the *stem* (patchification, patch encoding, positional embedding)
and the *decoder*. RangeDiT does the same with the pre-trained **DiT-XL/2** weights released by
Meta: the 28 transformer blocks, their attention, MLP and adaLN-Zero modulation come straight from
the diffusion checkpoint and are frozen by default; the stem, the positional embedding, the
conditioning and the decoder are trained.

```
3D Point Cloud
      │
      ▼  Spherical projectionls

 Range Image (64 × 2048, 5 ch)
      │
      ▼  Random 64 × 384 crop
 ┌─────────────────────────────────────────────────────────┐
 │ STEM      SalsaNext conv blocks → 2×8 patches → tokens  │  ← trained
 │           + positional embedding (DiT's sin-cos, resized)│
 ├─────────────────────────────────────────────────────────┤
 │ DiT-XL/2  28 × DiTBlock (adaLN-Zero)                    │  ← off-the-shelf
 │           conditioned on a fixed timestep + null class   │     (frozen)
 ├─────────────────────────────────────────────────────────┤
 │ DECODER   up-conv (2×8 pixel shuffle) + stem skip → 20  │  ← trained
 └─────────────────────────────────────────────────────────┘
      │
      ▼  Back-projection
 Per-point 3D Labels
```

---

## What changes when the backbone is a DiT rather than a ViT

| | ViT (RangeViT) | DiT (this repository) |
|---|---|---|
| Patch embedding | 16×16 RGB patches | 2×2 VAE-latent patches — **not reusable**, replaced by the conv stem |
| Positional embedding | learnt, interpolated | fixed 2-D sin-cos on a 16×16 grid, **resized** to the 32×48 range-view grid (and made learnable) |
| Class token | yes, dropped before decoding | none — every token is a patch token |
| Normalisation | LayerNorm with affine parameters | LayerNorm with `elementwise_affine=False`; scale/shift come from adaLN-Zero |
| Block conditioning | none | `c = t_embedder(t) + y_embedder(y)` modulates every block |
| Output head | `decoder` | `final_layer.linear` predicts noise/latent patches — dropped, replaced by the decoder |

The conditioning is the only part with no counterpart in RangeViT. Segmentation has neither a
diffusion timestep nor an ImageNet class, so RangeDiT feeds a **fixed timestep** (`cond_timestep`,
default 0) together with the **class-free ("null") label embedding** — both taken from the
pre-trained embedders — and learns a **zero-initialised offset** on top. Training therefore starts
from exactly the pre-trained modulation and adapts from there. Setting `cond_mode: context` also
conditions each scan on its own pooled stem features, through a zero-initialised projection.

Because the DiT LayerNorms have no affine parameters, the counterpart of RangeViT's
`unfreeze_layernorm` is **`unfreeze_adaln`**: it trains the per-block adaLN-Zero modulation, i.e.
the scales, shifts and residual gates. It is on by default.

---

## Repository Layout

```
rangediffseg/
├── dataset/
│   ├── projection.py        # Spherical 3D → 2D projection & back-projection indices
│   ├── augmentor.py         # Random 3D augmentations (flip, translate, rotate, scale,
│   │                        #   point dropout, intensity jitter)
│   ├── instance_bank.py     # Rare-class instance bank + copy-paste augmentation
│   ├── mix.py               # LaserMix / PolarMix scan mixing
│   ├── semantic_kitti.py    # SemanticKITTI parser, label remapping, class statistics
│   ├── nuscenes.py          # nuScenes-lidarseg parser, 32 → 17 class merging
│   └── range_view_loader.py # PyTorch Dataset: projection, crops, (feature, label, mask)
├── models/
│   ├── dit.py               # DiT backbone, state-dict compatible with facebookresearch/DiT
│   ├── stems.py             # ConvStem (patchification + patch encoding) / PatchEmbedding
│   ├── decoders.py          # up-conv and linear segmentation decoders
│   ├── model_utils.py       # positional-embedding resizing, padding, grid sizes
│   └── rangedit.py          # RangeDiT: stem + DiT trunk + decoder, pre-trained weight loading
├── utils/
│   ├── losses.py            # Lovász-softmax + focal softmax loss
│   ├── metrics.py           # IOUEval (confusion matrix → IoU / Acc / Recall)
│   ├── scheduler.py         # Warmup + cosine LR schedule
│   ├── inference.py         # Sliding-window inference over full-width range images
│   ├── knn.py               # RangeNet++ KNN post-processing of the 3-D labels
│   ├── ema.py               # Exponential moving average of the trainable weights
│   └── tools.py             # Meters, remaining-time estimate, logger, result tables
├── diffusion/               # Legacy DDPM U-Net baseline (see "Legacy" below)
├── tools/
│   └── run_ablations.py     # run a set of training variants and tabulate them
├── option.py                # config.yaml → settings object, CLI overrides
├── train.py                 # Training entry point
├── inference.py             # Evaluation entry point (2-D pixel and 3-D point mIoU)
├── config.yaml              # SemanticKITTI hyperparameters — primary file to edit
├── config_nusc.yaml         # nuScenes hyperparameters
└── requirements.txt
```

---

## 1. Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers `torch`, `numpy`, `scipy` and `pyyaml`. `tensorboard` is optional — if
it is installed, scalars are written next to the console log. `wandb` is also optional and is only
used when Weights & Biases logging is enabled. `timm`, `einops` and `prettytable` (used by the
upstream repositories) are **not** required: the DiT blocks, the anisotropic pixel shuffle and the
result tables are implemented here directly.

> Tested with Python 3.9+, PyTorch 1.12+.

---

## 2. Data Preparation

### 2.1 Download SemanticKITTI

1. Download the **velodyne point clouds** and **labels** from the [SemanticKITTI website](http://semantic-kitti.org/dataset.html#download).
2. Unzip everything into a single root directory.

### 2.2 Expected directory structure

```
/your/dataset/root/
└── sequences/
    ├── 00/
    │   ├── velodyne/          # raw LiDAR scans
    │   │   ├── 000000.bin
    │   │   └── ...
    │   └── labels/            # per-point semantic labels
    │       ├── 000000.label
    │       └── ...
    ...
    └── 21/
        └── velodyne/          # test sequences have no labels
```

> The standard split is: train = 00–07, 09–10 · val = 08 · test = 11–21.

### 2.3 Label encoding

Raw SemanticKITTI labels are remapped to 20 training classes (19 semantic + 1 ignore at index 0)
inside `dataset/semantic_kitti.py`. Class 0 is excluded from every metric, which matches the
official benchmark. No manual pre-processing is required.

### 2.4 nuScenes-lidarseg

Download **nuScenes** (full dataset) and the **nuScenes-lidarseg** annotations, and unzip them
into one root:

```
/your/nuscenes/root/
├── samples/LIDAR_TOP/*.pcd.bin          # keyframe scans (N × 5 float32)
├── lidarseg/v1.0-trainval/*_lidarseg.bin # per-point labels (N × uint8)
└── v1.0-trainval/*.json                  # metadata tables
```

The 32 general nuScenes classes are merged into the 16 + 1 lidarseg-challenge classes, using
RangeViT's mapping (`dataset/nuscenes.py`).

The list of keyframes per split comes from one of two places, set in `config_nusc.yaml`:

* **`info_path: null`** (default) — built with the **nuScenes devkit**
  (`pip install nuscenes-devkit`), which owns the official train/val scene split. The first run
  reads the metadata tables (slow, a few minutes) and caches the result in `index_cache_dir`
  (`./cache` by default); later runs load the cache.
* **`info_path: /path/to/nuscenes_lidar_n_label_data_info.json`** — RangeViT's pre-generated
  index ([download it from their repo](https://github.com/valeoai/rangevit/blob/main/dataset/nuScenes/nuscenes_lidar_n_label_data_info.json),
  7 MB). No devkit needed, and it reproduces exactly the 28 130 train / 6 019 val samples
  RangeViT uses.

---

## 3. Pre-trained DiT weights

Meta released the **DiT-XL/2** checkpoints only (`DiT-XL-2-256x256.pt`, `DiT-XL-2-512x512.pt`).
Set `model.pretrained_model` in `config.yaml` to one of those names and it is downloaded to
`pretrained_models/` on first use, exactly as `download.py` of the DiT repository does. A local
`.pt` path also works, as does `null` to train the trunk from scratch.

The checkpoint is loaded into the trunk after three deliberate substitutions:

* `x_embedder.*` (2×2 latent patch embedding) → dropped, the conv stem replaces it;
* `pos_embed` → bilinearly resized from the 16×16 DiT grid to the range-view token grid;
* `final_layer.linear.*` (noise/latent head) → dropped, the decoder replaces it.

Everything else — 28 blocks, the timestep and label embedders, the final modulated LayerNorm —
loads unchanged. The loader prints the resulting missing/unexpected keys so the substitution is
visible in the log.

> The DiT code and weights are released by Meta under **CC BY-NC 4.0** (non-commercial).
> RangeViT is Apache-2.0 and SalsaNext is MIT. Check those licences before using this for
> anything beyond research.

---

## 4. Configuration — What to Change

All settings live in **`config.yaml`**. The only **required** change is the dataset path.

```yaml
dataset:
  root: /your/dataset/root        # ← set this to your SemanticKITTI root
```

The two datasets differ only in the `dataset` section and the crop size — the model, the losses,
the schedule and the metrics are shared:

| | SemanticKITTI (`config.yaml`) | nuScenes (`config_nusc.yaml`) |
|---|---|---|
| `n_classes` | 20 (19 + ignore) | 17 (16 + ignore) |
| `proj_h` / `proj_w` | 64 × 2048 | 32 × 2048 |
| `fov_up` / `fov_down` | +3° / −25° | +10° / −30° |
| `image_size` (train crop) | `[64, 384]` | `[32, 384]` |
| `window_size` / `window_stride` | `[64, 384]` / `[64, 256]` | `[32, 384]` / `[32, 256]` |
| Focal-loss weights | point-frequency weighted | uniform (`use_cls_freq_weights: false`) |
| `lr` / `batch_size` / `n_epochs` | 2e-4 / 4 / 50 | 8e-4 / 8 / 150 |
| Prediction export | `.label` uint32, raw KITTI labels | `_lidarseg.bin` uint8, 17-class labels |

| Section | Key | Default | When to change |
|---|---|---|---|
| `dataset` | `proj_h` / `proj_w` | `64` / `2048` | Match your LiDAR (e.g. 32-beam → `proj_h: 32`) |
| `dataset` | `img_mean` / `img_stds` | KITTI statistics | Per-channel normalisation (RangeViT reuses the KITTI values for nuScenes) |
| `dataset` | `version` / `info_path` | `v1.0-trainval` / `null` | nuScenes only — see §2.4 |
| `model` | `backbone` | `DiT-XL/2` | Only DiT-XL/2 has released weights; `DiT-B/2`, `DiT-S/2`… for from-scratch runs |
| `model` | `pretrained_model` | `DiT-XL-2-256x256.pt` | `DiT-XL-2-512x512.pt`, a local path, or `null` for scratch |
| `model` | `image_size` | `[64, 384]` | Size of the random crop the network is trained on |
| `model` | `patch_size` / `patch_stride` | `[2, 8]` | Larger patches → fewer tokens → less memory |
| `model` | `D_h` | `256` | Hidden width of the stem and of the decoder |
| `model` | `skip_filters` | `256` | `0` disables the stem → decoder skip connection (else must equal `D_h`) |
| `model` | `conv_stem` | `ConvStem` | `none` uses a plain patch embedding (then `reuse_patch_emb` becomes possible) |
| `model` | `stem_pool` | `avg` | `dwconv` learns the stem→token reduction instead of averaging — see §7b |
| `model` | `decoder_stages` | `1` | `2` upsamples in two steps with an extra skip — see §7b |
| `model` | `cond_timestep` | `0` | Which diffusion timestep the frozen `t_embedder` is evaluated at |
| `model` | `cond_mode` | `static` | `context` also conditions on the pooled stem features |
| `model` | `freeze_dit_encoder` | `true` | `false` fine-tunes the whole trunk (much more memory) |
| `model` | `unfreeze_adaln` | `true` | Trains the adaLN-Zero modulation (the DiT analogue of unfreezing LayerNorms) |
| `model` | `adaln_bias_only` | `true` | Trains only the modulation bias — same expressive power, 1/`d_model` of the parameters |
| `model` | `unfreeze_attn` / `unfreeze_ffn` | `false` | Progressively unfreeze attention / MLP layers |
| `model` | `feature_taps` | `[]` | Extra DiT blocks whose output is fused in — see §7b |
| `model` | `n_trunk_blocks` | `null` | Run only the first N blocks; `0` bypasses the trunk — see §7b |
| `model` | `grad_checkpointing` | `false` | Recompute block activations instead of storing them |
| `model` | `head_prior_bias` | `false` | Initialise the classifier bias to the log class prior |
| `model` | `window_size` / `window_stride` | `[64, 384]` / `[64, 256]` | Sliding window used at validation and test time |
| `training` | `batch_size` | `4` | Adjust to fit your GPU VRAM |
| `training` | `lr` | `2e-4` | AdamW; lower it (e.g. `1e-5`) when the whole trunk is unfrozen |
| `training` | `n_epochs` | `50` | |
| `training` | `save_path` / `id` | `./log` / `rangedit_semantickitti` | Logs go to `<save_path>/log_<id>/` |
| `dataset` | `rare_classes` | tail class ids | Which classes count as the "tail" — see §4b |
| `dataset` | `rare_crop_p` | `0.5` | Probability of centring the training crop on a tail-class pixel |
| `dataset` | `copy_paste.*` | disabled | Rare-class instance copy-paste — see §4b |
| `dataset` | `mix.*` | disabled | LaserMix / PolarMix scan mixing — see §4b |
| `dataset` | `cls_weight_scheme` | `log_inv` | `log_inv`, `salsanext`, `inv_sqrt` or `uniform` |
| `training` | `val_3d` / `val_knn` | `false` | Score point-wise (3-D) mIoU during validation — the benchmark metric |
| `training` | `ema_decay` | `0.999` | EMA of the trainable weights; `0` disables |
| `training` | `grad_clip` | `1.0` | Gradient-norm clipping; `0` disables |
| `training` | `lovasz_weight` / `focal_weight` / `focal_gamma` | `1.0` / `1.0` / `2.0` | Loss term weights |

DiT-XL/2 is a 677 M-parameter trunk, so what you unfreeze decides whether the model fits on your
GPU. Measured trainable parameters for the default 64 × 384 crop:

| Setting | Trainable |
|---|---|
| `freeze_dit_encoder: true`, `adaln_bias_only: true` *(default)* | **9.4 M** — stem 1.5 M, decoder 6.0 M, pos. embedding 1.8 M, adaLN biases 0.2 M |
| `freeze_dit_encoder: true`, `adaln_bias_only: false` | 235.0 M |
| `freeze_dit_encoder: false` | 684.0 M |

The default is the direct analogue of RangeViT's "off-the-shelf encoder, learn the stem and the
decoder": 1.4 % of the network is trained. `adaln_bias_only` is worth understanding — the
modulation is `Linear(c)`, and with `cond_mode: static` the conditioning vector `c` is the same for
every sample, so each block's modulation output is a single constant vector that the bias alone can
already produce. Training the full matrix costs 223 M parameters and buys no extra expressive
power. Set it to `false` only if you want the modulation to *re-learn* how a varying `c` maps to
scales and shifts, which only matters with `cond_mode: context`.

### Augmentation

All 3-D augmentation parameters live under `dataset.augmentation`. Set any probability to `0.0`
to disable that augmentation.

Beyond RangeViT's geometric augmentations, `augmentor.py` also offers **point dropout**
(`p_drop_points`, `drop_points_range`) and **intensity jitter** (`p_intensity_jitter`,
`intensity_scale_range`, `intensity_noise_sigma`, `p_drop_intensity`). Point dropout simulates a
sparser sensor and is one of the few augmentations that helps thin structures — pole, traffic-sign,
trunk — which are otherwise learnt only at the exact point density of the training LiDAR. Intensity
is the channel that transfers worst across sequences and across sensors, so jittering it pushes the
decision onto geometry. Both are on by default now; set the probabilities to `0.0` to get RangeViT's
exact recipe back.

---

## 4b. Training for *all* the classes

The head classes are easy and the tail classes are where the mIoU is actually won or lost. The
reason is a **sampling** problem, not a loss problem. With the official SemanticKITTI point
frequencies, one 64 × 384 training crop of a 64 × 2048 image contains, on average:

| | vegetation | road | car | pole | bicycle | bicyclist | motorcyclist |
|---|---|---|---|---|---|---|---|
| pixels per crop | ~2160 | ~1610 | ~350 | ~23 | ~1.4 | ~1.0 | ~0.3 |

RangeViT's focal weighting turns a 5000:1 frequency gap into a 3.8:1 weight gap
(`alpha = log(1 + 1/(freq + 1e-3))`, normalised). No re-weighting scheme closes a gap that large
— a class that appears in a third of a pixel per sample simply does not produce enough gradient.
So the knobs below work on the data side first, and on the loss second.

### The knobs, in the order worth trying them

| # | Knob | Where | Cost | What it does |
|---|---|---|---|---|
| 1 | `dataset.rare_crop_p` | config | free | With this probability the training crop is centred on a tail-class pixel instead of being uniform. Nothing changes at inference. |
| 2 | `dataset.copy_paste` | config + one-off bank build | ~1 h once, ~15 % slower epochs | Pastes real rare-class instances into every scan. The single largest lever on the tail classes. |
| 3 | `training.ema_decay`, `training.grad_clip` | config | free | Stability. Tail-class IoU swings hard between epochs; the EMA averages that out and the best-checkpoint selection stops picking lucky epochs. |
| 4 | `--knn` at evaluation | CLI | free, no retraining | Fixes the labels of the points the projection discards. 3-D metric only. |
| 5 | `dataset.mix` (LaserMix / PolarMix) | config | +1 scan read per mixed sample | Breaks the "this object only ever appears in this context" correlation. Works on nuScenes too. |
| 6 | `dataset.cls_weight_scheme`, `training.lovasz_weight` | config | free | Loss-side weighting. Worth a sweep *after* 1 and 2, not before. |
| 7 | `model.cond_timestep`, `model.unfreeze_attn/ffn` | config | more memory / compute | Backbone capacity — see below. |

### 1. Tail-aware cropping

```yaml
dataset:
  rare_classes: [2, 3, 4, 5, 6, 7, 8, 16, 18, 19]  # incl. trunk, pole, traffic-sign
  rare_crop_p: 0.5
```

Half the crops are now centred (with jitter) on a randomly chosen tail-class pixel. This costs one
`np.isin` per sample and nothing at inference. Start here — it is the cheapest change in the list.

### 2. Rare-class copy-paste

SemanticKITTI `.label` files carry a **panoptic instance id in the upper 16 bits**, so instances can
be cut out exactly, with no clustering. Build the bank once:

```bash
python -m dataset.instance_bank \
    --data_root /your/semantickitti/root \
    --out ./cache/instance_bank_kitti.npz \
    --stride 2        # consecutive frames are near-duplicates; 2-3 costs little and halves the time
```

then switch it on:

```yaml
dataset:
  copy_paste:
    enabled:     true
    bank_path:   ./cache/instance_bank_kitti.npz
    classes:     [2, 3, 4, 5, 6, 7, 8]
    p:           0.8
    n_instances: [1, 4]
    occlusion_tolerance: 0.5
```

Each pasted instance is rotated around the sensor's z axis by a random azimuth, which preserves its
range and elevation — and therefore its point density in the range image. A placement is rejected
when more than `occlusion_tolerance` of the instance's pixels are already occupied by *closer* scene
points, i.e. when the object would be buried inside a wall; those placements produce labels the
network can never satisfy. Occlusion in the other direction is handled for free: `projection.py`
already sorts by decreasing depth, so whichever surface is closer wins the pixel.

> nuScenes-lidarseg ships no instance ids, so the bank is SemanticKITTI-only. Use `dataset.mix`
> there instead — the config validator will tell you.

### 3. EMA and gradient clipping

```yaml
training:
  ema_decay:  0.999     # 0 disables
  grad_clip:  1.0       # 0 disables
```

Validation and the `best_*` checkpoints use the **averaged** weights; `checkpoint.pth` keeps the live
weights so resuming is exact. The EMA shadows only the *trainable* parameters — 9.4 M of 684 M under
the default freezing — so it costs ~40 MB, not a second copy of the trunk.

Training also now computes both loss terms in **fp32** even under AMP. The focal term takes `log()`
of a probability clamped at `1e-6`, which is below the fp16 normal range, and Lovász sorts per-pixel
errors, so ties decided by fp16 rounding change the gradient. The instability shows up first on the
classes learnt from the fewest pixels.

### 4. KNN post-processing (evaluation only)

The spherical projection keeps **one point per pixel**. `inference.py` gives every discarded point
the label of the pixel it landed on, so occluded points silently inherit the label of whatever was
in front of them — and the error concentrates on exactly the classes that sit in front of or behind
something else. `--knn` re-labels each 3-D point by a vote over the k nearest pixels, where "near"
is measured in range, not in pixel distance (RangeNet++):

```bash
python inference.py config.yaml --checkpoint <ckpt> --knn --flip_tta
```

`--flip_tta` additionally averages the logits with those of the azimuth-mirrored scan — a spinning
LiDAR has no preferred handedness, so this is a label-preserving view. Neither needs retraining, and
`--knn` changes the 3-D metric only; the 2-D pixel metric is untouched.

### 5. Scan mixing

```yaml
dataset:
  mix:
    p_lasermix: 0.25   # swap inclination bands (groups of beams) with another scan
    p_polarmix: 0.25   # swap azimuth sectors with another scan
```

LaserMix respects how a spinning LiDAR samples the world, so the mixed scan still looks like a
plausible scan; PolarMix swaps wedges. Both need no instance annotations, so both work on nuScenes.
Each mixed sample reads a second scan, so raise `num_workers` when you turn these on.

### 6. Loss-side weighting

```yaml
dataset:
  cls_weight_scheme: log_inv    # log_inv (RangeViT) | salsanext | inv_sqrt | uniform
training:
  lovasz_weight: 1.0            # 1.5-3.0 is worth a sweep
  focal_gamma:   2.0
```

| scheme | formula | rare:road weight ratio |
|---|---|---|
| `log_inv` *(default, RangeViT)* | `log(1 + 1/(freq + eps))` | 3.8 : 1 |
| `salsanext` | `1/log(1.02 + freq)` (SalsaNext, Cylinder3D) | 10 : 1 |
| `inv_sqrt` | `1/sqrt(freq)` | 71 : 1 |
| `uniform` | — | 1 : 1 |

Aggressive weighting on its own trades head-class IoU for tail-class recall. It pays off once the
tail classes are actually *in* the batch often enough — i.e. together with (1) and (2), not instead
of them. Lovász is the term that optimises IoU directly and is the one that moves the tail classes;
upweighting it is the cheapest loss-side knob.

### 7. Backbone capacity

Two things are worth questioning about the frozen-trunk default when the tail classes plateau:

* **`cond_timestep: 0`.** The trunk is a *denoiser*. At `t = 0` it is asked to denoise an already-clean
  latent, which is close to the identity — the timestep where diffusion features are most
  semantically discriminative is typically well inside the schedule, not at either end. This is a
  one-line sweep with no memory cost: `--cond_timestep 0 / 50 / 100 / 200 / 400`. Do it early, it is
  cheap and it can move everything.
* **9.4 M trainable parameters.** Small objects are decided by a 32 × 48 token grid upsampled 2 × 8 in
  one shot. If the tail classes are still flat after (1)–(6), spend memory in this order:
  `unfreeze_adaln` with `adaln_bias_only: false` → `unfreeze_attn` → finer `patch_stride: [2, 4]`
  (doubles the tokens, directly helps small objects) → `freeze_dit_encoder: false` at `lr: 1e-5`.

### Validate on the metric you are going to report

The 2-D pixel mIoU that training has always validated on is a **proxy**. The spherical projection
keeps one point per pixel, so the pixel metric is blind to the 10-20 % of points it discards — and
blind to them *unevenly*, because thin and occluded classes lose the most. Selecting
`best_mean_iou_model.pth` on it means selecting on the wrong quantity, and reporting it in a paper
means reporting a number the benchmark does not score.

```yaml
training:
  val_3d:  true     # also score the point-wise mIoU during validation
  val_knn: true     # with KNN post-processing, so it matches inference.py --knn
```

This adds `best_point_mean_iou_model.pth` and `best_point_tail_mean_iou_model.pth`, logs per-class
point IoU to Weights & Biases, and prints the gap so you can see how much the projection is costing
you:

```
>>> Validation mIoU 0.5734 | tail mIoU 0.3011 (...) | worst motorcyclist 0.0000
>>> Validation POINT mIoU 0.5488 | point tail mIoU 0.2702  (pixel - point = +0.0246)
```

Validation gets slower (the point clouds are loaded and back-projected), so pair it with a larger
`val_frequency` if that matters.

### Reading the logs

Validation now prints and logs a **`tail_mean_iou`** (mean IoU over `rare_classes`) and the **worst
single class**, and writes per-class IoU to Weights & Biases every validation:

```
>>> Validation mIoU 0.5734 | tail mIoU 0.3011 (bicycle,motorcycle,...) | worst motorcyclist 0.0000
```

A third checkpoint, `best_tail_mean_iou_model.pth`, is saved alongside the existing ones. Overall
mIoU is dominated by the head classes and barely moves when a tail class collapses to zero, so it is
the wrong quantity to select on when the goal is to do well on *all* classes.

---

## 5. Training

```bash
# SemanticKITTI
python train.py config.yaml      --data_root /your/semantickitti/root --save_path ./log

# nuScenes
python train.py config_nusc.yaml --data_root /your/nuscenes/root      --save_path ./log
```

Every command-line flag is optional and overrides the config file
(`--id`, `--batch_size`, `--num_workers`, `--pretrained_model`, `--log_frequency`, `--seed`,
`--lr`, `--n_epochs`, `--warmup_epochs`, `--val_frequency`, `--window_stride`, `--ema_decay`,
`--rare_crop_p`, `--cls_weight_scheme`, `--lovasz_weight`, `--cond_timestep`).

**Resume from a checkpoint:**

```bash
python train.py config.yaml --checkpoint log/log_rangedit_semantickitti/checkpoint/checkpoint.pth
```

This restores the optimizer and AMP scaler, so use it for interrupted runs.

**Fine-tune from the best weights with a fresh optimizer:**

```bash
python train.py config.yaml \
    --finetune_from log/log_rangedit_semantickitti/checkpoint/best_mean_iou_model.pth \
    --lr 5e-5 --batch_size 6 --n_epochs 80 --warmup_epochs 1 --val_frequency 2 --window_stride 256
```

This is the recommended next step after a converged run: keep the trained model weights, restart
AdamW with a lower learning rate, and validate with overlapping sliding windows.

**Validation only:**

```bash
python train.py config.yaml --val_only --checkpoint <path/to/checkpoint.pth>
```

### What happens during training

1. Each scan is loaded, randomly augmented in 3-D, then projected to a 64 × 2048 (nuScenes:
   32 × 2048) range image with 5 channels `[range, x, y, z, intensity]` and randomly cropped to
   64 × 384 (nuScenes: 32 × 384).
2. The conv stem turns the crop into 32 × 48 (nuScenes: 16 × 48) tokens of dimension 1152, and the
   resized DiT sin-cos positional embedding is added.
3. The 28 frozen DiT blocks process the tokens, modulated by the conditioning vector.
4. The up-conv decoder upsamples the token grid back to 64 × 384 class logits, using the
   full-resolution stem features as a skip connection.
5. The loss is `focal_weight * focal + lovasz_weight * Lovász-softmax`, computed in fp32 even under
   AMP. On SemanticKITTI the focal weights come from the class point frequencies; on nuScenes they
   are uniform, as in RangeViT. Class 0 is ignored, and the focal term is masked to valid LiDAR
   pixels.
6. AdamW with a warmup + cosine schedule stepped once per iteration, gradient-norm clipping, and an
   EMA of the trainable weights.
7. Checkpoints are written to `<save_path>/log_<id>/checkpoint/`: `checkpoint.pth` every epoch (live
   weights, for resuming) plus `best_mean_iou_model.pth` / `best_mean_acc_model.pth` /
   `best_tail_mean_iou_model.pth` (EMA weights, the ones validation was scored with).

### Monitoring

The log goes to stdout and to `<save_path>/log_<id>/log/console.log`, in the RangeViT format —
`E[total epochs|current]`, `I[total iters|current]`, data time, process time, learning rate, loss,
running accuracy and IoU, and the estimated remaining time:

```
>>> Train E[050|001] I[2321|0001] DT[0.412] PT[0.633] LR 0.0 Loss 3.1204 Acc 0.0512 IOU 0.0148 RT 12:41:09
>>> Train E[050|001] I[2321|0101] DT[0.008] PT[0.271] LR 8.6e-06 Loss 2.4417 Acc 0.1839 IOU 0.0921 RT 10:58:22
...
>>> Validation E[050|001] I[4071|0001] DT[0.005] PT[0.728] LR 8.6e-06 Loss 2.7087 Acc 0.2928 IOU 0.1193 RT 0:52:10
```

To mirror the scalar logs to Weights & Biases, install `wandb` and pass `--use_wandb`. On a cluster
node without network/login access, use `--wandb_mode offline` and sync the run later:

```bash
python train.py config.yaml --use_wandb --wandb_project rangediffseg --wandb_mode offline
```

Every `val_frequency` epochs (and every `train_result_frequency` epochs for the training split) the
per-class table is printed:

```
============ Pixel-wise Evaluation Results (2D Eval) ============
Acc avg: 0.8912, IOU avg: 0.5734, Recall avg: 0.6521
+----------+---------------+---------+---------+---------+
| Class ID |  Class Name   |   IOU   |   Acc   | Recall  |
+----------+---------------+---------+---------+---------+
|    1     |      car      |  0.9012 |  0.9421 |  0.9502 |
|    2     |    bicycle    |  0.3471 |  0.5210 |  0.4832 |
...
+----------+---------------+---------+---------+---------+
---- Latext Format String ----
 & 90.1 & 34.7 & ... & 57.3
```

---

## 6. Evaluation / Testing

```bash
python inference.py config.yaml \
    --data_root /your/dataset/root \
    --checkpoint log/log_rangedit_semantickitti/checkpoint/best_mean_iou_model.pth
```

**Evaluate on the test split and export predictions for the benchmark server:**

```bash
python inference.py config.yaml \
    --checkpoint <path/to/best_mean_iou_model.pth> \
    --split test --save_eval_results
```

### What happens during evaluation

1. Full-width range images are processed with a **sliding window** (`window_size` /
   `window_stride`); overlapping logits are averaged.
2. The `(20, H, W)` logits are argmax-ed into a 2-D label map.
3. Each 3-D point is mapped back to its projected pixel through the cached `(px, py)` indices,
   giving a per-point label. Pass `--knn` to instead re-label each point by a range-weighted vote
   over its neighbouring pixels (RangeNet++), which fixes the points the projection discarded; pass
   `--flip_tta` to average the logits with the azimuth-mirrored scan. See §4b.
4. Both the **2-D pixel-wise** and the **3-D point-wise** mIoU are reported, per class, class 0
   excluded. The point-wise number is the one comparable to the SemanticKITTI leaderboard.
5. With `--save_eval_results`, predictions are written in each benchmark's format under
   `<save_path>/Eval_<split>/preds/`: SemanticKITTI as `.label` uint32 files in the original label
   space (`sequences/<seq>/predictions/`), nuScenes as `<sample_data_token>_lidarseg.bin` uint8
   files carrying the 17-class training labels (`lidarseg/`) — check the current lidarseg
   submission spec before uploading those.

---

## 7. Key Design Choices

| Choice | Detail |
|---|---|
| **Discriminative, not generative** | The DiT is used as a feature extractor in a single forward pass — no reverse diffusion at inference, following RangeViT rather than SegDiff. |
| **Conv stem over patch embedding** | The 2×2 latent patch embedding of DiT cannot ingest 5-channel range images; the SalsaNext-style stem also supplies the decoder's skip connection. |
| **Anisotropic 2×8 patches** | Range images are far wider than they are tall; 64 × 384 → 32 × 48 tokens. |
| **Resized sin-cos positional embedding** | DiT's fixed 16 × 16 grid is bilinearly resized to the range-view grid and then fine-tuned. It is re-interpolated on the fly whenever the input size changes. |
| **Pre-trained conditioning as a starting point** | The adaLN-Zero pathway is fed the pre-trained fixed-timestep + null-class embedding, plus a zero-initialised learnt offset. |
| **`unfreeze_adaln` replaces `unfreeze_layernorm`** | DiT LayerNorms have no affine parameters; the modulation is where per-block scale/shift lives. Its bias alone is enough under static conditioning. |
| **Focal + Lovász loss** | Same combination and class weighting as RangeViT. |
| **Sliding-window evaluation** | Trained on 64 × 384 crops, evaluated on full-width images. |

---

## 7b. Architecture — what is worth changing

The recipe is RangeViT's, but the trunk is not a ViT, and three of RangeViT's assumptions do not
transfer. These knobs all live in the `model` section and all default to the current behaviour.

### The question to answer first: does the trunk earn its place?

RangeViT's frozen encoder is an ImageNet **classifier** — its features are trained to be linearly
separable by category. DiT is a **denoiser**: it was never asked to produce a semantic
representation, and nobody has shown that a frozen diffusion trunk beats no trunk on range images.
Before tuning anything else, find out:

```bash
python train.py config.yaml --id ablate_bypass --n_trunk_blocks 0    # stem -> decoder, no trunk
python train.py config.yaml --id ablate_half   --n_trunk_blocks 14
python train.py config.yaml --id ablate_full                          # all 28 (default)
```

`n_trunk_blocks: 0` bypasses the trunk entirely; the blocks are physically dropped after the
checkpoint is loaded, so the run is also much cheaper. If bypass lands within noise of the full
model, the 677 M-parameter trunk is decorative and every further architectural effort belongs in the
stem and the decoder. If half the depth matches full depth, you have just halved your compute. This
is the single most informative run in this file and it costs one training job.

### `feature_taps` — read the trunk where the semantics are

`DiTEncoder.forward` currently feeds only `blocks[-1]` to the decoder. For a diffusion model that is
close to the worst choice available: the last blocks are specialised for predicting **noise**, a
high-frequency, low-semantics target, while the features that separate object categories sit in the
middle of the stack. This is the central finding of DDPM-Seg (Baranchuk et al., ICLR 2022) and it is
why every frozen-diffusion-backbone segmenter taps several depths.

```yaml
model:
  feature_taps: [6, 13, 20]   # 0-based block indices; DiT-XL/2 has 28 blocks
```

Each tap gets its own LayerNorm (the residual stream grows in scale with depth, so block 6 and
block 20 activations are not comparable) and they are fused as
`last + proj(concat(taps))` with **`proj` zero-initialised**. At initialisation the encoder output is
bitwise identical to the plain model — the same trick adaLN-Zero and `cond_offset` use — so training
starts from exactly the current model and learns how much of each depth to pull in. `[]` disables it.

### `grad_checkpointing` — and what freezing actually saves

Freezing the trunk saves the weight gradients and the AdamW optimizer state (that is the 9.4 M vs
684 M figure), but it saves much less **activation** memory than it looks. The stem sits *upstream*
of the blocks, so backward still traverses all 28 of them, and:

* every nonlinearity (GELU, the attention softmax, LayerNorm) must retain its input to compute the
  gradient *with respect to its input*, whether or not its weights are frozen;
* `unfreeze_adaln` makes the per-block scales and gates trainable, which forces the tensors they
  multiply to be retained as well.

Measured on DiT-B/2 at 64 × 384, batch 2, by summing every tensor autograd retains for the backward
pass:

| | retained activations | trainable params |
|---|---|---|
| trunk trainable | 3956 MiB | 137.2 M |
| trunk frozen, `adaln_bias_only` | **3306 MiB** | 7.0 M |
| trunk frozen + `grad_checkpointing` | **1414 MiB** | 7.0 M |

Freezing removes **95 %** of the trainable parameters but only **16 %** of the activation memory.
`grad_checkpointing: true` recomputes each block in the backward pass instead and takes off another
**57 %** — it is what makes `patch_stride: [2, 4]` or an unfrozen trunk fit at all. Cost: roughly one
extra forward pass.

### `head_prior_bias` — stop the tail from collapsing in the first epoch

The classifier is a single 1×1 conv with a zero bias, so the model starts by predicting a roughly
uniform distribution over 20 classes. The first steps are then dominated by pushing the ~99.9 % of
pixels that belong to head classes away from uniform, and the rare classes are often driven to
near-zero logits before they are ever learnt — which is exactly the failure that leaves motorcyclist
at 0.0 IoU for a whole run. Setting the bias to the log class prior (the RetinaNet trick) starts the
head where that pressure is already resolved:

```yaml
model:
  head_prior_bias: true
```

### `decoder_stages` and `stem_pool` — where small objects actually live

The trunk resolves nothing below one 2×8 token. A pedestrian at 20 m spans ~3×4 pixels, so
**everything the tail classes get comes through the stem and the decoder** — and both of them
currently throw that signal away at one specific point each.

**`stem_pool`.** `ConvStem.proj_block` reduces full-resolution features to the token grid with
`nn.AvgPool2d` over a 3×9 window: a fixed low-pass filter applied immediately before the trunk. A
3-pixel-wide pole is averaged into its background and never reaches the transformer as anything
distinct. `stem_pool: dwconv` replaces it with a depthwise strided convolution **initialised to
exactly that uniform kernel** (verified to reproduce `AvgPool2d` to 1e-7), so the model starts where
it starts today and can learn a per-channel kernel that preserves thin structure. Cost: 7 k
parameters. `stem_pool: conv` makes patchification fully learnt, but at +8 M it nearly doubles the
default trainable budget on its own.

**`decoder_stages`.** One 2×8 pixel shuffle takes the 32×48 token grid straight to 64×384 — a 16×
area upsample decided by a single 1×1 convolution, with the full-resolution stem features as the
only skip. `decoder_stages: 2` splits it into (2, 4) then (1, 2) and gives the first stage its own
skip at the intermediate width, strided down from the stem features inside the decoder (the stem
interface is unchanged). On DiT-XL/2 it is also **0.8 M cheaper**, because the wide 1×1 that produces
`d_decoder · s0 · s1` channels is replaced by two narrower ones:

| decoder params (D_h 256, skip 256) | `stages: 1` | `stages: 2` |
|---|---|---|
| DiT-XL/2 (d = 1152) | 5.97 M | **5.19 M** |
| DiT-S/2 (d = 384) | 2.83 M | 3.62 M |

### Still worth trying, not implemented here

| Change | Why | Cost |
|---|---|---|
| `patch_stride: [2, 4]` | Halves the token footprint (32×48 → 32×96), the only change that lets the *trunk* see small objects at all. | 2× activations, 4× attention — needs `grad_checkpointing` |
| `reuse_pos_emb: false` | DiT's positional embedding is *analytic* 2-D sin-cos, and the code already generates it correctly at 32×48 in `DiTEncoder.__init__` — then overwrites it with a bilinearly **blurred** resize of the 16×16 checkpoint grid. RangeViT had to interpolate because ViT's embedding is learnt; here you do not. | free A/B |
| Drop `cond_mode: context` | It pools the stem features to one vector per scan, i.e. a single global modulation for a dense-prediction task. Low ceiling. | — |

### Running the ablations

`tools/run_ablations.py` runs a set of variants against one base config and tabulates the result. It
writes one config per variant, runs `train.py` with `--id <base_id>_<variant>` so nothing overwrites
anything, and rebuilds the table from the on-disk logs (so an interrupted sweep resumes with
`--skip_existing`, and `--table_only` just reprints).

```bash
# the decisive one: does the frozen trunk earn its 677 M parameters?
python -m tools.run_ablations config.yaml --data_root <root> --suite trunk --n_epochs 15

python -m tools.run_ablations config.yaml --data_root <root> --suite depth     # how deep is enough
python -m tools.run_ablations config.yaml --data_root <root> --suite taps      # where to read it
python -m tools.run_ablations config.yaml --data_root <root> --suite timestep  # cond_timestep
python -m tools.run_ablations config.yaml --data_root <root> --suite decoder   # stem + decoder
python -m tools.run_ablations config.yaml --data_root <root> --suite tail      # tail-class knobs

# or anything you like
python -m tools.run_ablations config.yaml --data_root <root> \
    --variant baseline \
    --variant 'taps:model.feature_taps=[6,13,20]' \
    --variant 'both:model.decoder_stages=2,model.stem_pool=dwconv'
```

```
+----------------+--------+-----------+-------------+-----------+--------+
| variant        | mIoU   | tail mIoU | worst class | worst IoU | status |
+----------------+--------+-----------+-------------+-----------+--------+
| full           | 0.5734 | 0.3011    | motorcyclist| 0.0000    | ok     |
| bypass         | ...    | ...       | ...         | ...       | ok     |
+----------------+--------+-----------+-------------+-----------+--------+
```

Rank variants on a **short budget first** (`--n_epochs 15`); only re-run the survivors to convergence.
Note that `tail mIoU` and `worst IoU` are the columns to read — overall mIoU is dominated by the head
classes and barely moves when a tail class collapses.

---

**Not implemented:** the optional KPConv 3-D refiner of RangeViT, and its scan-based (ring-order)
projection variant — this repository always uses the spherical projection. Back-projection uses the
cached projection indices by default, which corresponds to RangeViT's `use_kpconv: false` setting;
`--knn` adds the lighter RangeNet++ post-processing on top.

### Legacy

`diffusion/` still holds the DDPM U-Net from the first version of this repository (SegDiff-style
iterative denoising of the segmentation map). It is self-contained and importable, but it is no
longer wired into `train.py` / `inference.py`. Delete it if you do not want to keep that baseline.

---

## 8. Citations

```bibtex
@inproceedings{peebles2023dit,
  title     = {Scalable Diffusion Models with Transformers},
  author    = {Peebles, William and Xie, Saining},
  booktitle = {ICCV},
  year      = {2023}
}

@inproceedings{rangevit2023,
  title     = {RangeViT: Towards Vision Transformers for 3D Semantic Segmentation in Autonomous Driving},
  author    = {Ando, Angelika and Gidaris, Spyros and Bursuc, Andrei and Puy, Gilles and Boulch, Alexandre and Marlet, Renaud},
  booktitle = {CVPR},
  year      = {2023}
}

@article{segdiff2021,
  title   = {SegDiff: Image Segmentation with Diffusion Probabilistic Models},
  author  = {Amit, Tomer and Nachmani, Eliya and Toker, Tal and Wolf, Lior},
  journal = {arXiv:2112.00390},
  year    = {2021}
}
```
