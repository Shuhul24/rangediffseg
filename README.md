# RangeDiffSeg / RangeDiT

**LiDAR semantic segmentation on range-view images with an off-the-shelf Diffusion Transformer (DiT)**

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
      ▼  Spherical projection
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
│   ├── augmentor.py         # Random 3D augmentations (flip, translate, rotate, scale)
│   ├── semantic_kitti.py    # SemanticKITTI parser, label remapping, class statistics
│   └── range_view_loader.py # PyTorch Dataset: projection, random crops, (feature, label, mask)
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
│   └── tools.py             # Meters, remaining-time estimate, logger, result tables
├── diffusion/               # Legacy DDPM U-Net baseline (see "Legacy" below)
├── option.py                # config.yaml → settings object, CLI overrides
├── train.py                 # Training entry point
├── inference.py             # Evaluation entry point (2-D pixel and 3-D point mIoU)
├── config.yaml              # All hyperparameters — primary file to edit
└── requirements.txt
```

---

## 1. Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers `torch`, `numpy`, `scipy` and `pyyaml`. `tensorboard` is optional — if
it is installed, scalars are written next to the console log. `timm`, `einops` and `prettytable`
(used by the upstream repositories) are **not** required: the DiT blocks, the anisotropic pixel
shuffle and the result tables are implemented here directly.

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

| Section | Key | Default | When to change |
|---|---|---|---|
| `dataset` | `proj_h` / `proj_w` | `64` / `2048` | Match your LiDAR (e.g. 32-beam → `proj_h: 32`) |
| `model` | `backbone` | `DiT-XL/2` | Only DiT-XL/2 has released weights; `DiT-B/2`, `DiT-S/2`… for from-scratch runs |
| `model` | `pretrained_model` | `DiT-XL-2-256x256.pt` | `DiT-XL-2-512x512.pt`, a local path, or `null` for scratch |
| `model` | `image_size` | `[64, 384]` | Size of the random crop the network is trained on |
| `model` | `patch_size` / `patch_stride` | `[2, 8]` | Larger patches → fewer tokens → less memory |
| `model` | `D_h` | `256` | Hidden width of the stem and of the decoder |
| `model` | `skip_filters` | `256` | `0` disables the stem → decoder skip connection (else must equal `D_h`) |
| `model` | `conv_stem` | `ConvStem` | `none` uses a plain patch embedding (then `reuse_patch_emb` becomes possible) |
| `model` | `cond_timestep` | `0` | Which diffusion timestep the frozen `t_embedder` is evaluated at |
| `model` | `cond_mode` | `static` | `context` also conditions on the pooled stem features |
| `model` | `freeze_dit_encoder` | `true` | `false` fine-tunes the whole trunk (much more memory) |
| `model` | `unfreeze_adaln` | `true` | Trains the adaLN-Zero modulation (the DiT analogue of unfreezing LayerNorms) |
| `model` | `adaln_bias_only` | `true` | Trains only the modulation bias — same expressive power, 1/`d_model` of the parameters |
| `model` | `unfreeze_attn` / `unfreeze_ffn` | `false` | Progressively unfreeze attention / MLP layers |
| `model` | `window_size` / `window_stride` | `[64, 384]` | Sliding window used at validation and test time |
| `training` | `batch_size` | `4` | Adjust to fit your GPU VRAM |
| `training` | `lr` | `2e-4` | AdamW; lower it (e.g. `1e-5`) when the whole trunk is unfrozen |
| `training` | `n_epochs` | `50` | |
| `training` | `save_path` / `id` | `./log` / `rangedit_semantickitti` | Logs go to `<save_path>/log_<id>/` |

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

---

## 5. Training

```bash
python train.py config.yaml --data_root /your/dataset/root --save_path ./log
```

Every command-line flag is optional and overrides the config file
(`--id`, `--batch_size`, `--num_workers`, `--pretrained_model`, `--log_frequency`, `--seed`).

**Resume from a checkpoint:**

```bash
python train.py config.yaml --checkpoint log/log_rangedit_semantickitti/checkpoint/checkpoint.pth
```

**Validation only:**

```bash
python train.py config.yaml --val_only --checkpoint <path/to/checkpoint.pth>
```

### What happens during training

1. Each scan is loaded, randomly augmented in 3-D, then projected to a 64 × 2048 range image with
   5 channels `[range, x, y, z, intensity]` and randomly cropped to 64 × 384.
2. The conv stem turns the crop into 32 × 48 tokens of dimension 1152, and the resized DiT sin-cos
   positional embedding is added.
3. The 28 frozen DiT blocks process the tokens, modulated by the conditioning vector.
4. The up-conv decoder upsamples the token grid back to 64 × 384 class logits, using the
   full-resolution stem features as a skip connection.
5. The loss is `focal + Lovász-softmax`, with the focal weights derived from the SemanticKITTI
   class frequencies. Class 0 is ignored, and the focal term is masked to valid LiDAR pixels.
6. AdamW with a warmup + cosine schedule stepped once per iteration.
7. Checkpoints are written to `<save_path>/log_<id>/checkpoint/`: `checkpoint.pth` every epoch plus
   `best_mean_iou_model.pth` / `best_mean_acc_model.pth`.

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
   giving a per-point label — no KNN needed.
4. Both the **2-D pixel-wise** and the **3-D point-wise** mIoU are reported, per class, class 0
   excluded. The point-wise number is the one comparable to the SemanticKITTI leaderboard.
5. With `--save_eval_results`, predictions are written as `.label` files in the original
   SemanticKITTI label space, under `<save_path>/Eval_<split>/preds/sequences/<seq>/predictions/`.

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

**Not implemented:** the optional KPConv 3-D refiner of RangeViT. Back-projection uses the cached
projection indices, which corresponds to RangeViT's `use_kpconv: false` setting.

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
