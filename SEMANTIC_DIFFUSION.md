# Conditional semantic diffusion

This path trains the DiT to predict clean semantic labels from corrupted label
codes, conditioned on a five-channel range image and occupancy. It is distinct
from the original pretrained DiT feature-transfer and input-noise auxiliary runs.
The original entry points/configs remain available.

## Architecture and objective

- Fixed centered one-hot codes for 19 classes; ignore label 0 has no class channel.
- Full-resolution label codes are packed by 2x8 convolutional patches; integer
  labels are never downsampled or interpolated.
- A convolutional condition encoder supplies aligned geometry, explicit grid
  positions, pooled multiscale context, and gated fine detail to a six-block DiT.
- Window self-attention, pooled cross-attention, depthwise spatial mixing and
  timestep-conditioned residual gates. This is geometry conditioning, not a
  claim of explicit depth-distance attention biases.
- Predict x0 class logits with valid-pixel CE + Lovasz. There is no separate
  coarse segmentor or standalone segmentation branch.
- Cosine corruption, configurable label-code scale, detached self-conditioning,
  and optional corruption of the model's own first prediction after warmup.
- Pure-noise initialization and deterministic DDIM updates given the initial
  seed; experiment with 1/3/5 steps. No ground truth is used during sampling.
- No circular padding on arbitrary crops. XYZ channels provide geometric context;
  crop-grid position encodings are not absolute global azimuth coordinates.

This is a first continuous-label diffusion experiment. Categorical diffusion and
a learned semantic VAE are alternative future ablations, not mixed into this run.

## Train

```bash
sbatch scripts/run_semantic_diffusion.sh
# Small real-data GPU verification, including unit tests:
SMOKE=1 sbatch --time=00:30:00 scripts/run_semantic_diffusion.sh
```

Each submission gets a unique scratch output directory. The default uses one GPU,
100 epochs, microbatch 4 and accumulation 4 (effective batch 16), AdamW with a
three-epoch warmup, BF16 when supported, and gradient clipping. The scheduler
counts optimizer updates; partial accumulation windows are normalized correctly.
Validation is full sequence 08 at epoch 1, every 5 epochs and the final epoch.
Train sequences exclude 08. Evaluation uses overlapping windows, fixed per-window
noise seeds, and both pixel and original-point IoU; selection uses point mIoU.
Projection back-mapping is direct (no KNN), consistently across this experiment.

`last.pt` includes optimizer/config/step/epoch and is written atomically. `best.pt`
is a smaller evaluation checkpoint. Restart from `last.pt` with the same config:

```bash
RESUME=/scratch/path/last.pt RUN_DIR=/scratch/path sbatch scripts/run_semantic_diffusion.sh
```

Epoch seeding reconstructs data order, augmentation and diffusion noise at restart;
bitwise GPU determinism is not promised. Mid-epoch preemption repeats that epoch.
The five-day job limit may require restarting before 100 epochs complete.

## Evaluate

```bash
python train_semantic_diffusion.py --config config_semantic_diffusion.yaml \
  --resume /scratch/path/best.pt --output /scratch/path/evaluation --eval-only --steps 3
```

Compare 1, 3, and 5 steps at identical settings. `metrics.jsonl` includes all 19
per-class IoUs, point and pixel mIoU, validation elapsed time, and scan count.
No benchmark improvement is claimed until validation completes.

## Controls

Set `model.mode: deterministic` for a matched one-pass zero-state transformer.
A matched recurrent control requires training with recurrent unrolling and is not
provided by merely evaluating that one-pass model repeatedly. Disable
self_condition_probability or prediction_noise_probability separately to ablate
those changes. Keep data/augmentation/evaluation identical across runs.

References: SegDiff (arXiv:2112.00390), DDP (arXiv:2303.17559), DDPS
(arXiv:2306.01721), and DiT (arXiv:2212.09748). This implementation is a new
range-view adaptation, not an official reproduction of those methods.
