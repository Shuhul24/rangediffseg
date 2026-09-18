"""
RangeFormer + conditional semantic-latent DiT for range-view LiDAR segmentation.

    5-ch range view -> REM -> Encoder S1 (F1) -> Encoder S2 -> latent condition C
                                                                   |
                       coarse head <----------------------------- C
                                                                   |
        semantic latent S_t --> conditional DiT (self-attn, cross-attn to C)
                                          |
                                        S0_hat -> Decoder S1 -> S2 -> logits

Selected with `model.model_type: rangeformer_dit_seg`. The existing RangeDiT
pipeline is untouched and remains the default.
"""

from .model import RangeFormerDiTSeg

__all__ = ['RangeFormerDiTSeg']
