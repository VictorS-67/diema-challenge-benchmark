# DIEM-A Emotion Recognition Benchmark

Baseline benchmark for emotion recognition from 3D motion capture skeletal data, built on the [DIEM-A dataset](https://www.cr-ict.riec.tohoku.ac.jp/diem-a/) (ACII 2025).

## Task

Classify 12 emotions (anger, contempt, disgust, fear, joy, sadness, surprise, jealousy, shame, guilt, gratitude, pride) from skeleton motion sequences captured in a full-body mocap suit.

## Baseline Model

**STGCN++** (Spatio-Temporal Graph Convolutional Network [[1]](https://arxiv.org/abs/1801.07455), multi-branch variant [[2]](https://arxiv.org/abs/2205.09443))

- Input representation: 6D continuous rotations ([Zhou et al.](https://arxiv.org/abs/1812.07035))
- 10-layer GCN: 64 → 128 → 256 channels
- 25-joint DIEM-A skeleton
- On-the-fly augmentation: yaw rotation, lateral mirroring, speed perturbation, joint noise

## Baseline Results

Evaluation uses Leave-Performer-Out (LPO) cross-validation: the dataset's 92 performers are split into K groups, and each fold holds out one group entirely for testing while training on the rest. This ensures the model is always evaluated on performers it has never seen during training, testing its ability to generalize across individuals rather than memorize actor-specific movement patterns.

10-fold LPO on 9,935 clips from 92 performers:

| Metric   | Mean   | Std    |
|----------|--------|--------|
| Accuracy | 27.11% | 3.67%  |
| F1 Score | 25.21% | 4.49%  |

Per-fold results in [`logs/lpo_10fold_results.csv`](logs/lpo_10fold_results.csv).

Random baseline: 8.33% (1/12 classes).

Tests were run on Ubuntu 20.04 with an NVIDIA RTX 4090 (24 GB VRAM).

## Setup

```bash
# Create a conda environment (Python 3.11 recommended)
conda create -n emo_mocap python=3.11
conda activate emo_mocap

# Install
pip install -e .
```

## Data Preparation

The DIEM-A dataset must be obtained separately (research license). Place BVH files in `data/raw/diema_bvh/`, then preprocess:

```bash
emo-preprocess --input data/raw/diema_bvh/ \
    --output data/processed/diema12_quat.npz \
    --emo2idx configs/emo_to_idx_12.txt
```

This converts BVH files to quaternion format, filtering to the 12 emotions listed in `configs/emo_to_idx_12.txt`.

## Training

Single fold:

```bash
emo-train --config configs/diema12_stgcn.yaml --fold 1 --num-folds 10 --test-after
```

All 10 LPO folds:

```bash
make lpo-train CONFIG=configs/diema12_stgcn.yaml FOLDS=10
```

Monitor training:

```bash
tensorboard --logdir logs/
```

## Configuration

All experiment parameters are in [`configs/diema12_stgcn.yaml`](configs/diema12_stgcn.yaml). Key settings:

```yaml
model:
  type: stgcn
  num_class: 12
  in_channels: 6          # 6D rotation representation
  plusplus: true           # multi-branch temporal convolution

training:
  base_lr: 0.2
  max_epochs: 65
  batch_size: 128
  early_stopping: true
  early_stopping_patience: 10

augmentation:
  enabled: true
  rotate: true             # random yaw rotation
  mirror: true             # lateral flip (50%)
  speed: true              # temporal stretching [0.8, 1.2]
  noise_sigma: 2.5         # joint noise in degrees
```

## References

1. Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *AAAI*. https://arxiv.org/abs/1801.07455
2. Duan, H., et al. (2022). Revisiting Skeleton-based Action Recognition. *CVPR*. https://arxiv.org/abs/2205.09443

## Dependencies

- [pybvh](https://pypi.org/project/pybvh/) -- BVH file parsing
- [pybvh-ml](https://pypi.org/project/pybvh-ml/) -- ML data pipeline for skeleton data
- PyTorch, PyTorch Lightning, TensorBoard
