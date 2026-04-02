"""Train an emotion recognition model.

Usage:
    # Single split (from config or override)
    emo-train --config configs/diema7_stgcn.yaml
    emo-train --config configs/diema7_stgcn.yaml --override training.max_epochs=50

    # LOPO cross-validation (on-the-fly split, no pkl files needed)
    emo-train --config configs/diema12_stgcn.yaml --fold 3 --num-folds 10
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

import pybvh_ml

from emo_mocap.tools.config import load_config_with_overrides
from emo_mocap.models.registry import get_model
from emo_mocap.data.loader import Loader
from emo_mocap.data.splits import generate_lopo_splits
from emo_mocap.training.lightning_model import LightningModel


def _build_pipeline(cfg):
    """Build a pybvh_ml AugmentationPipeline from config.

    Returns None if augmentation is disabled.
    """
    aug_cfg = cfg.augmentation
    if not aug_cfg.enabled:
        return None

    skeleton = cfg.skeleton
    steps = []

    if getattr(aug_cfg, "rotate", False):
        lo, hi = aug_cfg.rotate_range
        steps.append((
            pybvh_ml.rotate_quaternions_vertical, 1.0,
            {"angle_deg": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi),
             "up_idx": getattr(skeleton, "up_idx", 1)},
        ))

    if getattr(aug_cfg, "mirror", False):
        lr_pairs = [tuple(p) for p in skeleton.lr_joint_pairs]
        steps.append((
            pybvh_ml.mirror_quaternions, aug_cfg.mirror_prob,
            {"lr_joint_pairs": lr_pairs,
             "lateral_idx": getattr(skeleton, "lateral_idx", 0)},
        ))

    if getattr(aug_cfg, "speed", False):
        lo, hi = aug_cfg.speed_range
        steps.append((
            pybvh_ml.speed_perturbation_arrays, 1.0,
            {"factor": lambda rng, lo=lo, hi=hi: rng.uniform(lo, hi)},
        ))

    noise_sigma = getattr(aug_cfg, "noise_sigma", 0.0)
    if noise_sigma > 0:
        steps.append((
            pybvh_ml.add_joint_noise_quaternions, 1.0,
            {"sigma_deg": noise_sigma},
        ))

    return pybvh_ml.AugmentationPipeline(steps) if steps else None


def _build_lopo_split(data_path, fold, num_folds):
    """Generate a LOPO split dict on-the-fly for the given fold.

    Loads filenames from the npz, generates all K splits deterministically,
    and returns the split dict for the requested fold.
    """
    preprocessed = pybvh_ml.load_preprocessed(data_path)
    filenames = list(preprocessed.get("filenames", []))
    if not filenames:
        raise ValueError(
            f"No filenames found in {data_path}. "
            "Cannot generate LOPO splits without filename metadata."
        )
    all_splits = generate_lopo_splits(filenames, num_folds)
    return all_splits[fold - 1]  # fold is 1-indexed from CLI


def main():
    parser = argparse.ArgumentParser(description="Train an emotion recognition model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    parser.add_argument("--test-after", action="store_true", help="Run test after training")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold number for LOPO cross-validation (1-indexed)")
    parser.add_argument("--num-folds", type=int, default=None,
                        help="Total number of LOPO folds")
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)

    # Determine split: on-the-fly LOPO or from config
    use_lopo = args.fold is not None or args.num_folds is not None
    if use_lopo:
        if args.fold is None or args.num_folds is None:
            parser.error("--fold and --num-folds must be used together")
        if args.fold < 1 or args.fold > args.num_folds:
            parser.error(f"--fold must be between 1 and {args.num_folds}")
        split_dict = _build_lopo_split(cfg.data.data_path, args.fold, args.num_folds)
        split_path = None
    else:
        split_dict = None
        split_path = cfg.data.split_path

    # Build model
    model_cls = get_model(cfg.model.type)
    model = model_cls.from_config(cfg)

    # Build augmentation pipeline
    pipeline = _build_pipeline(cfg)

    # Build data module
    target_repr = getattr(cfg.data, "target_repr", "euler")
    loader = Loader(
        data_path=cfg.data.data_path,
        split_path=split_path,
        split_dict=split_dict,
        clip_length=cfg.training.clip_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        target_repr=target_repr,
        seed=cfg.data.seed,
        augmentation_pipeline=pipeline,
    )

    # Build Lightning model
    aux_loss_weights = cfg.training.aux_loss_weights
    if isinstance(aux_loss_weights, dict):
        pass  # already a dict
    else:
        aux_loss_weights = vars(aux_loss_weights) if hasattr(aux_loss_weights, "__dict__") else {}

    lit_model = LightningModel(
        model=model,
        base_lr=cfg.training.base_lr,
        num_class=cfg.model.num_class,
        optimizer=cfg.training.optimizer,
        scheduler_type=cfg.training.scheduler_type,
        scheduler_params=cfg.training.scheduler_params,
        weight_decay=cfg.training.weight_decay,
        aux_loss_weights=aux_loss_weights,
    )

    # Experiment name
    log_cfg = cfg.logging
    experiment_name = log_cfg.experiment_name
    if experiment_name is None:
        if use_lopo:
            experiment_name = f"{cfg.model.type}_fold{args.fold:02d}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{cfg.model.type}_{timestamp}"

    # Loggers and callbacks
    log_base = Path(log_cfg.log_dir) / experiment_name
    existing_versions = [
        int(m.group(1))
        for d in log_base.iterdir() if log_base.exists()
        if (m := re.match(r"^version_(\d+)$", d.name))
    ] if log_base.exists() else []
    version = max(existing_versions, default=-1) + 1

    csv_logger = CSVLogger(save_dir=log_cfg.log_dir, name=experiment_name, version=version)
    tb_logger = TensorBoardLogger(save_dir=log_cfg.log_dir, name=experiment_name, version=version)
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_acc:.4f}",
    )
    callbacks = [checkpoint_cb]
    if cfg.training.early_stopping:
        es_monitor = cfg.training.early_stopping_monitor
        es_mode = "min" if "loss" in es_monitor else "max"
        callbacks.append(EarlyStopping(
            monitor=es_monitor,
            mode=es_mode,
            patience=cfg.training.early_stopping_patience,
            verbose=True,
        ))

    # Ampere (compute capability 8.0+) and later GPUs have Tensor Cores that
    # can accelerate float32 matmuls at the cost of a small precision trade-off.
    # 'high' uses TF32 internally, which is standard practice for training.
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        deterministic=True,
        accelerator=cfg.training.accelerator,
    )

    trainer.fit(lit_model, datamodule=loader)

    if args.test_after:
        trainer.test(lit_model, datamodule=loader)

    print(f"Best checkpoint: {checkpoint_cb.best_model_path}")
    print(f"Best val_acc: {checkpoint_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
