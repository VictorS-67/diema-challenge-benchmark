"""Evaluate a trained model on the test set.

Usage:
    python -m emo_mocap.cli.evaluate --config configs/diema7_stgcn.yaml --checkpoint path/to/best.ckpt
"""

import argparse

import pytorch_lightning as pl

from emo_mocap.tools.config import load_config_with_overrides
from emo_mocap.models.registry import get_model
from emo_mocap.data.loader import Loader
from emo_mocap.training.lightning_model import LightningModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides (key=value)")
    args = parser.parse_args()

    cfg = load_config_with_overrides(args.config, args.override)

    # Build model architecture from config
    model_cls = get_model(cfg.model.type)
    model = model_cls.from_config(cfg)

    # Load trained weights via LightningModel
    lit_model = LightningModel.load_from_checkpoint(
        args.checkpoint,
        model=model,
        base_lr=cfg.training.base_lr,
        num_class=cfg.model.num_class,
    )

    # Build data module
    target_repr = getattr(cfg.data, "target_repr", "euler")
    loader = Loader(
        data_path=cfg.data.data_path,
        split_path=cfg.data.split_path,
        clip_length=cfg.training.clip_length,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        target_repr=target_repr,
        seed=cfg.data.seed,
    )

    trainer = pl.Trainer(deterministic=True)
    results = trainer.test(lit_model, datamodule=loader)

    if results:
        for key, value in results[0].items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
