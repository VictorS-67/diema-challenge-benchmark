"""Loader LightningDataModule for managing train/val/test data splits."""

import os
import pickle
from pathlib import Path

import torch.utils.data
import pytorch_lightning as pl

from emo_mocap.data.feeder import Feeder


def _default_num_workers() -> int:
    """Sensible default: half the available CPU cores, capped at 8.

    More than 8 workers rarely helps and uses extra memory. Capped rather
    than using all cores to leave headroom for the main process and OS.
    """
    return min((os.cpu_count() or 4) // 2, 8)


def _feeder_worker_init_fn(worker_id: int) -> None:
    """Re-seed each DataLoader worker's augmentation RNG independently.

    Without this, all workers fork an identical copy of the Feeder's _rng,
    causing every worker to apply the exact same random augmentations.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        if hasattr(dataset, "reseed_rng"):
            dataset.reseed_rng(worker_id)


class Loader(pl.LightningDataModule):
    """LightningDataModule that wraps Feeder datasets for each split.

    Loads a preprocessed npz file and a split pickle, creates Feeder
    instances for train/val/test, and provides DataLoaders for training.

    Args:
        data_path: path to the .npz data file (pybvh-ml format)
        split_path: path to the split dictionary pickle (mutually exclusive with split_dict)
        split_dict: in-memory split dictionary (mutually exclusive with split_path)
        clip_length: number of frames to sample (default: 64)
        batch_size: batch size for train/val (default: 64)
        num_workers: DataLoader workers (default: 4)
        target_repr: target representation for model input (default: 'euler')
        debug: if True, limit to 100 samples (default: False)
        seed: random seed for sampling (default: 255)
        augmentation_pipeline: pybvh_ml.AugmentationPipeline or None
        euler_orders: per-joint Euler orders (for quat→Euler conversion)
    """

    def __init__(
        self,
        data_path,
        split_path=None,
        split_dict=None,
        clip_length=64,
        batch_size=64,
        num_workers=None,
        target_repr="euler",
        debug=False,
        seed=255,
        augmentation_pipeline=None,
        euler_orders=None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else _default_num_workers()
        self.debug = debug
        self.clip_length = clip_length
        self.target_repr = target_repr
        self.seed = seed
        self.augmentation_pipeline = augmentation_pipeline
        self.euler_orders = euler_orders

        if split_dict is not None:
            self.split_dict = split_dict
        elif split_path is not None:
            with open(split_path, "rb") as f:
                self.split_dict = pickle.load(f)
        else:
            raise ValueError("Either split_path or split_dict must be provided")

        self.train_indices = [idx for _, idx in self.split_dict["train"]]
        self.val_indices = [idx for _, idx in self.split_dict["val"]]

        if "test" in self.split_dict and len(self.split_dict["test"]) > 0:
            self.test_indices = [idx for _, idx in self.split_dict["test"]]
        else:
            self.test_indices = self.val_indices

        if self.debug:
            debug_size = 100
            self.train_indices = self.train_indices[:debug_size]
            self.val_indices = self.val_indices[:debug_size]
            self.test_indices = self.test_indices[:debug_size]

    def setup(self, stage: str):
        common = dict(
            data_path=self.data_path,
            clip_length=self.clip_length,
            target_repr=self.target_repr,
            seed=self.seed,
            euler_orders=self.euler_orders,
        )

        if stage == "fit":
            self.dataset_train = Feeder(
                indices=self.train_indices,
                test=False,
                augmentation_pipeline=self.augmentation_pipeline,
                **common,
            )
            self.dataset_val = Feeder(
                indices=self.val_indices,
                test=True,
                **common,
            )

        if stage in ("test", "predict"):
            self.dataset_test = Feeder(
                indices=self.test_indices,
                test=True,
                **common,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_feeder_worker_init_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
