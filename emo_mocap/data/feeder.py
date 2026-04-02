"""Feeder dataset for loading preprocessed motion capture data.

Loads quaternion data from pybvh-ml's npz format, applies augmentation
in quaternion space, converts to the target representation, and packs
to (C, T, V) tensors for the model.
"""

import numpy as np
import torch
import torch.utils.data

import pybvh_ml


class Feeder(torch.utils.data.Dataset):
    """PyTorch Dataset that loads preprocessed skeleton sequences.

    The pipeline per sample:
    1. Load quaternion data from npz (root_pos + joint_quats)
    2. Augment in quaternion space (train mode only)
    3. Convert to target representation (Euler, 6D, etc.)
    4. Pack to (C, T, V) layout
    5. Temporal sample to fixed clip_length

    Args:
        data_path: path to the .npz file (pybvh-ml format)
        indices: optional list of clip indices to select a subset
        clip_length: number of frames to sample per sequence (default: 64)
        target_repr: target rotation representation for model input
            ('euler', '6d', 'quaternion', 'axisangle', 'rotmat')
        test: if True, use deterministic sampling (default: False)
        seed: random seed for reproducibility (default: 255)
        augmentation_pipeline: optional pybvh_ml.AugmentationPipeline
            (only applied in training mode)
        euler_orders: per-joint Euler orders (required when target_repr='euler')
    """

    def __init__(self, data_path, indices=None, clip_length=64,
                 target_repr="euler", test=False, seed=255,
                 augmentation_pipeline=None, euler_orders=None):
        preprocessed = pybvh_ml.load_preprocessed(data_path)
        self.clips = preprocessed["clips"]
        self.labels = preprocessed.get("labels")
        self.filenames = preprocessed.get("filenames", [])
        self.skeleton_info = preprocessed.get("skeleton_info", {})

        if indices is not None:
            self.clips = [self.clips[i] for i in indices]
            if self.labels is not None:
                self.labels = self.labels[indices]
            if self.filenames:
                self.filenames = [self.filenames[i] for i in indices]

        self.clip_length = clip_length
        self.target_repr = target_repr
        self.mode = "test" if test else "train"
        self.pipeline = augmentation_pipeline
        self._base_seed = seed
        self._rng = np.random.default_rng(seed)

        # Euler orders from skeleton_info (needed for quat→Euler conversion)
        self.euler_orders = euler_orders or self.skeleton_info.get("euler_orders")

    def reseed_rng(self, worker_id: int) -> None:
        """Re-seed the augmentation RNG for a specific DataLoader worker.

        Called by the DataLoader worker_init_fn so that each worker uses a
        distinct random stream, avoiding identical augmentations across workers.
        """
        self._rng = np.random.default_rng(self._base_seed + worker_id)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        root_pos = clip["root_pos"].copy()       # (F, 3)
        joint_data = clip["joint_data"].copy()    # (F, J, 4) quaternions

        # 1. Augment in quaternion space (train mode only)
        if self.pipeline is not None and self.mode == "train":
            joint_data, root_pos = self.pipeline(joint_data, root_pos, rng=self._rng)

        # 2. Convert to target representation
        if self.target_repr != "quaternion":
            joint_data = pybvh_ml.convert_arrays(
                joint_data, "quaternion", self.target_repr,
                euler_orders=self.euler_orders,
            )  # (F, J, C_target)

        # 3. Pack to CTV
        # center_root=False: centering was done at preprocessing time
        data_ctv = pybvh_ml.pack_to_ctv(root_pos, joint_data, center_root=False)

        # 4. Temporal sample
        num_frames = data_ctv.shape[1]
        frame_indices = pybvh_ml.uniform_temporal_sample(
            num_frames, self.clip_length, mode=self.mode, rng=self._rng,
        )
        frame_indices = frame_indices % num_frames
        data_ctv = data_ctv[:, frame_indices, :]  # (C, clip_length, V)

        # 5. Return
        label = int(self.labels[idx]) if self.labels is not None else -1
        filename = self.filenames[idx] if self.filenames else ""

        return (
            torch.tensor(data_ctv, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            filename,
        )
