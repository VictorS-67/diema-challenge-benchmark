"""Preprocess BVH files into npz format for training.

Stores data as quaternions — the canonical intermediate representation.
At training time, the Feeder converts to the target representation
(Euler, 6D, etc.) specified in the experiment config.

Only BVH files whose emotion appears in the emo2idx mapping are included.
This means the same raw BVH directory can produce different datasets
(e.g., 7-class or 13-class) by using different emo2idx files.

Usage:
    python -m emo_mocap.cli.preprocess --input data/raw/diema_bvh/ \
        --output data/processed/diema7_quat.npz \
        --emo2idx configs/emo_to_idx_7.txt

    With Bvh-level augmentation:
    python -m emo_mocap.cli.preprocess --input data/raw/diema_bvh/ \
        --output data/processed/diema7_quat.npz \
        --emo2idx configs/emo_to_idx_7.txt \
        --augment-copies 3 --augment-speed-range 0.8 1.2 --augment-dropout-rate 0.1
"""

import argparse
import tempfile
from pathlib import Path

import pybvh_ml


def _load_emo2idx(path):
    """Load emotion-to-index mapping from a text file.

    Expected format: one 'emotion index' pair per line.
    """
    emo2idx = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                emo2idx[parts[0]] = int(parts[1])
    return emo2idx


def _parse_diema_emotion(filename_stem):
    """Extract emotion string from DIEMA filename.

    Format: {nationality}_{performerID}_{emotion}_{scenario}_{intensity}
    Example: JP_06_anger_2_M → 'anger'
    """
    parts = filename_stem.split("_")
    return parts[2]


def _diema_label_fn(filename_stem, emo2idx):
    """Extract emotion label index from DIEMA filename."""
    emotion = _parse_diema_emotion(filename_stem)
    return emo2idx[emotion]


def _diema_filter_fn(filename_stem, emo2idx):
    """Return True if the file's emotion is in the mapping."""
    emotion = _parse_diema_emotion(filename_stem)
    return emotion in emo2idx


def _augment_and_preprocess(input_dir, output_path, emo2idx, copies,
                            speed_range, dropout_rate):
    """Preprocess with Bvh-level augmentation (speed perturbation, frame dropout).

    Filters, then generates augmented copies of included BVH files into a temp
    directory, and preprocesses everything at once.
    """
    import shutil
    import random
    import pybvh
    from pybvh.transforms import speed_perturbation, dropout_frames

    # Collect valid files
    valid_paths = [
        p for p in sorted(input_dir.glob("*.bvh"))
        if _diema_filter_fn(p.stem, emo2idx)
    ]

    with tempfile.TemporaryDirectory() as aug_dir:
        aug_path = Path(aug_dir)

        for bvh_path in valid_paths:
            # Copy original
            shutil.copy2(bvh_path, aug_path / bvh_path.name)

            # Generate augmented copies
            for i in range(copies):
                bvh = pybvh.read_bvh_file(bvh_path)

                if speed_range is not None:
                    factor = random.uniform(*speed_range)
                    bvh = speed_perturbation(bvh, factor)

                if dropout_rate is not None and dropout_rate > 0:
                    bvh = dropout_frames(bvh, dropout_rate)

                aug_stem = f"{bvh_path.stem}_aug{i:02d}"
                bvh.to_bvh_file(str(aug_path / f"{aug_stem}.bvh"))

        label_fn = lambda stem: _diema_label_fn(stem, emo2idx)
        return pybvh_ml.preprocess_directory(
            bvh_dir=aug_path,
            output_path=output_path,
            representation="quaternion",
            center_root=True,
            label_fn=label_fn,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BVH files into npz format (quaternion representation)"
    )
    parser.add_argument("--input", required=True, help="Directory containing BVH files")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--emo2idx", required=True,
                        help="Path to emotion-to-index mapping file")
    parser.add_argument("--augment-copies", type=int, default=0,
                        help="Number of augmented copies per sample (0 = disabled)")
    parser.add_argument("--augment-speed-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Speed perturbation range (e.g., 0.8 1.2)")
    parser.add_argument("--augment-dropout-rate", type=float, default=None,
                        help="Frame dropout rate for augmentation (e.g., 0.1)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    emo2idx = _load_emo2idx(args.emo2idx)

    if not any(input_dir.glob("*.bvh")):
        print(f"No BVH files found in {input_dir}")
        return

    if args.augment_copies > 0:
        result = _augment_and_preprocess(
            input_dir, output_path, emo2idx,
            args.augment_copies, args.augment_speed_range,
            args.augment_dropout_rate,
        )
    else:
        label_fn = lambda stem: _diema_label_fn(stem, emo2idx)
        filter_fn = lambda stem: _diema_filter_fn(stem, emo2idx)
        result = pybvh_ml.preprocess_directory(
            bvh_dir=input_dir,
            output_path=output_path,
            representation="quaternion",
            center_root=True,
            label_fn=label_fn,
            filter_fn=filter_fn,
        )

    print(f"Saved {result['num_clips']} clips to {output_path}")
    print(f"Representation: {result['representation']}")


if __name__ == "__main__":
    main()
