"""Export LOPO cross-validation splits to pickle files.

Optional tool for inspecting or saving splits to disk. Not required
for training — emo-train can generate splits on-the-fly with --fold
and --num-folds.

Usage:
    emo-generate-splits --data-path data/processed/diema12_quat.npz \
        --output-dir data/processed/diema12_splits/ --num-folds 10
"""

import argparse
import pickle
from pathlib import Path

import pybvh_ml

from emo_mocap.data.splits import generate_lopo_splits, parse_diema_actor


def main():
    parser = argparse.ArgumentParser(
        description="Export LOPO cross-validation splits to pickle files"
    )
    parser.add_argument("--data-path", required=True,
                        help="Path to the preprocessed .npz file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write fold pickle files")
    parser.add_argument("--num-folds", type=int, required=True,
                        help="Number of folds (K)")
    args = parser.parse_args()

    # Load filenames from npz
    preprocessed = pybvh_ml.load_preprocessed(args.data_path)
    filenames = list(preprocessed.get("filenames", []))
    if not filenames:
        print(f"Error: no filenames found in {args.data_path}")
        return

    # Generate splits
    splits = generate_lopo_splits(filenames, args.num_folds)

    # Save to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary header
    print(f"{'Fold':<10} {'Held-out actors':<50} {'Train':>7} {'Val':>7}")
    print("-" * 78)

    for fold_idx, split_dict in enumerate(splits):
        fold_name = f"fold_{fold_idx + 1:02d}"
        fold_path = output_dir / f"{fold_name}.pkl"

        with open(fold_path, "wb") as f:
            pickle.dump(split_dict, f)

        # Identify held-out actors for summary
        val_actors = sorted({
            parse_diema_actor(fname) for fname, _ in split_dict["val"]
        })
        actors_str = ", ".join(val_actors)
        if len(actors_str) > 48:
            actors_str = actors_str[:45] + "..."

        print(f"{fold_name:<10} {actors_str:<50} "
              f"{len(split_dict['train']):>7} {len(split_dict['val']):>7}")

    print(f"\nSaved {len(splits)} fold files to {output_dir}/")


if __name__ == "__main__":
    main()
