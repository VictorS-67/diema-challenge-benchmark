"""LOPO (Leave-One-Performer-Out) cross-validation split generation.

Generates K-fold splits where each fold holds out a group of actors
for validation/test, and uses the rest for training. Splits are
deterministic: sorted actors are distributed round-robin into K groups.

DIEMA filename format:
    {nationality}_{performerID}_{emotion}_{scenarioId}_{intensity}
    Example: JP_06_anger_1_H  →  actor ID = JP_06

Augmented filenames (e.g. JP_06_anger_1_H_aug00) are handled correctly:
the actor ID is always the first two underscore-separated parts.
"""

from collections import defaultdict


def parse_diema_actor(filename_stem: str) -> str:
    """Extract actor ID from a DIEMA filename stem.

    Args:
        filename_stem: filename without extension, e.g. 'JP_06_anger_1_H'
            or 'TW_30_joy_2_M_aug02'

    Returns:
        Actor ID string, e.g. 'JP_06' or 'TW_30'
    """
    parts = filename_stem.split("_")
    return f"{parts[0]}_{parts[1]}"


def generate_lopo_splits(
    filenames: list[str],
    num_folds: int,
    actor_fn=parse_diema_actor,
) -> list[dict]:
    """Generate K-fold LOPO splits from a list of filenames.

    Steps:
    1. Extract actor IDs from filenames via actor_fn
    2. Sort unique actors alphabetically
    3. Distribute actors round-robin into num_folds groups
    4. For each fold: held-out group → val and test, rest → train

    Round-robin ensures balanced folds. With 92 actors and 10 folds,
    folds 0-1 get 10 actors each, folds 2-9 get 9 each.

    Args:
        filenames: list of filename stems (one per clip in the dataset)
        num_folds: number of folds (K)
        actor_fn: callable that extracts an actor ID from a filename stem.
            Defaults to parse_diema_actor (DIEMA convention).

    Returns:
        List of K split dicts. Each dict has keys 'train', 'val', 'test'.
        Each value is a list of (filename, original_index) tuples,
        matching the format expected by Loader.
    """
    if num_folds < 2:
        raise ValueError(f"num_folds must be >= 2, got {num_folds}")

    # Group clip indices by actor
    actor_to_indices = defaultdict(list)
    for idx, fname in enumerate(filenames):
        actor_id = actor_fn(fname)
        actor_to_indices[actor_id].append(idx)

    # Sort actors alphabetically for determinism
    sorted_actors = sorted(actor_to_indices.keys())

    if num_folds > len(sorted_actors):
        raise ValueError(
            f"num_folds ({num_folds}) exceeds number of unique actors "
            f"({len(sorted_actors)})"
        )

    # Round-robin assignment: actor i goes to fold (i % num_folds)
    fold_actors = [[] for _ in range(num_folds)]
    for i, actor in enumerate(sorted_actors):
        fold_actors[i % num_folds].append(actor)

    # Build split dicts
    splits = []
    for fold_idx in range(num_folds):
        holdout_actors = set(fold_actors[fold_idx])
        train_entries = []
        val_entries = []

        for idx, fname in enumerate(filenames):
            actor_id = actor_fn(fname)
            entry = (fname, idx)
            if actor_id in holdout_actors:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

        splits.append({
            "train": train_entries,
            "val": val_entries,
            "test": val_entries,  # val == test for LOPO
        })

    return splits
