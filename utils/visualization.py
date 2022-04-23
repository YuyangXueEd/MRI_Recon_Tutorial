from pathlib import Path
import h5py

import numpy as np
import torch


def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    :param reconstructions (dict[str, np.array]): A dictionary mapping input
                                filenames to corresponding reconstructions (of shape
                                `num_slices x height x width`)
    :param out_dir (Path): Path to the output directory where the reconstructions
                               should be saved.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname, recons in reconstructions.items():
        fname = fname.split('/')[-1]
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)