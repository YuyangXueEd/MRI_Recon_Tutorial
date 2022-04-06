import argparse
import time
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import numpy as np
import requests
import torch

import fastmri
import fastmri.data.transforms as T
from fastmri.data import SliceDataset
from unet import Unet

UNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/unet/"
MODEL_FNAMES = {
    "unet_knee_sc": "knee_sc_leaderboard_state_dict.pt",
    # "unet_knee_mc": "knee_mc_leaderboard_state_dict.pt",
    # "unet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)
    # 1MB chunks
    chunk_size = 1 * 1024 * 1024
    total_size_in_bytes = int(response.headers.get("content-length", 0))

    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)

    progress_bar.close()


def run_unet_model(batch, model, device):
    image, _, mean, std, fname, slice_num, _ = batch
    output = model(image.to(device).unsqueeze(1)).squeeze(1).cpu()

    mean = mean.unsqueeze(1).unsqueeze(2)
    std = std.unsqueeze(1).unsqueeze(2)
    output = (output * std + mean).cpu()

    return output, int(slice_num[0]), fname[0]


def run_inference(challenge, state_dict_file, data_path, output_path, device):
    """

    :param challenge: Model to run
    :param state_dict_file: Device to run
    :param data_path: Path to subsampled
    :param output_path: data Path for saving reconstructions
    :param device: Path to saved state_dict (will download if not provided)
    :return:
    """

    model = Unet(in_c=1, out_c=1, feats=256, num_pool_layers=4, drop_prob=0.0)

    if state_dict_file is None:
        if not Path(MODEL_FNAMES[challenge]).exists():
            download_model(UNET_FOLDER + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

        state_dict_file = MODEL_FNAMES[challenge]

    model.load_state_dict(torch.load(state_dict_file))
    model = model.eval()

    # dataloader setup

    if "_sc" in challenge:
        dataset = SliceDataset(
            root=data_path,
            transform=T.UnetDataTransform(which_challenge="singlecoil"),
            challenge="singlecoil"
        )
    else:
        dataset = SliceDataset(
            root=data_path,
            transform=T.UnetDataTransform(which_challenge="multicoil"),
            challenge="multicoil",
        )

    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    start_time = time.perf_counter()
    outputs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_unet_model(batch, model, device)

        outputs[fname].append((slice_num, output))

    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    
    #fastmri.save_reconstructions(outputs, output_path / "reconstructions")

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time - start_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--challenge",
                        default="unet_knee_sc",
                        choices=(
                            "unet_knee_sc",
                            "unet_knee_mc",
                            "unet_brain_mc",
                        ),
                        type=str,
                        help="Model to run\n")

    parser.add_argument("--device",
                        default="cuda",
                        type=str,
                        help="Device to run\n")

    parser.add_argument("--state_dict_file",
                        default=None,
                        type=Path,
                        help="Path to saved state_dict (will download if not provided)\n")

    parser.add_argument("--data_path",
                        type=Path,
                        required=True,
                        help="Path to subsampled data\n")

    parser.add_argument("--output_path",
                        type=Path,
                        required=True,
                        help="Path for saving reconstructions\n")

    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        torch.device(args.device),
    )

    # download_model(UNET_FOLDER + MODEL_FNAMES.get("unet_knee_sc"), "unet_knee_sc.pt")
