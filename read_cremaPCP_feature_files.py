import os
import argparse

from tqdm import tqdm

import numpy as np

import deepdish as dd

import torch

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    return args


def enumerate_feature_files(feature_dir: str):
    feature_files = []
    for dirs, _, files in os.walk(feature_dir):
        for file in files:
            if file.endswith("h5"):
                feature_files.append(os.path.join(feature_dir, file))

    return feature_files


if __name__ == '__main__':
    args = get_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    feature_files = enumerate_feature_files(args.feature_dir)

    print("feature_files: {}".format(feature_files))

    data = []
    label = []

    for file in tqdm(feature_files):
        temp_crema = dd.io.load(file)["crema"]
        print(temp_crema.shape)
        idxs = np.arange(0, temp_crema.shape[0], 8)

        temp_tensor = torch.from_numpy(temp_crema[idxs].T)

        data.append(torch.cat((temp_tensor, temp_tensor))[:23].unsqueeze(0))
        label.append(os.path.splitext(os.path.basename(file))[0])

    data_for_network = {"data": data, "label": label}

    torch.save(data_for_network, os.path.join(args.save_dir, "data_for_network.pt"))
