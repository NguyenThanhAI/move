import os
import argparse
import tempfile

from tqdm import tqdm

import numpy as np

import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--infor_preserve_ratio", type=float, default=0.9)

    args = parser.parse_args()

    return args


def enumerate_feature_file(feature_dir: str):
    feature_files = []
    for dirs, _, files in os.walk(feature_dir):
        for file in files:
            if file.endswith(".npy"):
                feature_files.append(os.path.join(dirs, file))

    return feature_files


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    feature_files = enumerate_feature_file(args.feature_dir)  # Enumerate feature files

    num_mean_vector_dict = {}

    print("Calculate mean vector")

    for file in tqdm(feature_files):
        feature_vectors = np.load(file)

        num_vectors = feature_vectors.shape[0]

        mean_vector = np.mean(feature_vectors, axis=0)

        num_mean_vector_dict[os.path.splitext(os.path.basename(file))[0]] = (mean_vector, num_vectors)

    global_mean_vector = np.zeros_like(mean_vector, dtype=np.float32)
    global_num_vector = 0

    for key in num_mean_vector_dict.keys():
        mean_vector, num_vectors = num_mean_vector_dict[key]
        global_mean_vector += mean_vector * num_vectors
        global_num_vector += num_vectors

    global_mean_vector = global_mean_vector / global_num_vector

    covariance_matrix = np.zeros(shape=[global_mean_vector.shape[0], global_mean_vector.shape[0]], dtype=np.float32)

    print("Calculate covariance matrix")

    for file in tqdm(feature_files):
        feature_vectors = np.load(file)

        diff_feature_vectors = feature_vectors - global_mean_vector[np.newaxis, :]

        covariance_matrix += diff_feature_vectors.T.dot(diff_feature_vectors)

    covariance_matrix /= (global_num_vector - 1)

    eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

    sum_eig_values = np.sum(np.abs(eig_values))

    infor_preserve = 0.

    chosen_eig_vectors = []

    eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    for i, eig_pair in enumerate(eig_pairs):
        eig_value, eig_vector = eig_pair
        infor_preserve += eig_value / sum_eig_values
        chosen_eig_vectors.append(eig_vector)
        if infor_preserve > args.infor_preserve_ratio:
            print("Choose {} axes, Infor preserved: {}".format(len(chosen_eig_vectors), infor_preserve))
            break

    W_matrix = np.stack(chosen_eig_vectors, axis=1)

    print("Calculate PCA feature vectors")

    for file in tqdm(feature_files):
        feature_vectors = np.load(file)

        pca_feature_vectors = feature_vectors.dot(W_matrix)

        np.save(os.path.join(args.save_dir, "pca_" + os.path.basename(file)), pca_feature_vectors)
        print(pca_feature_vectors)
