import os
import argparse

import numpy as np

from sklearn.decomposition import PCA


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Thanh_Tuyet\Downloads\feature_dir")
    parser.add_argument("--num_segments", type=int, default=5)
    parser.add_argument("--num_vectors", type=int, default=1000)
    parser.add_argument("--num_dim", type=int, default=512)
    parser.add_argument("--infor_preserve_ratio", type=float, default=0.9)

    args = parser.parse_args()

    return args


def pca(feature_vector: np.ndarray, infor_preserve=0.9):
    assert len(feature_vector.shape) == 2
    mean_vector = np.mean(feature_vector, axis=0)
    diff_vector = feature_vector - mean_vector[np.newaxis, :]
    cov_matrix = diff_vector.T.dot(diff_vector) / (feature_vector.shape[0] - 1)
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    chosen_eig_vectors = []
    cumulative_infor_preserve = 0.

    sum_eig_values = np.sum(np.abs(eig_values))
    for i, pair in enumerate(eig_pairs):
        eig_value, eig_vector = pair
        cumulative_infor_preserve += eig_value / sum_eig_values
        chosen_eig_vectors.append(eig_vector)
        if cumulative_infor_preserve > infor_preserve:
            print("Choose {} axes, Infor preserved: {}".format(len(chosen_eig_vectors), cumulative_infor_preserve))
            break

    W_matrix = np.stack(chosen_eig_vectors, axis=1)

    pca_feature_vectors = feature_vector.dot(W_matrix)

    return pca_feature_vectors


if __name__ == '__main__':
    np.random.seed(1000)

    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    random_feature_vectors = np.random.rand(args.num_vectors, args.num_dim)

    pca_feature = pca(feature_vector=random_feature_vectors, infor_preserve=args.infor_preserve_ratio)

    print("pca feature: {}".format(pca_feature))

    splits = np.split(random_feature_vectors, args.num_segments)
    for i, arr in enumerate(splits):
        np.save(os.path.join(args.save_dir, "split_{}.npy".format(i + 1)), arr)

    #reducer = PCA(n_components=338)
    #reducer.fit(random_feature_vectors)
    #transform_feature = reducer.transform(random_feature_vectors)
    #print("pca feature: {}".format(transform_feature))
