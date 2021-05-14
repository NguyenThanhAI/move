import os
import argparse

import time

from tqdm import tqdm

import numpy as np

import deepdish as dd
from joblib import Parallel, delayed
from progress.bar import Bar

import acoss

#from acoss import extractors
#from acoss.extractors import AudioFeatures
from features import AudioFeatures


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio_dir", type=str, default=None, required=True)
    parser.add_argument("--feature_dir", type=str, default=None, required=True)
    parser.add_argument("--audio_format", type=str, default="mp3")

    args = parser.parse_args()

    return  args


def enumerate_audio_file(audio_dir: str, audio_format="mp3"):
    audio_files = []
    for dirs, _, files in os.walk(audio_dir):
        for file in files:
            #print("file: {}".format(file))
            if file.endswith(audio_format):
                audio_files.append(os.path.join(dirs, file))

    audio_files.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

    return audio_files


def enumerate_h5_file(feature_dir: str):
    file_list = []
    for dirs, _, files in os.walk(feature_dir):
        for file in files:
            if file.endswith(".h5"):
                file_list.append(os.path.join(dirs, file))

    file_list.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])

    return file_list


def compute_features(audio_path: str, params: dict, feature: AudioFeatures):
    #start = time.time()
    #feature = AudioFeatures(audio_file=audio_path, sample_rate=params["sample_rate"])
    feature.read_audio(audio_path)
    if feature.audio_vector.shape[0] == 0:
        raise IOError("Empty or invalid audio recording file -%s-" % audio_path)
    #end = time.time()
    #print("1: {}".format(end - start), end=" ")
    #start = time.time()
    if params["endtime"]:
        feature.audio_vector = feature.audio_slicer(endTime=params["endtime"])
    if params["downsample_audio"]:
        feature.audio_vector = feature.resample_audio(params["sample_rate"] / params["downsample_factor"])
    #end = time.time()
    #print("2: {}".format(end - start), end=" ")
    #start = time.time()
    out_dict = dict()
    # now we compute all the listed features in the profile dict and store the results to a output dictionary
    for method in params["features"]:
        assert method == "crema"
        out_dict[method] = getattr(feature, method)()
    #end = time.time()
    #print("3: {}".format(end - start), end=" ")

    #start = time.time()
    track_id = os.path.basename(audio_path).replace(params["input_audio_format"], "")
    out_dict["track_id"] = track_id

    #end = time.time()
    #print("4: {}".format(end - start))

    return out_dict


def compute_features_from_list_file(audio_dir: str, file_list: list, feature_dir: str, params: dict, feature: AudioFeatures):
    start_time = time.time()

    print("Length of file list before filtering: {}".format(len(file_list)))
    file_list = list(filter(lambda x: os.path.exists(x), file_list))
    print("Length of file list after filtering: {}".format(len(file_list)))

    assert len(file_list) > 0

    h5_list = enumerate_h5_file(feature_dir)
    h5_list = list(map(lambda x: os.path.splitext(os.path.basename(x))[0], h5_list))

    print("Number of extracted h5 file: {}".format(len(h5_list)))

    file_list = list(filter(lambda x: os.path.splitext(os.path.basename(x))[0] not in h5_list, file_list))

    print("Number of file list after filtering file been extracted cremaPCP: {}".format(len(file_list)))

    #progress_bar = Bar("acoss.extractor.compute_features_from_list_file",
    #                   max=len(file_list),
    #                   suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

    for song in tqdm(file_list):
        try:
            feature_dict = compute_features(audio_path=song, params=params, feature=feature)
            rel_path = os.path.relpath(song, audio_dir)
            rel_dir = os.path.dirname(rel_path)
            save_dir = os.path.join(feature_dir, rel_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            dd.io.save(os.path.join(save_dir, os.path.splitext(os.path.basename(song))[0] + ".h5"), feature_dict)
        except Exception as e:
            print("Error {} for computing features for audio file {}".format(e, song))
            continue
        #progress_bar.next()
    #progress_bar.finish()
    print("Process finished in {} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.feature_dir):
        os.makedirs(args.feature_dir, exist_ok=True)

    params = {"sample_rate": 44100,
              "input_audio_format": "." + args.audio_format,
              "downsample_audio": False,
              "downsample_factor": 2,
              "endtime": None,
              "features": ["crema"]}

    feature = AudioFeatures(sample_rate=params["sample_rate"])

    audio_files_list = enumerate_audio_file(audio_dir=args.audio_dir, audio_format=args.audio_format)

    compute_features_from_list_file(audio_dir=args.audio_dir, file_list=audio_files_list, feature_dir=args.feature_dir, params=params, feature=feature)
    #Parallel(n_jobs=4, verbose=1)(delayed(compute_features_from_list_file)(args.audio_dir, args.audio_format, params))
