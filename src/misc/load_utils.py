import pickle
import json
import numpy as np
from sklearn import preprocessing


def read_json(path):
    with open(path, "r") as fread:
        return json.load(fread)


def read_pickle(path):
    with open(path, "rb") as fread:
        reader = pickle.Unpickler(fread)
        return reader.load()


def save_pickle(path, data):
    with open(path, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        pickler.dump(data)


def read_keys_pickle(path):
    data = []
    with open(path, "rb") as fread:
        reader = pickle.Unpickler(fread)
        while True:
            try:
                data.append(reader.load())
            except EOFError:
                break
    return np.array(data)


def save_keys_pickle(path, data):
    with open(path, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        for line in data:
            pickler.dump(np.array(line))


def small_data(data, n_queries):
    """
    Warning: The data is only being cropped from the top and may not lead to significant reduction
    """
    docs_crop = max([max(l) for l in data["relevancy"][:n_queries]])
    return {
        "queries": data["queries"][:n_queries],
        "docs": data["docs"][:docs_crop],
        "relevancy": data["relevancy"][:n_queries],
        "relevancy_articles": data["relevancy_articles"][:n_queries],
        "docs_articles": data["docs_articles"][:docs_crop],
    }


def sub_data(data, train=False, in_place=True):
    assert in_place
    if train:
        # train queries
        data["queries"] = data["queries"][:data["boundaries"]["train"]]
        data["relevancy"] = data["relevancy"][:data["boundaries"]["train"]]
        data["relevancy_articles"] = data["relevancy_articles"][:data["boundaries"]["train"]]
    else:
        # dev queries
        data["queries"] = data["queries"][data["boundaries"]
                                          ["train"]:data["boundaries"]["dev"]]
        data["relevancy"] = data["relevancy"][data["boundaries"]
                                              ["train"]:data["boundaries"]["dev"]]
        data["relevancy_articles"] = data["relevancy_articles"][data["boundaries"]
                                                                ["train"]:data["boundaries"]["dev"]]
    return data


def center_data(data):
    data["docs"] = np.array(data["docs"])
    data["queries"] = np.array(data["queries"])
    data["docs"] -= data["docs"].mean(axis=0)
    data["queries"] -= data["queries"].mean(axis=0)
    return data


def norm_data(data):
    data["docs"] = np.array(data["docs"])
    data["queries"] = np.array(data["queries"])
    data["docs"] /= np.linalg.norm(data["docs"], axis=1)[:, np.newaxis]
    data["queries"] /= np.linalg.norm(data["queries"], axis=1)[:, np.newaxis]
    return data


def zscore_data(data, center=True):
    preprocessing.StandardScaler(
        copy=False, with_std=True, with_mean=center
    ).fit_transform(data["docs"])
    preprocessing.StandardScaler(
        copy=False, with_std=True, with_mean=center
    ).fit_transform(data["queries"])
    return data

def process_dims(dims):
    if dims == "custom":
        return [32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768]
    elif dims == "linspace":
        return np.linspace(32, 768, num=768 // 32, endpoint=True)
    elif dims.isdigit():
        return [int(dims)]
    else:
        raise Exception(f"Unknown --dims {dims} scheme")