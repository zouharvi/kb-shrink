import pickle
import json
import numpy as np
from sklearn import preprocessing
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


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
        data["queries"] = data["queries"][:data["boundaries"]["train"]]
        data["relevancy"] = data["relevancy"][:data["boundaries"]["train"]]
        data["relevancy_articles"] = data["relevancy_articles"][:data["boundaries"]["train"]]
    else:
        data["queries"] = data["queries"][data["boundaries"]["train"]:data["boundaries"]["dev"]]
        data["relevancy"] = data["relevancy"][data["boundaries"]["train"]:data["boundaries"]["dev"]]
        data["relevancy_articles"] = data["relevancy_articles"][data["boundaries"]["train"]:data["boundaries"]["dev"]]
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

def zscore_data(data):
    model_d = preprocessing.StandardScaler(
        with_std=True, with_mean=True
    ).fit(data["docs"])
    model_q = preprocessing.StandardScaler(
        with_std=True, with_mean=True
    ).fit(data["queries"])
    data["docs"] = model_d.transform(data["docs"])
    data["queries"] = model_q.transform(data["queries"])
    return data