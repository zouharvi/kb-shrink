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
    docs_crop_max = max([max(l) for l in data["relevancy"][:n_queries]])
    docs_crop_min = min([min(l) for l in data["relevancy"][:n_queries]])
    return {
        # query embeddings
        "queries": data["queries"][:n_queries],
        # doc embeddings
        "docs": data["docs"][docs_crop_min:docs_crop_max],
        # set of relevant spans (docs) ids for every query
        "relevancy": [
            {x - docs_crop_min for x in l}
            for l in data["relevancy"][:n_queries]
        ],
        # set of relevant spans (docs) ids for every article
        "relevancy_articles": [
            {x - docs_crop_min for x in l}
            for l in data["relevancy_articles"][:n_queries]
        ],
        # map from spans (docs) to articles
        "docs_articles": data["docs_articles"][docs_crop_min:docs_crop_max],
        "boundaries": data["boundaries"],
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
        data["queries"] = data["queries"][
            data["boundaries"]["train"]:data["boundaries"]["dev"]
        ]
        data["relevancy"] = data["relevancy"][
            data["boundaries"]["train"]:data["boundaries"]["dev"]
        ]
        data["relevancy_articles"] = data["relevancy_articles"][
            data["boundaries"]["train"]:data["boundaries"]["dev"]
        ]
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


class IdentityScaler:
    """
    Mock scaler that does nothing but has identity functions in place of actual scaling.
    """

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class CenterScaler:
    def transform(self, data):
        # make sure the data is np array
        data["docs"] = np.array(data["docs"])
        data["queries"] = np.array(data["queries"])

        self.offset_docs = data["docs"].mean(axis=0)
        self.offset_queries = data["queries"].mean(axis=0)

        data["docs"] -= self.offset_docs
        data["queries"] -= self.offset_queries
        return data

    def inverse_transform(self, data):
        # make sure the data is np array
        data["docs"] = np.array(data["docs"])
        data["queries"] = np.array(data["queries"])

        data["docs"] += self.offset_docs
        data["queries"] += self.offset_queries
        return data


class NormScaler:
    def transform(self, data):
        # make sure the data is np array
        data["docs"] = np.array(data["docs"])
        data["queries"] = np.array(data["queries"])

        self.scale_docs = np.linalg.norm(
            data["docs"], axis=1
        )[:, np.newaxis]

        self.scale_queries = np.linalg.norm(
            data["queries"], axis=1
        )[:, np.newaxis]

        data["docs"] /= self.scale_docs
        data["queries"] /= self.scale_queries
        return data

    def inverse_transform(self, data):
        # make sure the data is np array
        data["docs"] = np.array(data["docs"])
        data["queries"] = np.array(data["queries"])

        data["docs"] *= self.scale_docs
        data["queries"] *= self.scale_queries
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
