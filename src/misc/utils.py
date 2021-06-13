import pickle
import json
import numpy as np
from scipy.spatial.distance import minkowski
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0") 
else:
    DEVICE =  torch.device("cpu")


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


def center_data(data):
    data["docs"] = np.array(data["docs"])
    data["queries"] = np.array(data["queries"])
    data["docs"] = data["docs"] - data["docs"].mean(axis=0)
    data["queries"] = data["queries"] - data["queries"].mean(axis=0)
    return data

def norm_data(data):
    data["docs"] = np.array(data["docs"])
    data["queries"] = np.array(data["queries"])
    data["docs"] = data["docs"] / np.linalg.norm(data["docs"])
    data["queries"] = data["queries"] / np.linalg.norm(data["queries"])
    return data

def l2_sim(x, y):
    return -minkowski(x, y)


def order_l2_kdtree(data_queries, data_docs, fast):
    from sklearn.neighbors import KDTree
    data_queries = np.array(data_queries)
    data_docs = np.array(data_docs)

    index = KDTree(data_docs, metric="l2")

    def n_new_gen():
        for i, d in enumerate(data_queries):
            out = index.query(np.array([d]), len(data_docs))[1][0]
            yield out

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()

def order_l2(data_queries, data_docs, n, fast):
    """
    Generator which computes ordering of neighbours. If speed=False, the results are
    guaranteed to be accurate and correct.
    """
    import faiss
    data_queries = np.array(data_queries, dtype="float32")
    data_docs = np.array(data_docs, dtype="float32")

    index = faiss.IndexFlatL2(data_docs.shape[1])
    if fast:
        nlist = 200
        index = faiss.IndexIVFFlat(index, data_docs.shape[1], nlist, faiss.METRIC_L2)
        index.nprobe = 25
        index.train(data_docs)
    index.add(data_docs)

    def n_new_gen():
        for i, d in enumerate(data_queries):
            out = index.search(np.array([d]), n)[1][0]
            yield out

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def order_ip(data_queries, data_docs, n, fast):
    """
    Generator which computes ordering of neighbours. If speed=False, the results are
    guaranteed to be accurate and correct.
    """
    import faiss
    data_queries = np.array(data_queries, dtype="float32")
    data_docs = np.array(data_docs, dtype="float32")

    index = faiss.IndexFlatIP(data_docs.shape[1])
    if fast:
        nlist = 200
        index = faiss.IndexIVFFlat(index, data_docs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 25
        index.train(data_docs)
    index.add(data_docs)

    BATCH_LIMIT = 16
    def n_new_gen():
        batch = []
        for i, d in enumerate(data_queries):
            batch.append(d)
            if len(batch) >= BATCH_LIMIT or i == len(data_queries)-1:
                out = index.search(np.array(batch), n)[1]
                for el in out:
                    yield el
                batch = []

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def acc_l2(data_queries, data_docs, data_relevancy, n=20, fast=False, report=False):
    n_new_gen = order_l2(data_queries, data_docs, n, fast)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)


def acc_ip(data_queries, data_docs, data_relevancy, n=20, fast=False, report=False):
    n_new_gen = order_ip(data_queries, data_docs, n, fast)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)

def rprec_l2(data_queries, data_docs, data_relevancy, fast=False, report=False):
    n_new_gen = order_l2(data_queries, data_docs, 20, fast)
    return rprec_from_relevancy(data_relevancy, n_new_gen, report)

def rprec_ip(data_queries, data_docs, data_relevancy, fast=False, report=False):
    n_new_gen = order_ip(data_queries, data_docs, 20, fast)
    return rprec_from_relevancy(data_relevancy, n_new_gen, report)

def acc_from_relevancy(relevancy, n_new, n, report=False):
    def acc_local(doc_true, doc_hyp):
        """
        Accuracy for one query
        """
        if len(set(doc_true) & set(doc_hyp[:n])) != 0:
            return 1
        else:
            return 0

    acc_val = np.average([acc_local(x, y) for x, y in zip(relevancy, n_new)])
    if report:
        print(f"ACC (top {n}) is {acc_val:.3f} (best is 1, worst is 0)")

    return acc_val

def rprec_from_relevancy(relevancy, n_new, report=False):
    def rprec_local(doc_true, doc_hyp):
        """
        R-Precision for one query
        """
        return len(set(doc_hyp[:len(doc_true)]) & set(doc_true))/len(doc_true)

    rprec_val = np.average([rprec_local(x, y) for x, y in zip(relevancy, n_new)])
    if report:
        print(f"RPrec is {rprec_val:.3f} (best is 1, worst is 0)")

    return rprec_val