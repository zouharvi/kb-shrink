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


def parse_dataset_line(line, keep="inputs"):
    """
    @deprecated
    """
    line = json.loads(line)
    if keep == "inputs":
        return {"input": line["input"]}
    elif keep == "answers":
        return {"answer": [x["answer"] for x in line["output"] if "answer" in x]}
    elif keep == "all":
        return {"input": line["input"], "answer": [x["answer"] for x in line["output"] if "answer" in x]}
    else:
        raise Exception("Wrong `keep` argument")


def load_dataset(path, keep="inputs"):
    """
    @deprecated
    """
    with open(path, "r") as f:
        return [parse_dataset_line(line, keep) for line in f.readlines()]


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

def order_l2(data_queries, data_docs, fast):
    """
    Generator which computes ordering of neighbours. If speed=False, the results are
    guaranteed to be accurate and correct.
    """
    import faiss
    data_queries = np.array(data_queries, dtype="float32")
    data_docs = np.array(data_docs, dtype="float32")

    index = faiss.IndexFlatL2(data_docs.shape[1])
    if fast:
        nlist = 4
        index = faiss.IndexIVFFlat(index, data_docs.shape[1], nlist)
        index.train(data_docs)
    index.add(data_docs)

    def n_new_gen():
        for i, d in enumerate(data_queries):
            out = index.search(np.array([d]), len(data_docs))[1][0]
            yield out

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def order_ip(data_queries, data_docs, fast):
    """
    Generator which computes ordering of neighbours. If speed=False, the results are
    guaranteed to be accurate and correct.
    """
    import faiss
    data_queries = np.array(data_queries, dtype="float32")
    data_docs = np.array(data_docs, dtype="float32")

    index = faiss.IndexFlatIP(data_docs.shape[1])
    if fast:
        nlist = 4
        index = faiss.IndexIVFFlat(index, data_docs.shape[1], nlist)
        index.train(data_docs)
    index.add(data_docs)

    def n_new_gen():
        for i, d in enumerate(data_queries):
            out = index.search(np.array([d]), len(data_docs))[1][0]
            yield out

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def acc_l2(data_queries, data_docs, data_relevancy, n=20, fast=False, report=False):
    n_new_gen = order_l2(data_queries, data_docs, fast)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)


def acc_ip(data_queries, data_docs, data_relevancy, n=20, fast=False, report=False):
    n_new_gen = order_ip(data_queries, data_docs, fast)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)


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