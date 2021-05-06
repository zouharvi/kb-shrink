from copy import Error
import pickle
import json
import numpy as np
from scipy.spatial.distance import minkowski
import torch
import sys

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def read_pickle(path):
    with open(path, "rb") as fread:
        reader = pickle.Unpickler(fread)
        return reader.load()

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

def save_keys_pickle(data, path):
    with open(path, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        for line in data:
            pickler.dump(np.array(line))

def parse_dataset_line(line, keep="inputs"):
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
    with open(path, "r") as f:
        return [parse_dataset_line(line, keep) for line in f.readlines()]

def vec_sim(data, sim_func=np.inner):
    """
    Compute vector similarity matrix, excluding the diagonal.
    """
    return [
        [sim_func(vec1, vec2) for i2, vec2 in enumerate(data) if i1 != i2]
        for i1, vec1 in enumerate(data)
    ]

def vec_sim_order(data, sim_func=np.inner):
    return np.array([
        np.array(sorted(
            range(len(sims)),
            key=lambda x: sims[x],
            reverse=True
        )) for sims in vec_sim(data, sim_func)
    ])

def acc_old(order_old, order_new, n, report=False):
    """
    Deprecated in favor of acc_l2_fast or acc_ip_fast but works with other datatypes
    """
    order_old = [x[:n] for x in order_old]

    def acc_local(needles, stack):
        return 1/min([stack.index(needle)+1 for needle in needles])
    acc_val = np.average([acc_local(x,y) for x,y in zip(order_old, order_new)])

    if report:
        print(f"ACC (top {n}) is {acc_val:.3f} (best is 1, worst is 0)")

    return acc_val

def l2_sim(x, y):
    return -minkowski(x, y)

def order_l2_fast(data_queries, data_docs, n=20):
    from sklearn.neighbors import KDTree
    data_queries = np.array(data_queries)
    data_docs = np.array(data_docs)

    index = KDTree(data_docs, metric="l2")

    def n_new_gen():
        for i,d in enumerate(data_queries):
            out = index.query(np.array([d]), len(data_docs))[1][0]
            yield out

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()

def order_ip_fast(data_queries, data_docs, n=20):
    import faiss
    data_queries = np.array(data_queries, dtype="float32")
    data_docs = np.array(data_docs, dtype="float32")

    index = faiss.IndexFlatIP(data_docs.shape[1])
    index.add(data_docs)

    def n_new_gen():
        for i,d in enumerate(data_queries):
            out = index.search(np.array([d]), len(data_docs))[1][0]
            yield out

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()

def acc_l2_fast(data_queries, data_docs, data_relevancy, n=20, report=False):
    n_new_gen = order_l2_fast(data_queries, data_docs, n)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)

def acc_ip_fast(data_queries, data_docs, data_relevancy, n=20, report=False):
    n_new_gen = order_ip_fast(data_queries, data_docs, n)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)

def acc_from_relevancy(relevancy,  n_new, n, report=False):
    def acc_local(doc_true, doc_hyp):
        """
        Accuracy for one query
        """
        if len(set(doc_true) & set(doc_hyp[:n])) != 0:
            return 1
        else:
            return 0

    acc_val = np.average([acc_local(x,y) for x,y in zip(relevancy, n_new)])
    if report:
        print(f"ACC (top {n}) is {acc_val:.3f} (best is 1, worst is 0)")

    return acc_val