from copy import Error
import pickle
import json
import numpy as np
from scipy.spatial.distance import minkowski
import torch
import sys

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

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
    return [
        sorted(
            range(len(sims)),
            key=lambda x: sims[x],
            reverse=True
        ) for sims in vec_sim(data, sim_func)
    ]

def mrr_old(order_old, order_new, n, report=False):
    """
    Deprecated in favor of mrr_l2_fast or mrr_ip_fast but works with other datatypes
    """
    order_old = [x[:n] for x in order_old]

    def mrr_local(needles, stack):
        return 1/min([stack.index(needle)+1 for needle in needles])
    mrr_val = np.average([mrr_local(x,y) for x,y in zip(order_old, order_new)])

    if report:
        print(f"MRR (top {n}) is {mrr_val:.3f} (best is 1, worst is 0)")

    return mrr_val

def l2_sim(x, y):
    return -minkowski(x, y)

def mrr_l2_fast(data, data_new, n=20, report=False):
    from sklearn.neighbors import KDTree
    data = np.array(data)
    data_new = np.array(data_new)

    tree1 = KDTree(data, metric="l2")
    n_gold = tree1.query(data, n+1)[1]
    tree2 = KDTree(data_new, metric="l2")
    n_new = tree2.query(data_new, len(data_new))[1]
    # remove self references
    # With L2 we can be sure that `self` is the first element,
    # therefore this is faster that for IP. This is however violated
    # in case of multiple same vectors 
    n_gold = [x[x!=i] for i,x in enumerate(n_gold)]
    n_new = [x[x!=i] for i,x in enumerate(n_new)]

    return mrr_from_order(n_gold, n_new, n, report)

def mrr_ip_fast(data, data_new, n=20, report=False):
    import faiss
    data = np.array(data)
    data_new = np.array(data_new)

    index1 = faiss.IndexFlatIP(data.shape[1])
    index1.add(data)
    n_gold = index1.search(data, n+1)[1]
    index2 = faiss.IndexFlatIP(data_new.shape[1])
    index2.add(data_new)
    n_new = index2.search(data_new, len(data_new))[1]
    # remove self references
    n_gold = [x[x!=i] for i,x in enumerate(n_gold)]
    n_new = [x[x!=i] for i,x in enumerate(n_new)]

    return mrr_from_order(n_gold, n_new, n, report)

def mrr_ip_slow(data, data_new, n=20, report=False):
    n_gold = vec_sim_order(data, sim_func=np.inner)
    n_gold = [x[:n] for x in n_gold]
    n_new = vec_sim_order(data_new, sim_func=np.inner)

    return mrr_from_order(n_gold, n_new, n, report)

def mrr_from_order(n_gold, n_new, n, report=False):
    # compute mrr
    def mrr_local(needles, stack):
        mrr_candidates = 1/min([min(np.where(stack == needle))+1 for needle in needles])
        if len(mrr_candidates) == 0:
            raise Error("At least one needle is not present in the stack")
        return mrr_candidates[0]

    mrr_val = np.average([mrr_local(x,y) for x,y in zip(n_gold, n_new)])
    if report:
        print(f"MRR (top {n}) is {mrr_val:.3f} (best is 1, worst is 0)")

    return mrr_val