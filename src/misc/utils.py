import pickle
import json
import numpy as np
from scipy.spatial.distance import minkowski
import torch

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

def mrr(order_old, order_new, n, report=False):
    order_old = [x[:n] for x in order_old]

    def mrr_local(needles, stack):
        return 1/min([stack.index(needle)+1 for needle in needles])
    mrr_val = np.average([mrr_local(x,y) for x,y in zip(order_old, order_new)])

    if report:
        print(f"MRR (top {n}) is {mrr_val:.3f} (best is 1, worst is 0)")

    return mrr_val

def l2_sim(x, y):
    return -minkowski(x, y)