import pickle
import json
import numpy as np
import torch

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def read_keys_pickle(path):
    data = []
    with open(path, "rb") as fread:
        reader = pickle.Unpickler(fread)
        while True:
            try:
                data.append(reader.load()[0])
            except EOFError:
                break
    return np.array(data)

def save_keys_pickle(data, path):
    with open(path, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        for line in data:
            pickler.dump(line)

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