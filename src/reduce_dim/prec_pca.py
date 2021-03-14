#!/usr/bin/env python3

import sys, os
sys.path.append("src")
from misc.utils import read_keys_pickle
import pickle
import argparse
import random
from sklearn.decomposition import PCA
from scipy.spatial.distance import minkowski
import numpy as np
import matplotlib.pyplot as plt
from pympler.asizeof import asizeof

parser = argparse.ArgumentParser(description='Explore vector distribution')
parser.add_argument(
    '--keys-in', default="data/eli5-dev.embd",
    help='Input keys')
args = parser.parse_args()
data = read_keys_pickle(args.keys_in)
origSize = asizeof(data)

print(f"{'Method':<30}   {'Size':>4}            {'Loss':>8}")

def summary_performance(name, dataReduced, dataReconstructed):
    size = asizeof(dataReduced)
    distances = []
    for vec, vecNew in zip(data, dataReconstructed):
        distances.append(minkowski(vec, vecNew, 2))
    print(f"{name:<30} {size/1024/1024:>4.1f}MB ({size/origSize:.1f}x)   {np.average(distances):>10.7f}")

def pca_performance(components):
    model = PCA(n_components=components).fit(data)
    dataReduced = model.transform(data)
    summary_performance(f"PCA ({components})", dataReduced, model.inverse_transform(dataReduced))

def precision_performance(newType):
    dataReduced = data.astype(newType)
    summary_performance(f"Precision ({newType})", dataReduced, dataReduced.astype("float32"))

def precision_pca_performance(components, newType):
    dataReduced = data.astype(newType)
    model = PCA(n_components=components).fit(dataReduced)
    dataReduced = model.transform(dataReduced).astype("float32")
    summary_performance(f"Precision ({newType}), PCA ({components})", dataReduced, model.inverse_transform(dataReduced).astype("float32"))

def pca_precision_preformance(components, newType):
    model = PCA(n_components=components).fit(data)
    dataReduced = model.transform(data)
    dataReduced = dataReduced.astype(newType)
    summary_performance(f"PCA ({components}), Precision ({newType})", dataReduced, model.inverse_transform(dataReduced.astype("float32")))

summary_performance(f"Original ({data.dtype})", data, data)
pca_performance(512)
pca_performance(256)
precision_performance("float16")
precision_pca_performance(512, "float16")
pca_precision_preformance(512, "float16")
pca_precision_preformance(256, "float16")