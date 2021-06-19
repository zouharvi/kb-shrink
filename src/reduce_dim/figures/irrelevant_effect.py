#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())
THRESHOLDS = [x["threshold"] for x in DATA]
VALS_IP = [x["val_ip"] for x in DATA]
VALS_L2 = [x["val_l2"] for x in DATA]

plt.plot(THRESHOLDS, VALS_IP, label="IP", color="tab:blue")
plt.plot(THRESHOLDS, VALS_L2, label="L2", color="tab:red")
plt.legend()
plt.title("Effect of irrelevant documents on retrieval")
plt.ylabel("RPrec")
plt.xlabel("Documents available")
plt.show()