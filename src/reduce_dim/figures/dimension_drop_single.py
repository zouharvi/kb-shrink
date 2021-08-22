#!/usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--logfile', default="computed/tmp.log")
args = parser.parse_args()

with open(args.logfile, "r") as f:
    DATA = eval(f.read())
DIMS = list(range(768))
BASELINE = [x for x in DATA if x["dim"] == False][0]
DATA = [x for x in DATA if x["dim"] != False]
# IMPR_IP = [x["dim"] for x in sorted(DATA, key=lambda x: x["val_ip"], reverse=True) if x["val_ip"] > BASELINE["val_ip"]]
# IMPR_L2 = [x["dim"] for x in sorted(DATA, key=lambda x: x["val_l2"], reverse=True) if x["val_l2"] > BASELINE["val_l2"]]
print([x["dim"] for x in sorted(DATA, key=lambda x: x["val_ip"], reverse=True)])
print([x["dim"] for x in sorted(DATA, key=lambda x: x["val_l2"], reverse=True)])
print("improvement IP", len([x for x in DATA if x["val_ip"] >= BASELINE["val_ip"]]))
print("improvement L2", len([x for x in DATA if x["val_l2"] >= BASELINE["val_l2"]]))