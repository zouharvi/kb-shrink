#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np

with open(sys.argv[1], 'r') as f:
    data = [line.strip().replace(",", "").split(" ") for line in f.readlines()]
    data_loss = [float(x[3]) for x in data]
    data_mrr_ip = [float(x[5]) for x in data]
    data_mrr_l2 = [float(x[7]) for x in data]


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax3 = ax2

line1, = ax1.plot(
    # range(10, 1010, 10),
    data_loss,
    color="tab:blue",
)
line2, = ax2.plot(
    # range(10, 1010, 10),
    data_mrr_ip,
    color="tab:orange",
)
line3, = ax3.plot(
    # range(10, 1010, 10),
    data_mrr_l2,
    color="tab:green",
)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("L2 Loss")
ax2.set_ylabel("MRR (top 20 relevant)")

print(f"Loss MRR_IP correlation {np.corrcoef([data_loss, data_mrr_ip])[0,1]:.2f}")
print(f"Loss MRR_L2 correlation {np.corrcoef([data_loss, data_mrr_l2])[0,1]:.2f}")

plt.title("Autoencoder reconstruction loss vs. neighbour ordering")
plt.legend(
    [line1, line2, line3],
    ["Train L2 Loss", "Train MRR (inner product)", "Train MRR (L2)"],
    loc="upper right"
)
plt.show()
