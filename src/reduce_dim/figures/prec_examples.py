#!/usr/bin/env python3

import sys
sys.path.append("src")
import misc.plot_utils
from reduce_dim.precision.model import _transform_to_1, _transform_to_8, _transform_to_16
import numpy as np
import matplotlib.pyplot as plt

arr32 = np.random.RandomState(3).random(8) * 2 - 1
arr16 = _transform_to_16(arr32)
arr8 = _transform_to_8(arr32)
arr1_05 = _transform_to_1(arr32, offset=-0.5)
arr1_00 = _transform_to_1(arr32, offset=0)
arr_tex = np.array([arr32, arr16, arr8, arr1_05, arr1_00]).T
print(arr_tex.shape)
for line in arr_tex:
    print(" & ".join([
        str(x) if x < 0 or xi == 4 else "\\hspace{1.3mm}" + str(x)
        for xi, x in enumerate(line)
    ]) + " \\\\")
exit()
random_generator = np.random.RandomState(3)
data = [[], [], [], [], []]

for i in range(100):
    arr32 = random_generator.random(2) * 2 - 1
    arr16 = _transform_to_16(arr32)
    arr8 = _transform_to_8(arr32)
    arr1_05 = _transform_to_1(arr32, offset=-0.5)
    arr1_00 = _transform_to_1(arr32, offset=0)

    data[0].append(arr32)
    data[1].append(arr16)
    data[2].append(arr8)
    data[3].append(arr1_05)
    data[4].append(arr1_00)

STYLE = {"alpha": 0.5, "s": 30, "marker": "s"}

plt.figure(figsize=(6, 4))

plt.scatter(
    [x[0] for x in data[0]],
    [x[1] for x in data[0]],
    label="32-bit",
    **STYLE,
)
plt.scatter(
    [x[0] for x in data[1]],
    [x[1] for x in data[1]],
    label="16-bit",
    **STYLE,
)
plt.scatter(
    [x[0] for x in data[2]],
    [x[1] for x in data[2]],
    label="8-bit",
    **STYLE,
)
plt.scatter(
    [x[0] for x in data[3]][10:30],
    [x[1] for x in data[3]][10:30],
    label="1-bit (0.5)",
    **STYLE,
)
plt.scatter(
    [x[0] for x in data[4]][10:30],
    [x[1] for x in data[4]][10:30],
    label="1-bit (0)",
    **STYLE,
)
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout()
plt.savefig("figures/prec_examples.pdf")
plt.show()
