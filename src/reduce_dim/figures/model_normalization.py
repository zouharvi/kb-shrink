#!/usr/bin/env python3

import sys
sys.path.append("src")
import numpy as np
import matplotlib.pyplot as plt

# Hotpot
# DATA_BASE = {
#     "DPR\n(Avg)": {
#         "ip": 0.3160,
#         "l2": 0.2420,
#     },
#     "Sentence\nBERT\n(Avg)": {
#         "ip": 0.2586,
#         "l2": 0.2894,
#     },
#     "BERT\n(Avg)": {
#         "ip": 0.1451,
#         "l2": 0.1971,
#     },
#     "DPR\n[CLS]": {
#         "ip": 0.4029,
#         "l2": 0.1745,
#     },
#     "Sentence\nBERT\n[CLS]": {
#         "ip": 0.2599,
#         "l2": 0.2609,
#     },
#     "BERT\n[CLS]": {
#         "ip": 0.0044,
#         "l2": 0.0293,
#     },
# }

# DATA_N = {
#     "DPR\n(Avg)": 0.3171,
#     "Sentence\nBERT\n(Avg)": 0.3028,
#     "BERT\n(Avg)": 0.2024,
#     "DPR\n[CLS]": 0.3229,
#     "Sentence\nBERT\n[CLS]": 0.2647,
#     "BERT\n[CLS]": 0.0318,
# }

# DATA_C = {
#     "DPR\n(Avg)": {
#         "ip": 0.4111,
#         "l2": 0.3938,
#     },
#     "Sentence\nBERT\n(Avg)": {
#         "ip": 0.2917,
#         "l2": 0.3066,
#     },
#     "BERT\n(Avg)": {
#         "ip": 0.2548,
#         "l2": 0.2523,
#     },
#     "DPR\n[CLS]": {
#         "ip": 0.4482,
#         "l2": 0.4139,
#     },
#     "Sentence\nBERT\n[CLS]": {
#         "ip": 0.2533,
#         "l2": 0.2663,
#     },
#     "BERT\n[CLS]": {
#         "ip": 0.0947,
#         "l2": 0.0554,
#     },
# }

# DATA_NC = {
#     "DPR\n(Avg)": 0.4243,
#     "Sentence\nBERT\n(Avg)": 0.3132,
#     "BERT\n(Avg)": 0.3134,
#     "DPR\n[CLS]": 0.4533,
#     "Sentence\nBERT\n[CLS]": 0.2687,
#     "BERT\n[CLS]": 0.1244,
# }

# NaturalQuestions
DATA_BASE = {
    "DPR\n(Avg)": {
        "ip": 0.2608,
        "l2": 0.2644,
    },
    "Sentence\nBERT\n(Avg)": {
        "ip": 0.1050,
        "l2": 0.1513,
    },
    "BERT\n(Avg)": {
        "ip": 0.0192,
        "l2": 0.0507,
    },
    "DPR\n[CLS]": {
        "ip": 0.3333,
        "l2": 0.2755,
    },
    "Sentence\nBERT\n[CLS]": {
        "ip": 0.0043,
        "l2": 0.0316,
    },
    "BERT\n[CLS]": {
        "ip": 0.0004,
        "l2": 0.0013,
    },
}

DATA_N = {
    "DPR\n(Avg)": 0.2894,
    "Sentence\nBERT\n(Avg)": 0.1496,
    "BERT\n(Avg)": 0.0496,
    "DPR\n[CLS]": 0.3306,
    "Sentence\nBERT\n[CLS]": 0.0339,
    "BERT\n[CLS]": 0.0015,
}

DATA_C = {
    "DPR\n(Avg)": {
        "ip": 0.2894,
        "l2": 0.2856,
    },
    "Sentence\nBERT\n(Avg)": {
        "ip": 0.1241,
        "l2": 0.1339,
    },
    "BERT\n(Avg)": {
        "ip": 0.0650,
        "l2": 0.0537,
    },
    "DPR\n[CLS]": {
        "ip": 0.3396,
        "l2": 0.3269,
    },
    "Sentence\nBERT\n[CLS]": {
        "ip": 0.0143,
        "l2": 0.0354,
    },
    "BERT\n[CLS]": {
        "ip": 0.0005,
        "l2": 0.0019,
    },
}

DATA_NC = {
    "DPR\n(Avg)": 0.2940,
    "Sentence\nBERT\n(Avg)": 0.1326,
    "BERT\n(Avg)": 0.0785,
    "DPR\n[CLS]": 0.3448,
    "Sentence\nBERT\n[CLS]": 0.0383,
    "BERT\n[CLS]": 0.0034,
}

POS = np.arange(len(DATA_BASE))
BARHEIGHT = 0.15
BARSPACE = 0.15
YERRMOCK = np.random.rand(len(DATA_BASE))/10000

plt.figure(figsize=(4.4, 4))
ax = plt.gca()
ax.bar(
    POS + BARSPACE * 0,
    [x["ip"] for x in DATA_BASE.values()],
    width=BARHEIGHT, label="IP",
    color="tab:blue", hatch="",
)
ax.bar(
    POS + BARSPACE * 2,
    [x["ip"] for x in DATA_C.values()],
    width=BARHEIGHT, label="IP (center)",
    color="tab:blue", hatch="\\\\\\",
)
ax.bar(
    POS + BARSPACE * 4,
    [x for x in DATA_N.values()],
    width=BARHEIGHT, label="IP, L2 (norm)",
    color="silver", hatch="...",
)
ax.bar(
    POS + BARSPACE * 1,
    [x["l2"] for x in DATA_BASE.values()],
    width=BARHEIGHT, label="L2",
    color="tab:red", hatch="",
)
ax.bar(
    POS + BARSPACE * 3,
    [x["l2"] for x in DATA_C.values()],
    width=BARHEIGHT, label="L2 (center)",
    color="tab:red", hatch="\\\\\\",
)
ax.bar(
    POS + BARSPACE * 5,
    [x for x in DATA_NC.values()],
    width=BARHEIGHT, label="IP, L2 (center, norm)",
    color="dimgray", hatch="...\\\\\\",
)
ax.set_xticks(POS + BARSPACE * 2.5)
ax.set_xticklabels(DATA_BASE.keys())
ax.set_ylabel("R-Precision")
plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
plt.tight_layout()
plt.show()
