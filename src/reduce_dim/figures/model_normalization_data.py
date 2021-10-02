DATA = {
    "DPR\n(Avg)": """
    """,
    "Sentence\nBERT\n(Avg)": """
    """,
    "BERT\n(Avg)": """
    """,
    "DPR\n[CLS]": """
acc_ip: 0.4274
acc_l2: 0.0915
acc_ip (norm): 0.
acc_l2 (norm): 0.
acc_ip (center): 0.
acc_l2 (center): 0.
acc_ip (center, norm): 0.4295
acc_l2 (center, norm): 0.4305
    """,
    "Sentence\nBERT\n[CLS]": """
    """,
    "BERT\n[CLS]": """
    """
}

DATA_BASE = {}
DATA_N = {}
DATA_C = {}
DATA_NC = {}

for name, value in DATA.items():
    value = [float(x.split(':')[1].strip()) for x in value.split('\n')[1:-1]]
    DATA_BASE[name] = {
        "ip": value[0],
        "l2": value[1],
    }
    DATA_N[name] = value[2]
    DATA_C[name] = {
        "ip": value[3],
        "l2": value[4],
    }
    DATA_NC[name] = value[5]
