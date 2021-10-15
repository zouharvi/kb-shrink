DATA = {
    "DPR\n(Avg)": """
acc_ip: 0.179
acc_l2: 0.165
acc_ip (norm): 0.180
acc_l2 (norm): 0.179
acc_ip (center): 0.196
acc_l2 (center): 0.177
acc_ip (center, norm): 0.198
acc_l2 (center, norm): 0.197
    """,
    "Sentence\nBERT\n(Avg)": """
acc_ip: 0.118
acc_l2: 0.176
acc_ip (norm): 0.123
acc_l2 (norm): 0.132
acc_ip (center): 0.123
acc_l2 (center): 0.180
acc_ip (center, norm): 0.185
acc_l2 (center, norm): 0.186
    """,
    "BERT\n(Avg)": """
acc_ip: 0.044
acc_l2: 0.053
acc_ip (norm): 0.049
acc_l2 (norm): 0.049
acc_ip (center): 0.057
acc_l2 (center): 0.063
acc_ip (center, norm): 0.073
acc_l2 (center, norm): 0.070
    """,
    "DPR\n[CLS]": """
acc_ip: 0.403
acc_l2: 0.175
acc_ip (norm): 0.271
acc_l2 (norm): 0.269
acc_ip (center): 0.414
acc_l2 (center): 0.201
acc_ip (center, norm): 0.430
acc_l2 (center, norm): 0.431
    """,
    "Sentence\nBERT\n[CLS]": """
acc_ip: 0.141
acc_l2: 0.140
acc_ip (norm): 0.142
acc_l2 (norm): 0.144
acc_ip (center): 0.139
acc_l2 (center): 0.141
acc_ip (center, norm): 0.150
acc_l2 (center, norm): 0.150
    """,
    "BERT\n[CLS]": """
acc_ip: 0.000
acc_l2: 0.003
acc_ip (norm): 0.003
acc_l2 (norm): 0.004
acc_ip (center): 0.006
acc_l2 (center): 0.007
acc_ip (center, norm): 0.010
acc_l2 (center, norm): 0.011
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
        "ip": value[4],
        "l2": value[5],
    }
    DATA_NC[name] = value[6]

