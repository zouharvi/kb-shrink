# pruned
DATA = {
    "DPR\n(Avg)": """
rprec_a_ip: 0.4081
rprec_a_l2: 0.3346
rprec_a_ip (norm): 0.4113
rprec_a_l2 (norm): 0.4161
rprec_a_ip (center): 0.5613
rprec_a_l2 (center): 0.4365
rprec_a_ip (center, norm): 0.5433
rprec_a_l2 (center, norm): 0.5375
    """,
    "Sentence\nBERT\n(Avg)": """
rprec_a_ip: 0.3075
rprec_a_l2: 0.3665
rprec_a_ip (norm): 0.3879
rprec_a_l2 (norm): 0.3880
rprec_a_ip (center): 0.3845
rprec_a_l2 (center): 0.3653
rprec_a_ip (center, norm): 0.4146
rprec_a_l2 (center, norm): 0.4110
    """,
    "BERT\n(Avg)": """
rprec_a_ip: 0.1565
rprec_a_l2: 0.2521
rprec_a_ip (norm): 0.2524
rprec_a_l2 (norm): 0.2512
rprec_a_ip (center): 0.3962
rprec_a_l2 (center): 0.3227
rprec_a_ip (center, norm): 0.4472
rprec_a_l2 (center, norm): 0.4297
    """,
    "DPR\n[CLS]": """
rprec_a_ip: 0.6090
rprec_a_l2: 0.2397
rprec_a_ip (norm): 0.4629
rprec_a_l2 (norm): 0.4629
rprec_a_ip (center): 0.6317
rprec_a_l2 (center): 0.5342
rprec_a_ip (center, norm): 0.6168
rprec_a_l2 (center, norm): 0.6146
    """,
    "Sentence\nBERT\n[CLS]": """
rprec_a_ip: 0.3245
rprec_a_l2: 0.3046
rprec_a_ip (norm): 0.3154
rprec_a_l2 (norm): 0.3140
rprec_a_ip (center): 0.3121
rprec_a_l2 (center): 0.3031
rprec_a_ip (center, norm): 0.3369
rprec_a_l2 (center, norm): 0.3337
    """,
    "BERT\n[CLS]": """
rprec_a_ip: 0.0066
rprec_a_l2: 0.0278
rprec_a_ip (norm): 0.0276
rprec_a_l2 (norm): 0.0271
rprec_a_ip (center): 0.0882
rprec_a_l2 (center): 0.0799
rprec_a_ip (center, norm): 0.1324
rprec_a_l2 (center, norm): 0.1309
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