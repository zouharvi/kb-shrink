DATA = {
    "DPR\n(Avg)": """
acc_ip: 0.7904
acc_l2: 0.6878
acc_ip (norm): 0.7782
acc_l2 (norm): 0.7668
acc_ip (center): 0.8674
acc_l2 (center): 0.8498
acc_ip (center, norm): 0.8748
acc_l2 (center, norm): 0.8608
    """,
    "Sentence\nBERT\n(Avg)": """
acc_ip: 0.7064
acc_l2: 0.7098
acc_ip (norm): 0.7362
acc_l2 (norm): 0.7328
acc_ip (center): 0.7360
acc_l2 (center): 0.7480
acc_ip (center, norm): 0.7534
acc_l2 (center, norm): 0.7504
    """,
    "BERT\n(Avg)": """
acc_ip: 0.5302
acc_l2: 0.5764
acc_ip (norm): 0.5934
acc_l2 (norm): 0.5902
acc_ip (center): 0.7250
acc_l2 (center): 0.6898
acc_ip (center, norm): 0.7726
acc_l2 (center, norm): 0.7636
    """,
    "DPR\n[CLS]": """
acc_ip: 0.8220
acc_l2: 0.5742
acc_ip (norm): 0.7656
acc_l2 (norm): 0.7590
acc_ip (center): 0.8680
acc_l2 (center): 0.8322
acc_ip (center, norm): 0.8666
acc_l2 (center, norm): 0.8608
    """,
    "Sentence\nBERT\n[CLS]": """
acc_ip: 0.6884
acc_l2: 0.6762
acc_ip (norm): 0.6894
acc_l2 (norm): 0.6840
acc_ip (center): 0.6798
acc_l2 (center): 0.7012
acc_ip (center, norm): 0.6966
acc_l2 (center, norm): 0.6918
    """,
    "BERT\n[CLS]": """
acc_ip: 0.0458
acc_l2: 0.1724
acc_ip (norm): 0.1782
acc_l2 (norm): 0.1794
acc_ip (center): 0.4106
acc_l2 (center): 0.2622
acc_ip (center, norm): 0.4746
acc_l2 (center, norm): 0.4672
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
