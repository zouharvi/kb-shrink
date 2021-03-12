#!/usr/bin/env python3

import json
from transformers import BertTokenizer, BertModel

def load_line(line):
    line = json.loads(line)
    return {"input": line["input"], "answer": [x["answer"] for x in line["output"] if "answer" in x]}

with open("data/eli5-dev-kilt.jsonl", "r") as f:
    data = [load_line(line) for line in f.readlines()]

print("About to tokenize and compute embedding of", len(data), "samples")
print("Example input: `" + data[0]["input"] + "`")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")

for i, line in enumerate(data):
    if i % 10 == 0:
        print(f'{i/len(data)*100:6.2f}%')
    print(data[i]["input"])
    encoded_input = tokenizer(data[i]["input"], return_tensors='pt') 
    data[i]["input"] = model(**encoded_input)[1]
    print(data[i]["input"].shape)