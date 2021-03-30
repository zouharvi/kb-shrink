#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import parse_dataset_line, DEVICE
import pickle
import argparse
from transformers import BertTokenizer, BertModel

class BertWrap():
    def __init__(self, embd_model="bert-base-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(embd_model)
        self.model = BertModel.from_pretrained(embd_model, return_dict=True, output_hidden_states=True)
        self.model.train(False)

    def sentence_embd(self, sentence, type_out):
        encoded_input = self.tokenizer(sentence, return_tensors='pt')
        encoded_input = encoded_input
        output = self.model(**encoded_input)
        if type_out == "pooler":
            return output["pooler_output"][0].detach().numpy()
        elif type_out == "tokens_avg":
            # select the last layer
            return output["hidden_states"][-1][0].mean(dim=0).detach().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="data/eli5-dev.jsonl")
    parser.add_argument('--embd-out', default="data/eli5-dev.embd")
    parser.add_argument('--embd-model', default="bert-base-cased")
    parser.add_argument('--type-out', default="pooler")
    args = parser.parse_args()
    bert = BertWrap(args.embd_model)

    with open(args.dataset, "r") as fread, open(args.embd_out, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        for i, line in enumerate(fread):
            line = parse_dataset_line(line)
            output = bert.sentence_embd(line["input"], args.type_out)

            if i % 10 == 0:
                print(i, line["input"], output.shape)
            pickler.dump(output)
