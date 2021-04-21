#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import parse_dataset_line, DEVICE
import pickle
import argparse
from transformers import BertTokenizer, BertModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch

class BertWrap():
    def __init__(self, embd_model="bert-base-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(embd_model)
        self.model = BertModel.from_pretrained(embd_model, return_dict=True, output_hidden_states=True)
        self.model.train(False)
        self.model.to(DEVICE)

    def sentence_embd(self, sentence, type_out):
        encoded_input = self.tokenizer(sentence, return_tensors='pt')
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model(**encoded_input)
        if type_out == "pooler":
            # each dimension is bounded [0, 1] 
            return output["pooler_output"][0].cpu().numpy()
        elif type_out == "tokens_avg":
            # select the last layer
            # dimensions are *not* bounded [0, 1]
            return output["hidden_states"][-1][0].mean(dim=0).cpu().numpy()
        else:
            raise Exception("Unknown type out")

class DPRWrap():
    def __init__(self):
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.model.to(DEVICE)

    def sentence_embd(self,  sentence, type_out):
        input_ids = self.tokenizer(sentence, return_tensors='pt')["input_ids"].to(DEVICE)
        with torch.no_grad():
            output = self.model(input_ids, output_hidden_states=type_out=="tokens_avg")
        if type_out == "pooler":
            return output.pooler_output[0].detach().cpu().numpy()
        elif type_out == "tokens_avg":
            return output.hidden_states[-1][0].mean(dim=0).cpu().numpy()
        else:
            raise Exception("Unknown type out")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="data/eli5-dev.jsonl")
    parser.add_argument('--embd-out', default="data/eli5-dev.embd")
    parser.add_argument('--model', default="bert")
    parser.add_argument('--type-out', default="pooler")
    args = parser.parse_args()

    if args.model == "bert":
        model = BertWrap()
    elif args.model == "dpr":
        model = DPRWrap()
    else:
        raise Exception("Unknown model")

    with open(args.dataset, "r") as fread, open(args.embd_out, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        for i, line in enumerate(fread):
            line = parse_dataset_line(line, keep="inputs")
            output = model.sentence_embd(line["input"], args.type_out)

            if i % 10 == 0:
                print(i, line["input"], output.shape)
            pickler.dump(output)
