#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.utils import parse_dataset_line, DEVICE
import pickle
import argparse
from transformers import BertTokenizer, BertModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import AutoTokenizer, AutoModel
import torch

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # first element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).reshape(-1)

class BertWrap():
    def __init__(self, embd_model="bert-base-cased"):
        self.tokenizer = BertTokenizer.from_pretrained(embd_model)
        self.model = BertModel.from_pretrained(embd_model, return_dict=True, output_hidden_states=True)
        self.model.train(False)
        self.model.to(DEVICE)

    def sentence_embd(self, sentence, type_out):
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model(**encoded_input)
        if type_out == "cls":
            # each dimension is bounded [0, 1] 
            return output["pooler_output"][0].cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(output, encoded_input['attention_mask'])
            return sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

class SentenceBertWrap():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.model.train(False)
        self.model.to(DEVICE)

    def sentence_embd(self, sentence, type_out):
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model(**encoded_input)
        if type_out == "cls":
            # each dimension is bounded [0, 1] 
            return output["pooler_output"][0].cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(output, encoded_input['attention_mask'])
            return sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

class DPRWrap():
    def __init__(self):
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.model.to(DEVICE)

    def sentence_embd(self,  sentence, type_out):
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model(**encoded_input, output_hidden_states=type_out=="tokens")
        if type_out == "cls":
            return output.pooler_output[0].detach().cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(output["hidden_states"], encoded_input['attention_mask'])
            return sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="data/hotpot.jsonl")
    parser.add_argument('--embd', default="data/hotpot.embd")
    parser.add_argument('--model', default="bert")
    parser.add_argument('--type-out', default="cls")
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    if args.model == "bert":
        model = BertWrap()
    elif args.model == "sentencebert":
        model = SentenceBertWrap()
    elif args.model == "dpr":
        model = DPRWrap()
    else:
        raise Exception("Unknown model")

    with open(args.dataset, "r") as fread, open(args.embd, "wb") as fwrite:
        pickler = pickle.Pickler(fwrite)
        for i, line in enumerate(fread):
            if i >= args.n:
                break
            line = parse_dataset_line(line, keep="inputs")
            output = model.sentence_embd(line["input"], args.type_out)

            if i % 25 == 0:
                print(i, line["input"], output.shape)
            pickler.dump(output)
