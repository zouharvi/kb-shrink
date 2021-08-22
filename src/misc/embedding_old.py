#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.load_utils import DEVICE, save_pickle, read_json
import torch
from transformers import AutoTokenizer, AutoModel
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import BertTokenizer, BertModel
import argparse

# Mean Pooling - Take attention mask into account for correct averaging


def mean_pooling(model_output, attention_mask, layer_i=0):
    # first element of model_output contains all token embeddings
    token_embeddings = model_output[layer_i]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return (sum_embeddings / sum_mask).reshape(-1)


class BertWrap():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.model = BertModel.from_pretrained(
            "bert-base-cased", return_dict=True, output_hidden_states=True)
        self.model.train(False)
        self.model.to(DEVICE)

    def query_embd(self, sentence, type_out):
        encoded_input = self.tokenizer(
            sentence, padding=True,
            truncation=True, max_length=128,
            return_tensors='pt'
        )
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model(**encoded_input)
        if type_out == "cls":
            return output[0][0, 0].cpu().numpy()
        elif type_out == "pooler":
            return output["pooler_output"][0].cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(
                output, encoded_input['attention_mask'])
            return sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

    def doc_embd(self, *args):
        # Use the same encoding for document and queries
        return self.query_embd(*args)


class SentenceBertWrap():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/bert-base-nli-cls-token")
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/bert-base-nli-cls-token")
        self.model.train(False)
        self.model.to(DEVICE)

    def query_embd(self, sentence, type_out):
        encoded_input = self.tokenizer(
            sentence, padding=True, truncation=True, max_length=128, return_tensors='pt')
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model(**encoded_input)
        if type_out == "cls":
            return output[0][0, 0].cpu().numpy()
        elif type_out == "pooler":
            return output["pooler_output"][0].cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(
                output, encoded_input['attention_mask'])
            return sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

    def doc_embd(self, *args):
        # Use the same encoding for document and queries
        return self.query_embd(*args)


class DPRWrap():
    def __init__(self):
        self.tokenizer_q = DPRQuestionEncoderTokenizer.from_pretrained(
            'facebook/dpr-question_encoder-single-nq-base'
        )
        self.model_q = DPRQuestionEncoder.from_pretrained(
            'facebook/dpr-question_encoder-single-nq-base'
        )
        self.model_q.to(DEVICE)
        self.tokenizer_d = DPRContextEncoderTokenizer.from_pretrained(
            'facebook/dpr-ctx_encoder-single-nq-base'
        )
        self.model_d = DPRContextEncoder.from_pretrained(
            'facebook/dpr-ctx_encoder-single-nq-base'
        )
        self.model_d.to(DEVICE)

    def query_embd(self,  sentence, type_out):
        encoded_input = self.tokenizer_q(
            sentence, padding=True,
            truncation=True, max_length=128,
            return_tensors='pt'
        )
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model_q(
                **encoded_input, output_hidden_states=type_out in {"tokens", "cls"})
        if type_out == "cls":
            return output["hidden_states"][-1][0, 0].cpu().numpy()
        elif type_out == "pooler":
            return output.pooler_output[0].detach().cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(
                output["hidden_states"], encoded_input['attention_mask'], layer_i=-1)
            return sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")

    def doc_embd(self,  sentence, type_out):
        encoded_input = self.tokenizer_d(
            sentence, padding=True,
            truncation=True, max_length=128,
            return_tensors='pt'
        )
        encoded_input = encoded_input.to(DEVICE)
        with torch.no_grad():
            output = self.model_d(
                **encoded_input,
                output_hidden_states=type_out in {"tokens", "cls"}
            )
        if type_out == "cls":
            return output["hidden_states"][-1][0, 0].cpu().numpy()
        elif type_out == "pooler":
            return output.pooler_output[0].detach().cpu().numpy()
        elif type_out == "tokens":
            sentence_embedding = mean_pooling(
                output["hidden_states"], encoded_input['attention_mask'], layer_i=-1)
            return sentence_embedding.cpu().numpy()
        else:
            raise Exception("Unknown type out")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', default="/data/kilt-triv/full.json")
    parser.add_argument('--data-out', default="/data/kilt-triv/full.embd")
    parser.add_argument('--model', default="bert")
    parser.add_argument('--type-out', default="cls")
    parser.add_argument('--n', type=int, default=None)
    args = parser.parse_args()

    if args.model == "bert":
        model = BertWrap()
    elif args.model in {"sentencebert", "sbert"}:
        model = SentenceBertWrap()
    elif args.model == "dpr":
        model = DPRWrap()
    else:
        raise Exception("Unknown model")

    data = read_json(args.data_in)

    data_relevancy = []
    data_queries = []
    data_docs = []
    highest_doc = 0

    # compute query embedding
    for i, sample in enumerate(data["queries"]):
        if i == args.n:
            break

        output = model.query_embd(sample["input"], args.type_out)
        data_queries.append(output)
        assert len(sample["doc_ids"]) != 0
        data_relevancy.append(sample["doc_ids"])
        highest_doc = max(max(sample["doc_ids"]), highest_doc)

        if i % 50 == 0:
            print(i, sample["input"], output.shape)

    # compute doc embedding
    for i, doc in enumerate(data["docs"]):
        if i == highest_doc+1:
            break
        output = model.doc_embd(doc, args.type_out)
        data_docs.append(output)

        if i % 50 == 0:
            print(i, doc, output.shape)

    # store data
    data_out = {"queries": data_queries,
                "docs": data_docs, "relevancy": data_relevancy}
    save_pickle(args.data_out, data_out)
