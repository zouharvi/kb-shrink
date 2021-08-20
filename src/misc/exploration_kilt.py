#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import save_pickle
from kilt.knowledge_source import KnowledgeSource
from datasets import load_dataset
import argparse
import numpy as np
from collections import defaultdict

def split_paragraph(text):
    text = text.rstrip("\n").split(" ")
    return [" ".join(text[i*100:(i+1)*100]) for i in range(0, len(text)//100+1) if i*100 != len(text)]

def split_paragraph_list(text_list):
    return [
        span for span_list
        in [
            split_paragraph(text) for text in text_list
            if not text.startswith("BULLET::::") and not text.startswith("Section::::")
        ]
        for span in span_list
    ]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', default="/data/kilt-hp/full.json")
    parser.add_argument('--data-out', default="/data/big-hp/full.pkl")
    parser.add_argument('--wiki-n', type=int, default=10000)
    parser.add_argument('--query-n', type=int, default=None)
    args = parser.parse_args()

    # get the knowledge source
    ks = KnowledgeSource(database="kilt")
    # ks = KnowledgeSource(mongo_connection_string="kiltuser@localhost", collection="knowledgesource", database="kilt-mock")

    # count entries - 5903530
    print("Loaded", ks.get_num_pages(), "pages")

    cursor = ks.get_all_pages_cursor()
    data = defaultdict(lambda: {"relevancy": [], "spans": None})

    print("Processing Wikipedia spans")
    for cur_page in cursor[:args.wiki_n]:
        data[cur_page["wikipedia_id"]]["spans"] = split_paragraph_list(cur_page["text"])
    print(np.average([len(x["spans"]) for x in data.values()]), "spans on average")

    print("Processing Dataset")
    data_hotpot = load_dataset("kilt_tasks", name="hotpotqa")["train"]

    print("Loaded", len(data_hotpot), "queries")

    data_query = []
    query_i = 0
    for sample_i, sample in enumerate(data_hotpot):
        if args.query_n is not None and sample_i >= args.query_n:
            break
        assert len(sample["output"]) == 1

        provenances = sample["output"][0]["provenance"]

        if len(provenances) == 0:
            continue

        if all([provenance["wikipedia_id"] in data.keys() for provenance in provenances]):
            for provenance in provenances:
                data[provenance["wikipedia_id"]]["relevancy"].append(query_i)
            data_query.append(sample["input"])                
            query_i += 1
        
    print("Added queries:", len(data_query))

    print("Reshaping data")
    data_docs = []
    data_relevancy = [[] for _ in data_query]


    data = list(data.items())
    while len(data):
        _, tmp = data.pop(0)
        span_texts = tmp["spans"]
        span_relevancy = tmp["relevancy"]
        for span_i, span in enumerate(span_texts):
            data_docs.append(span)
            for relevancy in span_relevancy:
                data_relevancy[relevancy].append(len(data_docs))

    print(
        "Saving",
        len(data_query), "queries,",
        len(data_docs), "docs,",
        len(data_relevancy), "relevancies",
        sum([len(x) for x in data_relevancy]), "relevancies total",
    )
    save_pickle(args.data_out, {"queries": data_query, "docs": data_docs, "relevancy": data_relevancy})