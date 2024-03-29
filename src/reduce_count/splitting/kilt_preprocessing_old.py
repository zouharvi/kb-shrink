#!/usr/bin/env python3

import sys; sys.path.append("src")
from misc.load_utils import save_pickle
from kilt.knowledge_source import KnowledgeSource
from datasets import load_dataset
import argparse
import numpy as np
from collections import defaultdict
from itertools import chain

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
    parser.add_argument('--data-out', default="/data/hp/full.pkl")
    parser.add_argument('--wiki-n', type=int, default=None)
    parser.add_argument('--query-n', type=int, default=None)
    parser.add_argument('--prune-unused', action="store_true")
    args,_ = parser.parse_known_args()

    # get the knowledge source
    ks = KnowledgeSource(database="kilt")
    # ks = KnowledgeSource(mongo_connection_string="kiltuser@localhost", collection="knowledgesource", database="kilt-mock")

    # count entries - 5903530
    print("Loaded", ks.get_num_pages(), "pages")

    cursor = ks.get_all_pages_cursor()
    data = defaultdict(lambda: {"relevancy": [], "spans": None})

    print("Processing Wikipedia spans")
    total_pages = ks.get_num_pages()
    for cur_page_i, cur_page in enumerate(cursor[:args.wiki_n]):
        if cur_page_i % 10000 == 0:
            print(f'\r{cur_page_i/total_pages*100:0.2f}%', end='')
        data[cur_page["wikipedia_id"]]["spans"] = split_paragraph_list(cur_page["text"])
    print(np.average([len(x["spans"]) for x in data.values()]), "spans on average")

    print("Processing Dataset")
    hotpot_all = load_dataset("kilt_tasks", name="hotpotqa")
    data_hotpot = chain(hotpot_all["train"], hotpot_all["validation"], hotpot_all["test"])

    print(
        len(hotpot_all["train"]), "train queries",
        len(hotpot_all["validation"]), "dev queries", 
        len(hotpot_all["test"]), "test queries",
    )

    data_query = []
    # store articles which is required for evaluation
    data_relevancy_articles = []
    query_i = 0
    query_train_max = None
    query_dev_max = None
    query_test_max = None
    for sample_i, sample in enumerate(data_hotpot):
        if args.query_n is not None and sample_i >= args.query_n:
            break

        if len(sample["output"]) == 0:
            continue

        assert len(sample["output"]) == 1

        provenances = sample["output"][0]["provenance"]

        if len(provenances) == 0:
            continue

        if all([provenance["wikipedia_id"] in data.keys() for provenance in provenances]):
            provenance_articles = set()
            for provenance in provenances:
                provenance_articles.add(int(provenance["wikipedia_id"]))
                data[provenance["wikipedia_id"]]["relevancy"].append(query_i)
            data_query.append(sample["input"])
            data_relevancy_articles.append(provenance_articles)                
            query_i += 1

            # set boundaries
            if sample_i < len(hotpot_all["train"]):
                query_train_max = query_i
            elif sample_i < len(hotpot_all["train"]) + len(hotpot_all["validation"]):
                query_dev_max = query_i
            else:
                query_test_max = query_i
        
    print(
        "Added queries:", len(data_query),
        "boundaries", {"train": query_train_max, "dev": query_dev_max, "test": query_test_max}
    )

    print("Reshaping data")
    data_docs = []
    data_docs_articles = []
    data_relevancy = [[] for _ in data_query]

    for span_article, span_obj in data.items():
        span_texts = span_obj["spans"]
        span_relevancy = span_obj["relevancy"]
        if args.prune_unused and len(span_relevancy) == 0:
            continue
        for span_i, span in enumerate(span_texts):
            data_docs.append(span)
            data_docs_articles.append(int(span_article))
            for relevancy in span_relevancy:
                data_relevancy[relevancy].append(len(data_docs))

    print(
        "Saving",
        len(data_query), "queries,",
        len(data_docs), "docs,",
        sum([len(x) for x in data_relevancy]), "relevancies total",
        np.average([len(x) for x in data_relevancy]), "relevancies average",
        sum([len(x) for x in data_relevancy_articles]), "articles total",
        np.average([len(x) for x in data_relevancy_articles]), "articles average",
    )
    save_pickle(
        args.data_out,
        {
            "queries": data_query, "docs": data_docs,
            "relevancy": data_relevancy, "relevancy_articles": data_relevancy_articles,
            "docs_articles": data_docs_articles,
            "boundaries": {"train": query_train_max, "dev": query_dev_max, "test": query_test_max}
        }
    )
    print("data_query[0]:")
    print(data_query[0])
    print("\ndata_docs[0]:")
    print(data_docs[0])
    print("\ndata_relevancy[0]:")
    print(data_relevancy[0])
    print("\ndata_relevancy_articles[0]:")
    print(data_relevancy_articles[0])
    print("\ndata_docs_articles[0]:")
    print(data_docs_articles[0])