#!/usr/bin/env python3

import sys
sys.path.append("src")
from reduce_count.splitting.splitter_models import get_splitter, split_paragraph_list
from misc.load_utils import save_pickle
from kilt.knowledge_source import KnowledgeSource
from datasets import load_dataset
import argparse
import numpy as np
from collections import defaultdict
from itertools import chain
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-out', default="/data/hp/full.pkl")
    parser.add_argument('--wiki-n', type=int, default=None)
    parser.add_argument('--splitter', default="fixed")
    parser.add_argument('--query-dataset', default="hotpotqa")
    parser.add_argument('--query-n', type=int, default=None)
    parser.add_argument('--no-prune-unused', action="store_true")
    parser.add_argument('--splitter-width', type=int, default=100)
    parser.add_argument('--splitter-overlap', type=int, default=0)
    args = parser.parse_args()
    splitter = get_splitter(args.splitter, args)

    # get the knowledge source
    ks = KnowledgeSource(collection="knowledgesource", database="kilt")

    # count entries - 5903530
    print("Loaded", ks.get_num_pages(), "pages")

    cursor = ks.get_all_pages_cursor()
    data = defaultdict(lambda: {"relevancy": [], "spans": None})

    print("Processing Wikipedia spans")
    # we know the total beforehand so we can tell tqdm
    for cur_page in tqdm(cursor[:args.wiki_n], total=ks.get_num_pages()):
        data[cur_page["wikipedia_id"]]["spans"] = split_paragraph_list(
            cur_page["text"],
            splitter
        )

    print(
        np.average([len(x["spans"]) for x in data.values()]),
        "spans on average"
    )

    print("Processing Dataset")
    hotpot_all = load_dataset("kilt_tasks", name=args.query_dataset)
    data_query_source = chain(
        hotpot_all["train"],
        hotpot_all["validation"],
        hotpot_all["test"]
    )

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
    for sample_i, sample in tqdm(enumerate(data_query_source)):
        # early stopping
        if args.query_n is not None and sample_i >= args.query_n:
            break

        if len(sample["output"]) == 0:
            continue

        # true only for HP
        # assert len(sample["output"]) == 1
        # unwrap and flatten all provenances
        provenances = [i for x in sample["output"] for i in x["provenance"]]

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
        "\nboundaries_train", query_train_max,
        "\nboundaries_dev", query_dev_max,
        "\nboundaries_test", query_test_max,
    )

    print("Reshaping data")
    # build data docs articles and data relevancy
    data_docs = []
    data_docs_articles = []
    data_relevancy = [[] for _ in data_query]

    for span_article, span_obj in tqdm(data.items()):
        span_texts = span_obj["spans"]
        span_relevancy = span_obj["relevancy"]
        if not args.no_prune_unused and len(span_relevancy) == 0:
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
        sum([len(x) for x in data_relevancy]),
        "relevancies total,",
        f'{np.average([len(x) for x in data_relevancy]):.2f}',
        "relevancies average,",
        sum([len(x) for x in data_relevancy_articles]),
        "articles total,",
        f'{np.average([len(x) for x in data_relevancy_articles]):.2f}',
        "articles average",
    )
    save_pickle(
        args.data_out,
        {
            "queries": data_query, "docs": data_docs,
            "relevancy": data_relevancy, "relevancy_articles": data_relevancy_articles,
            "docs_articles": data_docs_articles,
            "boundaries": {"train": query_train_max, "dev": query_dev_max, "test": query_test_max},
        }
    )