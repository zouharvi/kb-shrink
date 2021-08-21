#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import save_pickle
from kilt.knowledge_source import KnowledgeSource
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki-n', type=int, default=None)
    args = parser.parse_args()

    # get the knowledge source
    ks = KnowledgeSource(database="kilt")

    # count entries - 5903530
    print("Loaded", ks.get_num_pages(), "pages")

    cursor = ks.get_all_pages_cursor()
    count_sentences = []
    count_words = []

    print("Processing Wikipedia spans")
    for cur_page in cursor[:args.wiki_n]:
        count_sentences.append(sum([x.count(".") for x in cur_page["text"]]))
        count_words.append(sum([len(x.split()) for x in cur_page["text"]]))

    print(f"Average number of sentences per article: {np.average(count_sentences):.2f}")
    print(f"Average number of words per article: {np.average(count_words):.2f}")