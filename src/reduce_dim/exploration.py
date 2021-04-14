#!/usr/bin/env python3

import sys
sys.path.append("src")
from misc.utils import read_keys_pickle, DEVICE
from KILT.kilt.knowledge_source import KnowledgeSource

# get the knowledge souce
ks = KnowledgeSource(mongo_connection_string="kiltuser@localhost", collection="kiltwiki", database="kilt")

# count entries - 5903530
ks.get_num_pages()

# get page by id
page = ks.get_page_by_id(27097632)