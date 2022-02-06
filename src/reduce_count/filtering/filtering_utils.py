from misc.retrieval_utils import retrieved_ip
from multiprocessing import Pool
import time
import numpy as np

def _update_relevancy(args):
    relevancy, offset_relevancy = args
    return [
        doc-offset_relevancy[doc]
        for doc in relevancy
    ]

def recomp_relevancy(offset_relevancy, relevancy, cur_time=time.time(), verbose=True):
    if verbose:
        print("G", f"{time.time()-cur_time:.0f}s", flush=True)
        cur_time = time.time()

    pool = Pool()
    relevancy = list(
        pool.map(
            _update_relevancy,
            zip(relevancy, [offset_relevancy] * len(relevancy))
        )
    )
    return relevancy

def prune_docs(data, data_dev, to_prune, verbose=True):
    cur_time = time.time()
    # +10 because there's a leak in relevancy
    original_data_len = len(data["docs"])+10

    offset = 0
    for doc in to_prune:
        doc -= offset
        data["docs"].pop(doc)
        offset += 1

    if verbose:
        print("E", f"{time.time()-cur_time:.0f}s", flush=True)
        cur_time = time.time()

    to_prune_local = list(to_prune)
    offset = 0
    offset_relevancy = np.zeros(original_data_len, dtype=np.int32)
    for i, _ in enumerate(offset_relevancy):
        # this may be a off-by-one error but shouldn't matter because the specific document is pruned
        if len(to_prune_local) != 0 and i == to_prune_local[0]:
            offset += 1
            to_prune_local.pop(0)
        offset_relevancy[i] = offset

    if verbose:
        print("F", f"{time.time()-cur_time:.0f}s", flush=True)
        cur_time = time.time()

    data["relevancy"] = recomp_relevancy(offset_relevancy, data["relevancy"], cur_time, verbose=verbose)
    if data_dev is not None:
        data_dev["relevancy"] = recomp_relevancy(offset_relevancy, data_dev["relevancy"], cur_time, verbose=verbose)

    if verbose:
        print("H", f"{time.time()-cur_time:.0f}s", flush=True)
        cur_time = time.time()

    return data


def filter_step(data, data_dev, cur_time=time.time()):
    print("A", f"{time.time()-cur_time:.0f}s", flush=True)

    logdata = {}
    traindata = {"negative": [], "positive": [], "neutral": []}

    logdata["docs_old"] = len(data["docs"])
    all_retrieved = retrieved_ip(
        data["queries"], data["docs"], data["relevancy"], n=20
    )
    
    print("B", f"{time.time()-cur_time:.0f}s", flush=True)
    cur_time = time.time()

    to_prune_negative = set()
    to_prune_positive = set()
    for docs_retrieved, docs_relevant in zip(all_retrieved, data["relevancy"]):
        # add docs which were previously relevant but did not help in retrieval
        to_prune_negative |= docs_retrieved - set(docs_relevant)
        to_prune_positive |= docs_retrieved & set(docs_relevant)
    to_prune = list(to_prune_negative - to_prune_positive)
    to_prune.sort()

    print("C", f"{time.time()-cur_time:.0f}s", flush=True)
    cur_time = time.time()

    logdata["to_prune"] = len(to_prune)
    logdata["positive"] = len(to_prune_positive)
    logdata["negative"] = len(to_prune_negative)

    # save training data
    for i in to_prune_negative - to_prune_positive:
        traindata["negative"].append(data["docs"][i])
    for i in to_prune_positive:
        traindata["positive"].append(data["docs"][i])
    for i in (set(range(len(data["docs"]))) - to_prune_negative) - to_prune_positive:
        traindata["neutral"].append(data["docs"][i])

    print("D", f"{time.time()-cur_time:.0f}s", flush=True)

    prune_docs(data, data_dev, to_prune)
    logdata["docs"] = len(data["docs"])

    return traindata, logdata