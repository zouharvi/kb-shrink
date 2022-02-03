from misc.retrieval_utils import retrieved_ip
from multiprocessing import Pool
import time

# TODO: maybe it's faster to build a direct index->offset mapping?
def _update_doc_offset(doc, offset_relevancy):
    for doc_offset, offset_val in offset_relevancy:
        if doc_offset <= doc:
            return doc - offset_val
    return doc

def _update_relevancy(args):
    relevancy, offset_relevancy = args
    return [_update_doc_offset(x, offset_relevancy) for x in relevancy]

def recomp_relevancy(offset_relevancy, relevancy, cur_time=time.time()):
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

def prune_docs(data, data_dev, to_prune):
    cur_time = time.time()

    offset = 0
    for doc in to_prune:
        doc -= offset
        data["docs"].pop(doc)
        offset += 1

    print("E", f"{time.time()-cur_time:.0f}s", flush=True)
    cur_time = time.time()


    prev_offset = -1
    offset_relevancy = {-1: 0}
    for doc in to_prune:
        offset_relevancy[doc] = offset_relevancy[prev_offset] + 1
        prev_offset = doc
    
    print("F", f"{time.time()-cur_time:.0f}s", flush=True)
    cur_time = time.time()

    # remove phony target
    offset_relevancy.pop(-1)
    # sort from highest to lowest doc
    offset_relevancy = sorted(
        offset_relevancy.items(),
        key=lambda x: x[0], reverse=True
    )
    
    # offset_relevancy_index = [0]*len(data["docs"])
    # for i, _ in enumerate(offset_relevancy_index):
    # # this can be further sped up because the offset_relevancy is sorted 
    #     for doc_offset, offset_val in offset_relevancy:
    #         if doc_offset <= doc:
    #             offset_relevancy_index[i] = offset_val

    data["relevancy"] = recomp_relevancy(offset_relevancy, data["relevancy"], cur_time)
    data_dev["relevancy"] = recomp_relevancy(offset_relevancy, data_dev["relevancy"], cur_time)

    print("H", f"{time.time()-cur_time:.0f}s", flush=True)
    cur_time = time.time()


def filter_step(data, data_dev, cur_time=time.time()):
    print("A", f"{time.time()-cur_time:.0f}s", flush=True)

    logdata = {}
    traindata = {"negative": [], "positive": [], "neutral": []}

    logdata["docs_old"] = len(data["docs"])
    all_retrieved = retrieved_ip(
        data["queries"], data["docs"], data["relevancy"], n=10
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
    for i in to_prune_negative:
        traindata["negative"].append(data["docs"][i])
    for i in to_prune_positive:
        traindata["positive"].append(data["docs"][i])
    for i in (set(range(len(data["docs"]))) - set(to_prune_negative)) - set(to_prune_positive):
        traindata["neutral"].append(data["docs"][i])

    print("D", f"{time.time()-cur_time:.0f}s", flush=True)

    prune_docs(data, data_dev, to_prune)
    logdata["docs"] = len(data["docs"])

    return traindata, logdata