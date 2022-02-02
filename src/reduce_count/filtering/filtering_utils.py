from misc.retrieval_utils import retrieved_ip
import multiprocessing

# global state
# TODO: may not work as expected
offset_relevancy = None

def _update_doc_offset(doc):
    global offset_relevancy
    for doc_offset, offset_val in offset_relevancy:
        if doc_offset <= doc:
            return doc - offset_val
    return doc

def _update_relevancy(relevancy):
    return [_update_doc_offset(x) for x in relevancy]

def neg_pos_step(data):
    global offset_relevancy

    print("A", flush=True)
    logdata = {}
    traindata = {"negative": [], "positive": [], "neutral": []}

    logdata["docs_old"] = len(data["docs"])
    all_retrieved = retrieved_ip(
        data["queries"], data["docs"], data["relevancy"], n=10
    )
    print("B", flush=True)

    to_prune_negative = set()
    to_prune_positive = set()
    for docs_retrieved, docs_relevant in zip(all_retrieved, data["relevancy"]):
        # add docs which were previously relevant but did not help in retrieval
        to_prune_negative |= docs_retrieved - set(docs_relevant)
        to_prune_positive |= docs_retrieved & set(docs_relevant)
    to_prune = list(to_prune_negative - to_prune_positive)
    to_prune.sort()

    print("C", flush=True)
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

    print("D", flush=True)

    offset = 0
    for doc in to_prune:
        doc -= offset
        data["docs"].pop(doc)
        offset += 1
    print("D", flush=True)

    logdata["docs"] = len(data["docs"])

    prev_offset = -1
    offset_relevancy = {-1: 0}
    for doc in to_prune:
        offset_relevancy[doc] = offset_relevancy[prev_offset] + 1
        prev_offset = doc
    print("E", flush=True)

    # remove phony target
    offset_relevancy.pop(-1)
    # sort from highest to lowest doc
    offset_relevancy = sorted(
        offset_relevancy.items(),
        key=lambda x: x[0], reverse=True
    )
    print("F", flush=True)


    pool = multiprocessing.Pool()

    data["relevancy"] = list(
        pool.map(
            _update_relevancy,
            data["relevancy"]
        )
    )
    print("G", flush=True)

    return traindata, logdata