from misc.retrieval_utils import retrieved_ip

def harmful_useful_pass(data):
    print(f"Running retrieval on {len(data['docs'])} docs")
    all_retrieved = retrieved_ip(
        data["queries"], data["docs"], data["relevancy"], n=10
    )

    print("Computing which documents to prune")
    to_prune_harmful = set()
    to_prune_useful = set()
    for docs_retrieved, docs_relevant in zip(all_retrieved, data["relevancy"]):
        # add docs which were previously relevant but did not help in retrieval
        to_prune_harmful |= docs_retrieved - set(docs_relevant)
        to_prune_useful |= docs_retrieved & set(docs_relevant)
    to_prune = list(to_prune_harmful - to_prune_useful)
    to_prune.sort()

    print("Harmful:", len(to_prune_harmful))
    print("Useful:", len(to_prune_useful))
    print("To-prune:", len(to_prune))

    print("Pruning documents")
    offset = 0
    for doc in to_prune:
        doc -= offset
        data["docs"].pop(doc)
        offset += 1
    print("Docs:", len(data["docs"]))

    print("Offseting relevancy")
    prev_offset = -1
    offset_relevancy = {-1: 0}
    for doc in to_prune:
        offset_relevancy[doc] = offset_relevancy[prev_offset] + 1
        prev_offset = doc

    # remove phony target
    offset_relevancy.pop(-1)
    # sort from highest to lowest doc
    offset_relevancy = sorted(
        offset_relevancy.items(),
        key=lambda x: x[0], reverse=True
    )

    def update_doc_offset(doc):
        for doc_offset, offset_val in offset_relevancy:
            if doc_offset <= doc:
                return doc - offset_val
        return doc


    data["relevancy"] = [
        [update_doc_offset(x) for x in relevancy] for relevancy in data["relevancy"]
    ]

    # # TODO: there's a faster way to do this with just a single pass
    # for offset in to_prune:
    #     data["relevancy"] = [
    #         [x - 1 if x > offset else x for x in relevancy]
    #         for relevancy in data["relevancy"]
    #     ]
    # print("Relevancy", data["relevancy"])

    return data