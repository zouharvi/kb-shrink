import numpy as np
import torch
from .retrieval_metric_utils import *

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")


def order_l2_kdtree(data_queries, data_docs, fast):
    from sklearn.neighbors import KDTree
    data_queries = np.array(data_queries)
    data_docs = np.array(data_docs)

    index = KDTree(data_docs, metric="l2")

    def n_new_gen():
        for i, d in enumerate(data_queries):
            out = index.query(np.array([d]), len(data_docs))[1][0]
            yield out

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def order_l2(data_queries, data_docs, retrieve_counts, fast):
    """
    Generator which computes ordering of neighbours. If speed=False, the results are
    guaranteed to be accurate and correct.
    """
    import faiss
    data_queries = np.array(data_queries, dtype="float32")
    data_docs = np.array(data_docs, dtype="float32")

    index = faiss.IndexFlatL2(data_docs.shape[1])
    if fast:
        nlist = 200
        index = faiss.IndexIVFFlat(
            index, data_docs.shape[1], nlist, faiss.METRIC_L2)
        index.nprobe = 25
        index.train(data_docs)
    index.add(data_docs)

    BATCH_LIMIT = 256

    def n_new_gen():
        batch = []
        batch_n = []
        for i, (d, n) in enumerate(zip(data_queries, retrieve_counts)):
            batch.append(d)
            batch_n.append(n)
            if len(batch) >= BATCH_LIMIT or i == len(data_queries) - 1:
                out = index.search(np.array(batch), max(batch_n))[1]
                for el in out:
                    yield el
                batch = []
                batch_n = []

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def order_ip(data_queries, data_docs, retrieve_counts, fast):
    """
    Generator which computes ordering of neighbours. If speed=False, the results are
    guaranteed to be accurate and correct.
    """
    import faiss
    data_queries = np.array(data_queries, dtype="float32")
    data_docs = np.array(data_docs, dtype="float32")

    index = faiss.IndexFlatIP(data_docs.shape[1])
    if fast:
        nlist = 200
        index = faiss.IndexIVFFlat(
            index, data_docs.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = 25
        index.train(data_docs)
    index.add(data_docs)

    BATCH_LIMIT = 256

    def n_new_gen():
        batch = []
        batch_n = []
        for i, (d, n) in enumerate(zip(data_queries, retrieve_counts)):
            batch.append(d)
            batch_n.append(n)
            if len(batch) >= BATCH_LIMIT or i == len(data_queries) - 1:
                out = index.search(np.array(batch), max(batch_n))[1]
                for el in out:
                    yield el
                batch = []
                batch_n = []

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def acc_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, n=20, fast=True
):
    n_new_gen = order_l2(
        data_queries, data_docs, [n] * len(data_queries), fast
    )
    return acc_from_relevancy(data_relevancy, n_new_gen, n)


def acc_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, n=20, fast=True
):
    n_new_gen = order_ip(
        data_queries, data_docs, [n] * len(data_queries), fast
    )
    return acc_from_relevancy(data_relevancy, n_new_gen, n)


def rprec_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_l2(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_from_relevancy(data_relevancy, n_new_gen)


def rprec_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_ip(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_from_relevancy(data_relevancy, n_new_gen)


def rprec_a_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_ip(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_a_from_relevancy(
        data_relevancy, n_new_gen, data_relevancy_articles, data_docs_articles
    )


def rprec_a_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_l2(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_a_from_relevancy(
        data_relevancy, n_new_gen, data_relevancy_articles, data_docs_articles
    )


def hits_a_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_ip(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return hits_a_from_relevancy(
        data_relevancy, n_new_gen, data_relevancy_articles, data_docs_articles
    )


def hits_a_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_l2(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return hits_a_from_relevancy(
        data_relevancy, n_new_gen, data_relevancy_articles, data_docs_articles
    )


def intersection_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_ip(
        data_queries, data_docs,
        [20] * len(data_queries), fast
    )
    return intersection_from_relevancy(
        data_relevancy, n_new_gen
    )


def intersection_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles, fast=True
):
    n_new_gen = order_l2(
        data_queries, data_docs
        [20] * len(data_queries), fast
    )
    return intersection_from_relevancy(
        data_relevancy, n_new_gen
    )
