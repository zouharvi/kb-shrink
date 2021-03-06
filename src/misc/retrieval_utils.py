import numpy as np
from scipy.spatial.distance import minkowski
import argparse
import torch

if torch.cuda.is_available():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0")
    args,_ = parser.parse_known_args()
    DEVICE = torch.device("cuda:" + args.gpu)
    print(args.gpu, DEVICE)
else:
    DEVICE = torch.device("cpu")

def l2_sim(x, y):
    return -minkowski(x, y)


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
            if len(batch) >= BATCH_LIMIT or i == len(data_queries)-1:
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
            if len(batch) >= BATCH_LIMIT or i == len(data_queries)-1:
                out = index.search(np.array(batch), max(batch_n))[1]
                for el in out:
                    yield el
                batch = []
                batch_n = []

    # pass generators so that the resulting vectors don't have to be stored in memory
    return n_new_gen()


def acc_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles,
    n=20, fast=True, report=False
):
    n_new_gen = order_l2(data_queries, data_docs, [n]*len(data_queries), fast)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)


def acc_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles,
    n=20, fast=True, report=False
):
    n_new_gen = order_ip(data_queries, data_docs, [n]*len(data_queries), fast)
    return acc_from_relevancy(data_relevancy, n_new_gen, n, report)


def rprec_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles,
    n=20, fast=True, report=False
):
    n_new_gen = order_l2(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_from_relevancy(data_relevancy, n_new_gen, report)

def rprec_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles,
    n=20, fast=True, report=False
):
    n_new_gen = order_ip(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_from_relevancy(data_relevancy, n_new_gen, report)

def rprec_a_ip(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles,
    n=20, fast=True, report=False
):
    n_new_gen = order_ip(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_a_from_relevancy(
        data_relevancy, n_new_gen, data_relevancy_articles, data_docs_articles, report
    )

def rprec_a_l2(
    data_queries, data_docs, data_relevancy, data_relevancy_articles, data_docs_articles,
    n=20, fast=True, report=False
):
    n_new_gen = order_l2(
        data_queries, data_docs,
        [len(x) for x in data_relevancy],
        fast
    )
    return rprec_a_from_relevancy(
        data_relevancy, n_new_gen, data_relevancy_articles, data_docs_articles, report
    )

def acc_from_relevancy(relevancy, n_new, n, report=False):
    def acc_local(doc_true, doc_hyp):
        """
        Accuracy for one query
        """
        if len(set(doc_true) & set(doc_hyp[:n])) != 0:
            return 1
        else:
            return 0

    acc_val = np.average([acc_local(x, y) for x, y in zip(relevancy, n_new)])
    if report:
        print(f"ACC (top {n}) is {acc_val:.3f} (best is 1, worst is 0)")

    return acc_val


def rprec_from_relevancy(relevancy, n_new, report=False):
    def rprec_local(doc_true, doc_hyp):
        """
        R-Precision for one query
        """
        return len(set(doc_hyp[:len(doc_true)]) & set(doc_true))/len(doc_true)

    rprec_val = np.average([
        rprec_local(x, y)
        for x, y in zip(relevancy, n_new)
    ])
    if report:
        print(f"RPrec is {rprec_val:.3f} (best is 1, worst is 0)")

    return rprec_val


def rprec_a_from_relevancy(relevancy, n_new, relevancy_articles, docs_articles, report=False):
    def rprec_local(doc_true, articles_true, doc_hyp):
        """
        R-Precision for one query
        """
        
        articles_hyp = {docs_articles[doc] for doc in doc_hyp[:len(doc_true)]}
        return len(articles_hyp & articles_true)/len(articles_true)

    rprec_val = np.average([
        rprec_local(*x)
        for x in zip(relevancy, relevancy_articles, n_new)
    ])
    if report:
        print(f"RPrec is {rprec_val:.3f} (best is 1, worst is 0)")

    return rprec_val
