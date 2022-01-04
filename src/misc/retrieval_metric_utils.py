import numpy as np

def acc_from_relevancy(relevancy, n_new, n):
    def acc_local(doc_true, doc_hyp):
        """
        Accuracy for one query
        """
        if len(set(doc_true) & set(doc_hyp[:n])) != 0:
            return 1
        else:
            return 0

    acc_val = np.average([acc_local(x, y) for x, y in zip(relevancy, n_new)])

    return acc_val


def rprec_from_relevancy(relevancy, n_new):
    def rprec_local(doc_true, doc_hyp):
        """
        R-Precision for one query
        """
        return len(set(doc_hyp[:len(doc_true)]) & set(doc_true)) / len(doc_true)

    rprec_val = np.average([
        rprec_local(x, y)
        for x, y in zip(relevancy, n_new)
    ])

    return rprec_val


def rprec_a_from_relevancy(relevancy, n_new, relevancy_articles, docs_articles):
    def rprec_local(doc_true, articles_true, doc_hyp):
        """
        R-Precision for one query
        """

        articles_hyp = {
            docs_articles[doc]
            if doc < len(docs_articles)
            else -1  # hotfix for irrelevant effects (adding new docs)
            for doc in doc_hyp[:len(doc_true)]
        }
        return len(articles_hyp & articles_true) / len(articles_true)

    rprec_val = np.average([
        rprec_local(*x)
        for x in zip(relevancy, relevancy_articles, n_new)
    ])

    return rprec_val


def hits_a_from_relevancy(relevancy, n_new, relevancy_articles, docs_articles):
    def rprec_local(doc_true, articles_true, doc_hyp):
        """
        R-Precision for one query
        """

        articles_hyp = {
            docs_articles[doc]
            if doc < len(docs_articles)
            else -1  # hotfix for irrelevant effects (adding new docs)
            for doc in doc_hyp[:len(doc_true)]
        }
        return len(articles_hyp & articles_true)

    rprec_val = [
        rprec_local(*x)
        for x in zip(relevancy, relevancy_articles, n_new)
    ]

    return rprec_val



def intersection_from_relevancy(relevancy, n_new):
    def rprec_local(doc_true, doc_hyp):
        """
        Useful spans for one query
        """
        return set(doc_hyp[:len(doc_true)]) & set(doc_true)

    rprec_val = [
        rprec_local(x, y)
        for x, y in zip(relevancy, n_new)
    ]

    return rprec_val