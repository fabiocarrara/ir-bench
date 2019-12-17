import numpy as np
import itertools


def dcg(y_true, y_score, p=None, normalized=False):
    """
    Compute the (normalized) Discounted Cumulative Gain @ rank :param p.
    :param y_true: a 2-D array (n_query x n_samples) containing groundtruth relevance values rel_i
    :param y_score: a 2-D array (n_query x n_samples) containing the predicted ranking scores
    :param p: the rank at which to compute the (n)DCG; if None, p = n_samples
    :param normalized: whether to compute the normalized DCG
    :return: (n)DCG score
    """
    y_true = np.atleast_2d(y_true)
    y_score = np.atleast_2d(y_score)

    n_queries, n_samples = y_true.shape
    p = p if p else n_samples

    if normalized:
        optimal = np.sort(y_true, axis=1)[:, ::-1]
        optimal = optimal[:, :p]
        discounted_gain = (2 ** optimal - 1) / np.log2(np.arange(1, p + 1) + 1).reshape(1, -1)
        idcg = discounted_gain.sum(axis=1, keepdims=True)

    order = y_score.argsort(axis=1)[:, ::-1]  # descending order by rows
    order = order[:, :p]  # TODO this may truncate a tie group

    y_true = y_true[np.arange(n_queries).reshape(n_queries, 1), order]
    # y_score = y_score[np.arange(n_queries).reshape(n_queries, 1), order]

    def _tdcg(ranked_relevances):
        for i, (rel, group) in enumerate(itertools.groupby(ranked_relevances), start=1):
            pass  # TODO Ties-aware

    # nDCG (no ties)
    discounted_gain = (2 ** y_true - 1) / np.log2(np.arange(1, p + 1) + 1).reshape(1, -1)
    dcg = discounted_gain.sum(axis=1, keepdims=True)
    
    if normalized:
        dcg /= idcg
        
    return dcg.squeeze()


if __name__ == '__main__':

    y_true = [[0, 1, 2, 1], [1, 2, 2, 3]]
    y_score = [[0.15, 0.7, 0.06, 0.1], [0.5, 0.4, 0.1, 0.9]]
    
    print(dcg(y_true, y_score, normalized=True))
