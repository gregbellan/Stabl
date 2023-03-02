import numpy as np


def jaccard_similarity(list1, list2):
    """
    Function to compute the jaccard similarity between two list of different sizes.

    Parameters
    ----------
    list1: array-like
        First list of elements

    list2: array-like
        Second list of elements

    Returns
    -------
    Jaccard similarity between the two lists.
    """

    intersection = len(set(list1).intersection(list2))
    union = (len(list1) + len(list2)) - intersection

    if (intersection == 0) and (union == 0):
        return 0

    return float(intersection) / union


def jaccard_matrix(list_of_lists, remove_diag=True):
    """
    Function to compute the jaccard matrix from a list of lists.
    Jaccard_Matrix[i, j] = jaccard_similarity(list_of_lists[i], list_of_lists[j])

    Parameters
    ----------
    list_of_lists: array-like
        Should contain lists of elements

    remove_diag: bool, default=True
        If True, the diagonal (only made of ones) will be removed.

    Returns
    -------
    jaccard_matrix: array-like, size=(len(list_of_lists), len(list_of_lists))
    """

    N = len(list_of_lists)
    jaccard_mat = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            jaccard_mat[i, j] = jaccard_similarity(list_of_lists[i], list_of_lists[j])

    if remove_diag:
        jaccard_mat = jaccard_mat[~np.eye(jaccard_mat.shape[0], dtype=bool)].reshape(jaccard_mat.shape[0], -1)

    return jaccard_mat

