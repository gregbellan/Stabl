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
        jaccard_mat = jaccard_mat[~np.eye(jaccard_mat.shape[0], dtype=bool)].reshape(
            jaccard_mat.shape[0], -1)

    return jaccard_mat


def adjusted_similarity(list1, list2, nb_total_elements):
    """Function to compute the adjusted similarity between two sets of elements.
    The adjusted similarity takes into account the size of the lists and adjust for random effects.

    Parameters
    ----------
    list1: array-like
        First list of elements

    list2: array-like
        Second list of elements

    nb_total_elements: int,
        Total number of elements in the original set of elements.

    Returns
    -------
    adjusted_similarity: float
        metrics in (-1, 1], closer to 1 = good, closer to -1 = bad.
        0 convention when list1 and list2 are empty and when list1 and list2 are equal to the original set.
    """
    set1, set2 = set(list1), set(list2)

    r = len(set1.intersection(set2))  # intersection
    k1, k2 = len(set1), len(set2)
    u = (len(list1) + len(list2)) - r  # union

    if u > nb_total_elements:
        raise ValueError(
            f"Union cardinal:{u} is greater than the total number of elements: {nb_total_elements}.")

    if k1 == nb_total_elements or k2 == nb_total_elements or k1 == 0 or k2 == 0:
        return 0

    num = r - (k1 * k2) / nb_total_elements
    denum = min(k1, k2) - max(0, k1 + k2 - nb_total_elements)

    return num / denum


def adjusted_similarity_values(list_of_lists, nb_total_elements):
    """Function to compute the adjusted similarity values.
    To do so we compute the matrix of adjusted similarities where each element [i,j] is the adjusted similarity for
    lists i and j. We then only return the upper triangle of the matrix (also removing the diagonal).

    Parameters
    ----------
    list_of_lists: array, size=n
        list of lists to compute the adjusted similarities for.
        array of size n

    nb_total_elements: int
        Total number of elements in the original set of elements.

    Returns
    -------
    adjusted similarity values: array, size=n(n-1)/2
        Values of the upper triangular matrix.
    """
    N = len(list_of_lists)
    adjusted_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            adjusted_matrix[i, j] = adjusted_similarity(list_of_lists[i],
                                                        list_of_lists[j],
                                                        nb_total_elements)

    adjusted_matrix_upper = adjusted_matrix[np.triu_indices_from(adjusted_matrix, k=1)]
    return adjusted_matrix_upper


def adjusted_similarity_measure(list_of_lists, nb_total_elements, stat="median"):
    """

    Parameters
    ----------
    list_of_lists: array, size=n
        list of lists to compute the adjusted similarities for.
        array of size n

    nb_total_elements: int
        Total number of elements in the original set of elements.

    stat: str, default="median"
        stat to apply to the adjusted similarity values.
        Can either be "median" or "mean"

    Returns
    -------
    statistic, err:
        if stat = "median", statistic is the median of the adjusted similarity values and err are the 1st and 3rd
        quantile
        if stat is "mean", the statistic is the mean and err is the standard error.
    """

    adjusted_matrix_upper = adjusted_similarity_values(list_of_lists, nb_total_elements)

    if stat == "median":
        return np.median(adjusted_matrix_upper), list(
            np.quantile(adjusted_matrix_upper, [.25, .75]))

    elif stat == "mean":
        return np.mean(adjusted_matrix_upper), np.std(adjusted_matrix_upper)

    else:
        raise ValueError(f"stat not recognized. Should either be 'median' or 'mean'. Got {stat}")


def pearson_similarity(list_i, list_j, d):
    """Function to compute the pearson similarity between two lists of selected features.

    Parameters
    ----------
    list_i: array
        List of selected features

    list_j: array
        List of selected features

    d: int
        Total number of features in the original dataset

    Returns
    -------
    pearson_similarity: float
        similarity score
    """

    set1, set2 = set(list_i), set(list_j)

    intersection = len(set1.intersection(set2))  # intersection
    ki, kj = len(set1), len(set2)
    expR = (ki * kj) / d
    pi, pj = ki / d, kj / d
    upsilon_i, upsilon_j = (pi * (1 - pi)) ** (1 / 2), (pj * (1 - pj)) ** (1 / 2)

    if (ki == d and kj == d) or (ki == 0 and kj == 0):
        similarity = 1
    elif ki != d and ki != 0 and kj != d and kj != 0:
        similarity = (intersection - expR) / (d * upsilon_i * upsilon_j)
    else:
        similarity = 0

    return similarity


def pearson_similarity_values(list_of_lists, d):
    """Function to compute the person similarity values.
    To do so we compute the matrix of pearson similarities where each element [i,j] is the pearson similarity for
    lists i and j. We then only return the upper triangle of the matrix (also removing the diagonal).

    Parameters
    ----------
    list_of_lists: array, size=n
        list of lists to compute the pearson similarities for

    d: int
        Total number of elements in the original set of elements

    Returns
    -------
    adjusted similarity values: array, size=n(n-1)/2
        Values of the upper triangular matrix.
    """

    N = len(list_of_lists)
    person_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            person_matrix[i, j] = pearson_similarity(
                list_of_lists[i],
                list_of_lists[j],
                d
            )

    person_matrix_upper = person_matrix[np.triu_indices_from(person_matrix, k=1)]
    return person_matrix_upper


def pearson_similarity_measure(list_of_lists, d, stat="median"):
    """

    Parameters
    ----------
    list_of_lists: array, size=n
        list of lists to compute the pearson similarities for.
        array of size n

    d: int
        Total number of elements in the original set of elements.

    stat: str, default="median"
        stat to apply to the adjusted similarity values.
        Can either be "median" or "mean"

    Returns
    -------
    statistic, err:
        if stat = "median", statistic is the median of the pearson similarity values and err are the 1st and 3rd
        quantile
        if stat is "mean", the statistic is the mean and err is the standard error.
    """

    pearson_matrix_upper = pearson_similarity_values(list_of_lists, d)

    if stat == "median":
        return np.median(pearson_matrix_upper), list(np.quantile(pearson_matrix_upper, [.25, .75]))

    elif stat == "mean":
        return np.mean(pearson_matrix_upper), np.std(pearson_matrix_upper)

    else:
        raise ValueError(f"stat not recognized. Should either be 'median' or 'mean'. Got {stat}")


def fdr_similarity(list1, list2):
    """
    Function to compute the fdr between two list of different sizes.

    Parameters
    ----------
    list1: array-like
        First list of elements

    list2: array-like
        Second list of elements. Considered as the true set.

    Returns
    -------
    fdr between the two lists.
    """

    tp = len(set(list1).intersection(list2))
    fp = len(set(list1).difference(list2))
    fn = len(set(list2).difference(list1))
    if fp + tp == 0:
        return 0

    return fp / (tp + fp)


def tpr_similarity(list1, list2):
    """
    Function to compute the tpr between two list of different sizes.

    Parameters
    ----------
    list1: array-like
        First list of elements

    list2: array-like
        Second list of elements. Considered as the true set.

    Returns
    -------
    fdr between the two lists.
    """

    tp = len(set(list1).intersection(list2))
    fp = len(set(list1).difference(list2))
    fn = len(set(list2).difference(list1))
    if fn + tp == 0:
        return 0

    return tp / (tp + fn)


def fscore_similarity(list1, list2, beta=1):
    """
    Function to compute the fscore between two list of different sizes.

    Parameters
    ----------
    list1: array-like
        First list of elements

    list2: array-like
        Second list of elements. Considered as the true set.

    beta: int
        f-score number

    Returns
    -------
    fdr between the two lists.
    """

    tp = len(set(list1).intersection(list2))
    fp = len(set(list1).difference(list2))
    fn = len(set(list2).difference(list1))

    num = (1 + beta**2) * tp
    den = (1 + beta**2) * tp + beta**2 * fn + fp

    if den == 0:
        return 0

    return num / den
