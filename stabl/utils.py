import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, r2_score, average_precision_score, \
    mean_absolute_error, mean_squared_error, precision_recall_curve, auc
from sklearn.model_selection import RepeatedKFold, cross_val_predict, \
    ParameterGrid, LeaveOneOut


def fit_predict(estimator, X, y, train, test, task_type):
    """
    Function to fit an estimator and retrieve the associated predictions

    Parameters
    ----------
    estimator: estimator
        The estimator to fit on the data

    X: array-like, size=(n_repeats, n_features)
        Input array

    y: array_like, size=(n_repeats, )
        Target array

    train: list
        The list of train indices

    test:list
        The list of test indices

    task_type: str
        The type of task to perform
        - "binary": for binary classification
        - "multiclass": for multiclass classification
        - "regression" : for regression tasks

    Returns
    -------
    predictions: array-like
        - if task_type = "binary" or "regression", predictions size=(n_repeats, )
        - if task_type = "multiclass", predictions size=(n_repeats, n_classes)
    """

    n_samples = len(y)

    if task_type in ["binary", "regression"]:
        results = np.empty(n_samples)
    else:
        n_classes = len(np.unique(y))
        results = np.empty((n_samples, n_classes))

    results[:] = np.nan

    if task_type == 'binary':
        results[test] = estimator.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    elif task_type == "multiclass":
        results[test] = estimator.fit(X[train], y[train]).predict_proba(X[test])
    elif task_type == "regression":
        results[test] = estimator.fit(X[train], y[train]).predict(X[test])
    else:
        raise ValueError(f"Task type should be in ['binary' ,'multiclass', 'regression']. Got {task_type}")

    return results


def nonpartition_cross_val_predict(
        estimator,
        X,
        y,
        task_type,
        splitter,
        groups=None
):
    """
    This function is a variation of the cross_val_predict function of scikit-learn
    The idea is that we can pass a non partition splitter (i.e. some samples can be found in several
    fold).
    The prediction on each sample is the median of its predictions in each fold where it is evaluated

    Parameters:
    -----------
    estimator: Classifier
        Classifier to be evaluated.

    X: array-like 
        Input data.

    y: array-like 
        Outcome.

    splitter: non-partition splitter.
        Can be any non-partitioning splitter like `RepeatedStratifiedFold` for example

    task_type: str
        The type of task to perform
        - "binary": for binary classification
        - "multiclass": for multiclass classification
        - "regression" : for regression tasks

    Returns
    -------
    predictions: array-like
        The array of raw predictions

    median_prediction: array-like
        The array of predictions' median
    """
    X, y = np.array(X), np.array(y)

    predictions = np.array(
        Parallel(
            n_jobs=-1,
            prefer='processes'
        )(delayed(fit_predict)(
            estimator,
            X,
            y,
            train,
            test,
            task_type
        ) for train, test in splitter.split(X, y, groups=groups)))

    median_prediction = np.nanmedian(predictions, axis=0)

    if task_type == "multiclass":
        median_prediction = median_prediction / np.sum(median_prediction, axis=1, keepdims=True)

    return predictions, median_prediction


def compute_CI(
        y_true,
        y_preds,
        confidence_level=0.95,
        scoring="roc_auc",
        return_CI_predictions=False
):
    """
    Function to predict the confidence interval a level 1-alpha
    for a given set of target and predicted probabilities.
    
    Parameters
    ----------
    return_CI_predictions
    y_true: array-like, size=(n_repeats,)
        Array of binary outcomes
        
    y_preds: array-like, size=(n_repeats,)
        Array of predicted probabilities.
        
    confidence_level: float, 0<alpha<1, default=0.95
        confidence level
    
    scoring: str, default="auc"
        String to indicate what type of curve the CI is computed for.
        - "roc_auc": Area under the Receiver Operating Curve
        - "average_precision": Average Precision
        - "r2": R2-score
        
    Returns
    -------
    """
    scores = []
    percentiles = (1 - confidence_level) / 2, 1 - (1 - confidence_level) / 2
    percentiles = np.array(percentiles) * 100

    sampled_indices = np.zeros(shape=(1000, y_true.shape[0]), dtype=int)

    y_true, y_preds = np.array(y_true), np.array(y_preds)

    for iteration in range(1000):
        samples = np.random.choice(a=y_true.shape[0], size=y_true.shape[0], replace=True)
        if scoring == "roc_auc":
            while len(np.unique(y_true[samples])) != 2:
                samples = np.random.choice(a=y_true.shape[0], size=y_true.shape[0], replace=True)

            scores.append(roc_auc_score(y_true[samples], y_preds[samples]))
        if scoring == "average_precision":
            while len(np.unique(y_true[samples])) != 2:
                samples = np.random.choice(a=y_true.shape[0], size=y_true.shape[0], replace=True)

            scores.append(average_precision_score(y_true[samples], y_preds[samples]))
        if scoring == "prc_auc":
            while len(np.unique(y_true[samples])) != 2:
                samples = np.random.choice(a=y_true.shape[0], size=y_true.shape[0], replace=True)

            precision, recall, _ = precision_recall_curve(y_true[samples], y_preds[samples])
            prc_auc = auc(recall, precision)
            scores.append(prc_auc)
        if scoring == "roc_auc_ovr":
            while len(np.unique(y_true[samples])) < 2:
                samples = np.random.choice(a=y_true.shape[0], size=y_true.shape[0], replace=True)
            scores.append(roc_auc_score(y_true[samples], y_preds[samples], multi_class="ovr"))
        if scoring == "r2":
            scores.append(r2_score(y_true[samples], y_preds[samples]))
        if scoring == "rmse":
            scores.append(np.sqrt(mean_squared_error(y_true[samples], y_preds[samples])))
        if scoring == "mae":
            scores.append(mean_absolute_error(y_true[samples], y_preds[samples]))

        sampled_indices[iteration] = samples

    if scoring in ["roc_auc", "prc_auc", "average_precision", "roc_auc_ovr"] and return_CI_predictions:
        score_sorted = np.argsort(scores)
        indices_up = sampled_indices[score_sorted[int(percentiles[1] * 10) - 1]]
        indices_low = sampled_indices[score_sorted[int(percentiles[0] * 10) - 1]]
        df_bound = pd.DataFrame(
            {"target_low": np.array(y_true[indices_low]),
             "preds_low": np.array(y_preds[indices_low]),
             "target_up": np.array(y_true[indices_up]),
             "preds_up": np.array(y_preds[indices_up])
             }
        )
        return df_bound, np.percentile(scores, percentiles)

    else:
        return np.percentile(scores, percentiles)


def permutation_test_between_clfs(y_true, pred_probas_1, pred_probas_2, scoring="roc_auc", n_repeats=1000):
    """
    Function to perform a permutation test between two classifiers predictions.

    Parameters
    ----------
    y_true: pd.Series
        pandas Series containing the true outcome.

    pred_probas_1: pd.Series
        predicted probabilities of the first classifier.

    pred_probas_2: pd.Series
        predicted probabilities of the second classifier.

    scoring: string, default="roc_auc"
        can also be average_precision.

    n_repeats: int, default=1000
        Number of experiments to compute the p-value.

    Returns
    -------
    observed_difference: float
        Difference in score between the two classifiers

    p-value: float
        Associated p-value.
    """

    score_differences = []

    if scoring == "roc_auc":
        score1 = roc_auc_score(y_true.ravel(), pred_probas_1.ravel())
        score2 = roc_auc_score(y_true.ravel(), pred_probas_2.ravel())

    elif scoring == "average_precision":
        score1 = average_precision_score(y_true.ravel(), pred_probas_1.ravel())
        score2 = average_precision_score(y_true.ravel(), pred_probas_2.ravel())

    else:
        raise ValueError(f"")

    observed_difference = score1 - score2

    for _ in range(n_repeats):
        mask = np.random.randint(2, size=len(pred_probas_1.ravel()))
        p1 = np.where(mask, pred_probas_1.ravel(), pred_probas_2.ravel())
        p2 = np.where(mask, pred_probas_2.ravel(), pred_probas_1.ravel())

        if scoring == "roc_auc":
            score1 = roc_auc_score(y_true.ravel(), p1)
            score2 = roc_auc_score(y_true.ravel(), p2)

        elif scoring == "average_precision":
            score1 = average_precision_score(y_true.ravel(), p1)
            score2 = average_precision_score(y_true.ravel(), p2)

        score_differences.append(score1 - score2)

    if observed_difference >= 0:
        p_value = np.mean(score_differences >= observed_difference)
    else:
        p_value = np.mean(score_differences <= observed_difference)

    return observed_difference, p_value
