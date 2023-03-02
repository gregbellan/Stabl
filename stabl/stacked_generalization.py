# __LIBRARIES__

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, r2_score


# __FUNCTIONS__

def stacked_multi_omic(df_predictions, y, task_type, n_iter=10000):
    """
    Functions to compute the stacked generalization using the prediction of 
    models trained on individual omics.

    The function automatically handles missing values.

    Parameters
    ----------
    df_predictions: pd.DataFrame
        pandas DataFrame containing all the predictions for each omic.

    y: pd.Series
        pandas Series containing all the outcomes.

    task_type: string
        "binary" or "regression"

    n_iter: int
        Number of iterations to perform; each iteration corresponds to a random search 
        of weights to test.

    Returns
    -------
    df_predictions: pd.DataFrame
        pandas DataFrame containing the predictions for each omic as well as the 
        weighted stacked generalization predictions

    df_weights: pd.DataFrame
        pandas DataFrame containing the final weights associated to each omic.
    """

    df_predictions = df_predictions.drop(columns=y.name, errors="ignore")

    best_score = -100
    best_weights = []
    best_probs = []

    for i in range(n_iter):
        weights = np.random.uniform(0, 10, df_predictions.shape[1])
        weighted_probs = ((df_predictions * weights).sum(1)) / ((~df_predictions.isna() * weights).sum(1))
        if task_type == "binary":
            try:
                score = roc_auc_score(y, weighted_probs)
            except:
                continue
        elif task_type == "regression":
            try:
                score = r2_score(y, weighted_probs)
            except:
                continue
        if score > best_score:
            best_probs = weighted_probs
            best_score = score
            best_weights = weights

    df_weights = pd.DataFrame(data={"Associated weight": best_weights},
                              index=df_predictions.columns
                              )
    df_predictions["Stacked Gen. Predictions"] = best_probs
    df_predictions[y.name] = y

    return df_predictions, df_weights