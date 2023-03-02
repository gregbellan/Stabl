import os
from pathlib import Path
import pandas as pd
import numpy as np

from .utils import compute_CI, permutation_test_between_clfs
from .visualization import scatterplot_regression_predictions, boxplot_binary_predictions, plot_roc, \
    plot_prc
from .metrics import jaccard_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error

from scipy import stats
from scipy.stats import mannwhitneyu


def save_plots(predictions_dict, y, task_type, save_path):
    """
    Function to save the plots when performing a stabl against lasso benchmark.
    In the case of binary classification, the function will save the associated roc and precision recall curve as well
    as the boxplot of binary classification predictions.
    In the case of regression, the function will save the associated scatter plot of regression predictions.

    Parameters
    ----------
    predictions_dict: dict
        Dictionary of predictions. Each value can either be a pandas Series or a pandas DataFrame,
        depending on whether we computed the predictions once or on multiple folds.
        The predictions_dict should contain the predictions associated to the Lasso and have the key "Lasso"

    y: pd.Series
        Outcome pandas Series used for evaluation. In the case of cross validation, this is also the outcome used for
        training. In the case of validation this is the validation outcome.

    task_type: str
        Type of task. Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the plots

    use_median_preds: bool, default=True
        If True the function will take for each predictions_dict value the median over the columns. This corresponds to
        the case of cross-validation where we have some predictions at each fold that we want to median from.

    """
    for name, predictions in predictions_dict.items():

        saving_path = Path(save_path, name)
        os.makedirs(saving_path, exist_ok=True)
        pd.concat(
            [predictions, y.loc[predictions.index]], axis=1).to_csv(
            os.path.join(saving_path, f"{name} predictions.csv")
        )

        if task_type == "binary":
            plot_roc(
                y_true=y,
                y_preds=predictions,
                show_CI=True,
                export_file=True,
                show_fig=False,
                path=os.path.join(saving_path, f"{name} ROC.pdf")
            )

            plot_prc(
                y_true=y,
                y_preds=predictions,
                show_CI=True,
                export_file=True,
                show_fig=False,
                path=os.path.join(saving_path, f"{name} PR Curve.pdf")
            )

            boxplot_binary_predictions(
                y_true=y,
                y_preds=predictions,
                export_file=True,
                show_fig=False,
                path=os.path.join(saving_path, f"{name} Boxplot of median predictions.pdf")
            )

        elif task_type == "regression":
            scatterplot_regression_predictions(
                y_true=y,
                y_preds=predictions,
                export_file=True,
                show_fig=False,
                path=os.path.join(saving_path, f"{name} Scatter-plot of median predictions.pdf")
            )


def compute_scores_table(
        predictions_dict,
        y,
        task_type="binary",
        selected_features_dict=None
):
    """Function to output the table of scores
    for a STABL against Lasso benchmark on a single omic.

    Parameters
    ----------
    selected_features_dict
    predictions_dict: dict
        Dictionary of raw predictions (should contain a "Lasso" key).

    y: pd.Series
        pandas Series containing the outcomes.

    task_type: string, default="binary"
        Type of task, can either be "binary" or "regression".

    Returns
    -------
    table_of_scores: pd.DataFrame
    """

    scores_columns = []
    if selected_features_dict is not None:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision", "N features", "CVS"]

        elif task_type == "regression":
            scores_columns = ["R2", "RMSE", "MAE", "N features", "CVS"]
            
    else:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision"]

        elif task_type == "regression":
            scores_columns = ["R2", "RMSE", "MAE"]

    table_of_scores = pd.DataFrame(data=None, columns=scores_columns)

    for model, preds in predictions_dict.items():
        stabl_preds = predictions_dict["STABL"]

        if task_type == "binary":
            model_roc = roc_auc_score(y, preds)
            model_roc_CI = compute_CI(y, preds, scoring="roc_auc")
            cell_value = f"{model_roc:.3f} [{model_roc_CI[0]:.3f}, {model_roc_CI[1]:.3f}]"
            if model != "STABL":
                p_value = permutation_test_between_clfs(y, preds, stabl_preds, scoring="roc_auc")[1]
                cell_value = cell_value + f" (p={p_value})"
            table_of_scores.loc[model, "ROC AUC"] = cell_value

            model_ap = average_precision_score(y, preds)
            model_ap_CI = compute_CI(y, preds, scoring="average_precision")
            cell_value = f"{model_ap:.3f} [{model_ap_CI[0]:.3f}, {model_ap_CI[1]:.3f}]"
            if model != "STABL":
                p_value = permutation_test_between_clfs(y, preds, stabl_preds, scoring="average_precision")[1]
                cell_value = cell_value + f" (p={p_value})"
            table_of_scores.loc[model, "Average Precision"] = cell_value

        elif task_type == "regression":
            model_r2 = r2_score(y, preds)
            model_r2_CI = compute_CI(y, preds, scoring="r2")
            table_of_scores.loc[model, "R2"] = f"{model_r2:.3f} [{model_r2_CI[0]:.3f}, {model_r2_CI[1]:.3f}]"

            model_rmse = np.sqrt(mean_squared_error(y, preds))
            model_rmse_CI = compute_CI(y, preds, scoring="rmse")
            table_of_scores.loc[model, "RMSE"] = f"{model_rmse:.3f} [{model_rmse_CI[0]:.3f}, {model_rmse_CI[1]:.3f}]"

            model_mae = mean_absolute_error(y, preds)
            model_mae_CI = compute_CI(y, preds, scoring="mae")
            table_of_scores.loc[model, "MAE"] = f"{model_mae:.3f} [{model_mae_CI[0]:.3f}, {model_mae_CI[1]:.3f}]"

        if selected_features_dict is not None:
            sel_features_stabl = selected_features_dict["STABL"]["Fold nb of features"]
            jaccard_mat_stabl = jaccard_matrix(selected_features_dict["STABL"]["Fold selected features"], remove_diag=False)
            jaccard_val_stabl = jaccard_mat_stabl[np.triu_indices_from(jaccard_mat_stabl, k=1)]

            median_features = np.median(sel_features_stabl)
            iqr_features = np.quantile(sel_features_stabl, [.25, .75])
            cell_value = f"{median_features:.3f} [{iqr_features[0]:.3f}, {iqr_features[1]:.3f}]"
            table_of_scores.loc["STABL", "N features"] = cell_value

            jaccard_median = np.median(jaccard_val_stabl)
            jaccard_iqr = np.quantile(jaccard_val_stabl, [.25, .75])
            cell_value = f"{jaccard_median:.3f} [{jaccard_iqr[0]:.3f}, {jaccard_iqr[1]:.3f}]"
            table_of_scores.loc["STABL", "CVS"] = cell_value

            if model != "STABL":
                sel_features = selected_features_dict[model]["Fold nb of features"]
                jaccard_mat = jaccard_matrix(selected_features_dict[model]["Fold selected features"], remove_diag=False)
                jaccard_val = jaccard_mat[np.triu_indices_from(jaccard_mat, k=1)]
                p_value_feature = mannwhitneyu(x=sel_features, y=sel_features_stabl).pvalue
                p_value_feature = f" (p={p_value_feature:.3e})"
                p_value_cvs = mannwhitneyu(x=jaccard_val, y=jaccard_val_stabl).pvalue
                p_value_cvs = f" (p={p_value_cvs:.3e})"

                median_features = np.median(sel_features)
                iqr_features = np.quantile(sel_features, [.25, .75])
                cell_value = f"{median_features:.3f} [{iqr_features[0]:.3f}, {iqr_features[1]:.3f}]" + p_value_feature
                table_of_scores.loc[model, "N features"] = cell_value

                jaccard_median = np.median(jaccard_val)
                jaccard_iqr = np.quantile(jaccard_val, [.25, .75])
                cell_value = f"{jaccard_median:.3f} [{jaccard_iqr[0]:.3f}, {jaccard_iqr[1]:.3f}]" + p_value_cvs
                table_of_scores.loc[model, "CVS"] = cell_value

    return table_of_scores


def compute_features_table(
        selected_features_dict,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        task_type="binary"
):
    """

    Parameters
    ----------
    selected_features_dict

    X_train: pd.DataFrame
        Training input dataframe

    y_train: pd.Series
        Training outcome pandas Series.

    X_test: pd.DataFrame, default=None
        Testing input dataframe

    y_test: pd.Series, default=None
        Testing outcome pandas Series

    task_type: str, default="binary"
        task type "binary" for binary classification, "regression" for regression.

    Returns
    -------

    """
    all_features = []
    for model, el in selected_features_dict.items():
        all_features += list(selected_features_dict[model])
        
    all_features = np.unique(all_features)
    
    df_out = pd.DataFrame(
        data=False, 
        index=all_features, 
        columns=[f"Selected by {model}" for model in selected_features_dict.keys()])
    
    for model in selected_features_dict.keys():
        df_out.loc[selected_features_dict[model], f"Selected by {model}"] = True

    if task_type == "binary":
        df_out["Train Mannwithney pvalues"] = [mannwhitneyu(
            X_train.loc[y_train == 0, i],
            X_train.loc[y_train == 1, i],
            nan_policy="omit")[1] for i in all_features]

        df_out["Train T-test pvalues"] = [stats.ttest_ind(
            X_train.loc[y_train == 1, i],
            X_train.loc[y_train == 0, i],
            nan_policy="omit")[1] for i in all_features]
        
    elif task_type == "regression":
        df_out["Train Pearson-r pvalues"] = [stats.pearsonr(
            X_train.loc[:, i].dropna(),
            y_train.loc[X_train.loc[:, i].dropna().index])[1] for i in all_features]

        df_out["Train Spearman-r pvalues"] = [stats.spearmanr(
            X_train.loc[:, i].dropna(),
            y_train.loc[X_train.loc[:, i].dropna().index])[1] for i in all_features]

    if X_test is not None:
        if task_type == "binary":
            df_out["Test Mannwithney pvalues"] = [mannwhitneyu(X_test.loc[y_test == 0, i],
                                                               X_test.loc[y_test == 1, i],
                                                               nan_policy="omit")[1]
                                                  for i in all_features]

            df_out["Test T-test pvalues"] = [stats.ttest_ind(X_test.loc[y_test == 1, i],
                                                             X_test.loc[y_test == 0, i],
                                                             nan_policy="omit")[1]
                                             for i in all_features]
        elif task_type == "regression":

            df_out["Test Pearson-R pvalues"] = [stats.pearsonr(X_test.loc[:, i].dropna(),
                                                               y_test.loc[X_test.loc[:, i].dropna().index]
                                                               )[1]
                                                for i in all_features]

            df_out["Test Spearman-R pvalues"] = [stats.spearmanr(X_test.loc[:, i].dropna(),
                                                                 y_test.loc[X_test.loc[:, i].dropna().index]
                                                                 )[1]
                                                 for i in all_features]

    return df_out
