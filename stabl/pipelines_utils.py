import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_selection._base import _get_feature_importances
from sklearn.feature_selection._from_model import _calculate_threshold

from .utils import compute_CI, permutation_test_between_clfs
from .visualization import scatterplot_regression_predictions, boxplot_binary_predictions, plot_roc, plot_prc
from .metrics import jaccard_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error

from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.model_selection import GridSearchCV


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

    """
    for name, predictions in predictions_dict.items():
        print(name, predictions)
        saving_path = Path(save_path, name)
        os.makedirs(saving_path, exist_ok=True)
        pd.concat([predictions, y.loc[predictions.index]], axis=1).to_csv(
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
                paths=os.path.join(saving_path, f"{name} Scatter-plot of median predictions.pdf")
            )


def compute_scores_table(
        predictions_dict,
        y,
        task_type="binary",
        selected_features_dict=None
):
    """Function to output the table of scores for benchmarking.

    Parameters
    ----------
    predictions_dict: dict
        Dictionary of raw predictions of shape : {model_name: predictions}.

    y: pd.Series
        pandas Series containing the outcomes.

    task_type: string, default="binary"
        Type of task, can either be "binary" or "regression".

    selected_features_dict: dict, default=None
        Dictionary of selected features of shape : {model_name: pd.DataFrame} 
        Each DataFrame should contain "Fold nb of features" and "Fold selected features" keys.

    Returns
    -------
    table_of_scores: pd.DataFrame
        pandas DataFrame containing the scores of each model.
    """

    scores_columns = []
    if selected_features_dict is not None:
        if task_type == "binary":
            scores_columns = ["ROC AUC",
                              "Average Precision", "N features", "CVS"]

        elif task_type == "regression":
            scores_columns = ["R2", "RMSE", "MAE", "N features", "CVS"]

    else:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision"]

        elif task_type == "regression":
            scores_columns = ["R2", "RMSE", "MAE"]

    table_of_scores = pd.DataFrame(data=None, columns=scores_columns)

    for model, preds in predictions_dict.items():

        for metric in scores_columns:
            if metric == "ROC AUC":
                model_roc = roc_auc_score(y, preds)
                model_roc_CI = compute_CI(y, preds, scoring="roc_auc")
                cell_value = f"{model_roc:.3f} [{model_roc_CI[0]:.3f}, {model_roc_CI[1]:.3f}]"

            elif metric == "Average Precision":
                model_ap = average_precision_score(y, preds)
                model_ap_CI = compute_CI(y, preds, scoring="average_precision")
                cell_value = f"{model_ap:.3f} [{model_ap_CI[0]:.3f}, {model_ap_CI[1]:.3f}]"

            elif metric == "N features":
                sel_features = selected_features_dict[model]["Fold nb of features"]
                median_features = np.median(sel_features)
                iqr_features = np.quantile(sel_features, [.25, .75])
                cell_value = f"{median_features:.3f} [{iqr_features[0]:.3f}, {iqr_features[1]:.3f}]"

            elif metric == "CVS":
                jaccard_mat = jaccard_matrix(
                    selected_features_dict[model]["Fold selected features"], remove_diag=False)
                jaccard_val = jaccard_mat[np.triu_indices_from(
                    jaccard_mat, k=1)]
                jaccard_median = np.median(jaccard_val)
                jaccard_iqr = np.quantile(jaccard_val, [.25, .75])
                cell_value = f"{jaccard_median:.3f} [{jaccard_iqr[0]:.3f}, {jaccard_iqr[1]:.3f}]"

            elif metric == "R2":
                model_r2 = r2_score(y, preds)
                model_r2_CI = compute_CI(y, preds, scoring="r2")
                cell_value = f"{model_r2:.3f} [{model_r2_CI[0]:.3f}, {model_r2_CI[1]:.3f}]"

            elif metric == "RMSE":
                model_rmse = np.sqrt(mean_squared_error(y, preds))
                model_rmse_CI = compute_CI(y, preds, scoring="rmse")
                cell_value = f"{model_rmse:.3f} [{model_rmse_CI[0]:.3f}, {model_rmse_CI[1]:.3f}]"

            elif metric == "MAE":
                model_mae = mean_absolute_error(y, preds)
                model_mae_CI = compute_CI(y, preds, scoring="mae")
                cell_value = f"{model_mae:.3f} [{model_mae_CI[0]:.3f}, {model_mae_CI[1]:.3f}]"
            else:
                raise ValueError(f"Metric not recognized.")

            table_of_scores.loc[model, metric] = cell_value

    return table_of_scores


def compute_pvalues_table(
        predictions_dict,
        y,
        task_type="binary",
        selected_features_dict=None
):
    """Function to output the p-values table for benchmarking. The p-values are computed between each model.

    Parameters
    ----------
    predictions_dict: dict
        Dictionary of raw predictions of shape : {model_name: predictions}.

    y: pd.Series
        pandas Series containing the outcomes.

    task_type: string, default="binary"
        Type of task, can either be "binary" or "regression".

    selected_features_dict : dict, default=None
        Dictionary of selected features of shape : {model_name: pd.DataFrame} 
        Each DataFrame should contain "Fold nb of features" and "Fold selected features" keys.

    Returns
    -------
    p_values_dict: dict of pd.DataFrame ({metric: pd.DataFrame})
        Dictionary of pandas DataFrames containing the p-values on the metric between each model.
    """

    scores_columns = []
    if selected_features_dict is not None:
        if task_type == "binary":
            scores_columns = ["ROC AUC",
                              "Average Precision", "N features", "CVS"]

        elif task_type == "regression":
            scores_columns = ["Prediction", "N features", "CVS"]

    else:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision"]

        elif task_type == "regression":
            scores_columns = ["Prediction"]

    p_values_dict = {
        s: pd.DataFrame(
            columns=np.array(predictions_dict.keys()),
            index=np.array(predictions_dict.keys())
        ) for s in scores_columns
    }

    for metric in scores_columns:
        p_values_df = p_values_dict[metric]
        for model, preds in predictions_dict.items():
            for model2, preds2 in predictions_dict.items():

                if metric == "ROC AUC":
                    p_value = permutation_test_between_clfs(
                        y, preds, preds2, scoring="roc_auc")[1]

                elif metric == "Average Precision":
                    p_value = permutation_test_between_clfs(
                        y, preds, preds2, scoring="average_precision")[1]

                elif metric == "N features":
                    sel_features = selected_features_dict[model]["Fold nb of features"]
                    sel_features2 = selected_features_dict[model2]["Fold nb of features"]
                    p_value = mannwhitneyu(
                        x=sel_features, y=sel_features2).pvalue

                elif metric == "CVS":
                    jaccard_mat = jaccard_matrix(
                        selected_features_dict[model]["Fold selected features"], remove_diag=False)
                    jaccard_val = jaccard_mat[np.triu_indices_from(
                        jaccard_mat, k=1)]

                    jaccard_mat2 = jaccard_matrix(
                        selected_features_dict[model2]["Fold selected features"], remove_diag=False)
                    jaccard_val2 = jaccard_mat2[np.triu_indices_from(
                        jaccard_mat2, k=1)]

                    p_value = mannwhitneyu(
                        x=jaccard_val, y=jaccard_val2).pvalue

                else:
                    p_value = mannwhitneyu(
                        x=preds, y=preds2).pvalue

                p_values_df.loc[model, model2] = p_value

    return p_values_dict


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


class BenchmarkWrapper():
    """Wrapper for benchmarking models with basic implement of necessary methods.
    """
    def __init__(
            self,
            model,
            fit=None,
            predict=None,
            use_predict_proba=True,
            get_support=None,
            get_importances=None,
            threshold=1e-5
    ) -> None:
        """Initiate the wrapper.

        Parameters
        ----------
        model : sklearn estimator
            Model to wrap.
        fit : function, optional
            If provided, it is the function used as the fit function of the wrapper.
            It is directly the called function, so you can access to the model by
            'self.model' or with the fit function of the model directly.
            If None, the wrapper will automatically take the 'fit' function of the model,
            or throws an error. By default, it is set to None.

        predict : function, optional
            If provided, it is the function used as the predict function of the wrapper.
            It is directly the called function, so you can access to the model by
            'self.model' or with the predict function of the model directly.
            If None, the wrapper will automatically take the 'predict' function of the model,
            or throws an error. By default, it is set to None.

        get_support : function, optional
            If provided, it is the function used as the get_support function of the wrapper. 
            It is directly the called function, so you can access to the model by
            'self.model' or with the get_support function of the model directly. 
            If None, the wrapper will automatically take the 'get_support' function of the model, 
            or determine the support with the 'get_importances' function like in 
            SelectFromModel class of sklearn, or throws an error. 
            By default, it is set to None.

        get_importances : function, optional
            If provided, it is the function used as the 'get_importances' function of the wrapper. 
            It is directly the called function, so you can access to the model by
            'self.model' or with the 'get_importances' function of the model directly. 
            If None, the wrapper will automatically take the 'get_importances' function of the model, 
            or determine the feature importances thanks to the '_get_feature_importances' function of 
            sklearn (get with coef_ or feature_importances_ attribute of model) and take the absolute value, 
            or throws an error. By default, it is set to None.

        threshold : str or float, optional
            The threshold value to use for feature selection if 'get_support' is not provided or implemented in the model. 
            Features whose absolute importance value is greater or equal are kept while the others are discarded.
            If “median” (resp. “mean”), then the threshold value is the median (resp. the mean) of the feature importances.
            A scaling factor (e.g., “1.25*mean”) may also be used. 
            If None and if the estimator has a parameter penalty set to l1, 
            either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5. 
            Otherwise, “mean” is used by default., by default None

        use_predict_proba : bool, optional
            If True, the predict_proba function is used for the predict function.
        """
        self.model = model
        self.threshold = threshold
        if fit is None:
            fit = getattr(self.model, "fit", None)
        if fit is None:
            raise NotImplemented(
                "The model does not have the method 'fit'. Please provide it.")
        self._fit = fit
        if predict is None:
            if use_predict_proba:
                if hasattr(self.model, "predict_proba"):
                    def predict(x): return getattr(self.model, "predict_proba")(x)[:, 1].flatten()
            else:
                predict = getattr(self.model, "predict", None)
        self._predict = predict
        self._set_attr("get_importances", get_importances)
        self._set_attr("get_support", get_support)

    def _set_attr(self, attr, value):
        """Set the attribute of the wrapper with the value provided or the attribute of the model.
        """
        if value is None:
            value = getattr(self.model, attr, None)
        if value is None:
            value = getattr(self, f"_{attr}", None)
        if value is not None:
            setattr(self, attr, value)
        else:
            raise NotImplemented(
                f"The model does not have the method '{attr}'. Please provide it.")

    def fit(self, X, y, **kwargs):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        self._fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """Predict using the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """
        if self._predict is None:
            raise NotImplemented(
                "The model does not have the method 'predict'. Please provide it.")
        res = self._predict(X, **kwargs)
        return res

    def _get_importances(self):
        """ Get the feature importances of the model and return the absolute value. """
        try:
            scores = _get_feature_importances(
                estimator=self.model.best_estimator_ if isinstance(
                    self.model, GridSearchCV
                ) else self.model, getter="auto", transform_func=None).flatten()
            res = np.abs(scores)
            return res
        except ValueError:
            raise NotImplemented(
                f"The model does not have the method 'get_importances' and there is no way to retrieve "
                f"the feature importances. Please provide it.")

    def _get_support(self, indices=False):
        """ Get the support of the model. """
        scores = self._get_importances()
        threshold = _calculate_threshold(self.model, scores, self.threshold)
        support = scores > threshold
        if indices:
            return np.where(support)[0]
        return support.astype(int)

