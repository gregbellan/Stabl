from .visualization import boxplot_features, scatterplot_features
import os
from pathlib import Path
from warnings import warn
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from knockpy.knockoffs import GaussianSampler
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin, SelectFromModel
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import ParameterGrid,  GroupShuffleSplit, KFold
from sklearn.utils import safe_mask
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted
from sklearn.metrics import r2_score, roc_auc_score
from tqdm.autonotebook import tqdm
from .unionfind import UnionFind
import warnings

from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Distributions as dist
from julia import Bigsimr as bs

def ss_cv(X, y, stab_sel):
    stab_sel = clone(stab_sel)
    stab_sel.set_params(auto_ss=False)
    threshold_grid = np.arange(0.01, 1., 0.01)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    all_scores = []
    task_type = "classification" if len(np.unique(y)) <= 2 else "regression"

    linreg = LinearRegression(n_jobs=-1) if task_type == "regression" else LogisticRegression(penalty=None, class_weight="balanced", max_iter=int(1e6), random_state=stab_sel.random_state, n_jobs=-1)
    for train, test in splitter.split(X, y):
        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        stab_sel.fit(X_train, y_train)

        tmp_scores = []
        for t in threshold_grid:
            sel_features = list(stab_sel.get_support(indices=True, new_hard_threshold=t))

            if len(sel_features) > 0:
                X_train_red = X_train.iloc[:, sel_features]
                X_test_red = X_test.iloc[:, sel_features]

                linreg.fit(X_train_red, y_train)
                preds = linreg.predict(X_test_red) if task_type == "regression" else linreg.predict_proba(X_test_red)[:, 1]

            else:
                preds = [np.median(y_train) if task_type == "regression" else 0.5]*len(y_test)

            tmp_scores.append(r2_score(y_test, preds) if task_type == "regression" else roc_auc_score(y_test, preds))

        all_scores.append(tmp_scores)

    avg_scores = np.mean(all_scores, axis=0)
    best_threshold = threshold_grid[np.argmax(avg_scores)]

    return best_threshold


def classic_bootstrap(y, n_subsamples, replace=True, class_weight=None, rng=np.random.default_rng(None), **kwargs):
    """Function to create a bootstrap sample from the original dataset.
    Weights can be used to make some samples more likely to be selected.

    Parameters
    ----------
    y : array-like, shape(n_repeats, )
        The outcome array for classification or regression

    n_subsamples : int
        The number of subsamples indices returned by the bootstrap 

    replace : bool, default=True
        Whether to replace samples when bootstrapping

    class_weight: str or dict or None, default=None
        This is the sampling weights used in the bootstrap process
            - If None, no weights are used.
            - If 'balanced', the weights are automatically computed so that
            the weights are balanced and the probabilities of sampling different
            classes are adjusted.
            - Can also be a dictionary of this format {class1:value1, class2:value2}
            values are weights not probabilities. They will automatically be converted
            into probabilities.

    rng: np.random.default_rng, default=np.random.default_rng(None)
        RandomState generator

    Returns
    -------
    sampled_indices : array-like, shape(n_subsamples, )
        Sampled indices

    """
    n_samples = y.shape[0]

    if n_subsamples > n_samples and replace is False:
        raise ValueError("When `replace` is set to False, n_subsamples cannot be greater than the "
                         f"number of samples in the original dataset. Got `n_repeats`={n_samples} "
                         f"and `n_subsamples`={n_subsamples}")

    if class_weight is not None:
        samples_weight = compute_sample_weight(class_weight, y)
        sampling_probs = samples_weight / samples_weight.sum()

    else:
        sampling_probs = None

    sampled_indices = rng.choice(
        a=n_samples,
        size=n_subsamples,
        replace=replace,
        p=sampling_probs
    )

    # Handling the case of binary classification where we only select one class
    if len(np.unique(y[sampled_indices])) < 2:
        sampled_indices = classic_bootstrap(
            y,
            n_subsamples,
            replace=replace,
            class_weight=class_weight,
            rng=rng
        )

    return sampled_indices


def group_bootstrap(y, n_subsamples, groups, replace=False, rng=np.random.RandomState(None), **kwargs):
    """Function to create a bootstrap sample from the original dataset.
    Weights can be used to make some samples more likely to be selected.

    Parameters
    ----------
    y : array-like, shape(n_repeats, )
        The outcome array for classification or regression

    n_subsamples : int
        The number of subsamples indices returned by the bootstrap 

    replace : bool, default=True
        Whether to replace samples when bootstrapping

    class_weight: str or dict or None, default=None
        This is the sampling weights used in the bootstrap process
            - If None, no weights are used.
            - If 'balanced', the weights are automatically computed so that
            the weights are balanced and the probabilities of sampling different
            classes are adjusted.
            - Can also be a dictionary of this format {class1:value1, class2:value2}
            values are weights not probabilities. They will automatically be converted
            into probabilities.

    rng: np.random.default_rng, default=np.random.default_rng(None)
        RandomState generator

    Returns
    -------
    sampled_indices : array-like, shape(n_subsamples, )
        Sampled indices

    """
    n_samples = y.shape[0]

    if n_subsamples > n_samples and replace is False:
        raise ValueError("When `replace` is set to False, n_subsamples cannot be greater than the "
                         f"number of samples in the original dataset. Got `n_repeats`={n_samples} "
                         f"and `n_subsamples`={n_subsamples}")

    subsample_prop = n_subsamples / n_samples

    sampled_indices = GroupShuffleSplit(n_splits=1, train_size=subsample_prop, random_state=rng).split(y, groups=groups)
    sampled_indices = next(sampled_indices)[0]
    # Handling the case of binary classification where we only select one class
    if len(np.unique(y[sampled_indices])) < 2:
        sampled_indices = group_bootstrap(
            y,
            n_subsamples,
            groups=groups,
            replace=replace,
            rng=rng
        )

    return sampled_indices


def _bootstrap_generator(
        n_bootstraps,
        bootstrap_func,
        y,
        n_subsamples,
        replace,
        random_state=None,
        **kwargs
):
    """Function that creates bootstrapped indices, used in the Stabl process.
    The function returns a generator containing the indices for each bootstrap.

    Parameters
    ----------
    n_bootstraps: int
        Number of bootstraps for each value of the lambda parameter.

    bootstrap_func: python function
        The function use to draw the indices. 
        Should have at least the following parameters:
            - y: target array
            - n_subsamples: number of samples to draw from the original data set
            - replace: boolean indicating if we want to replace the samples 

    y: array-like, size(n_repeats, )
        Targets

    n_subsamples: int
        number of samples to draw from the original data set

    replace: bool
        If set to True, the bootstrap will be done such that the samples are 
        replaced during the process.

    random_state: int,
        Random state for reproducibility.

    **kwargs: arguments
        Further arguments we want to pass to bootstrap_func.
    """
    rng = np.random.RandomState(random_state)
    for _ in range(n_bootstraps):

        # Generating the bootstrapped indices
        subsample = bootstrap_func(
            y=y,
            n_subsamples=n_subsamples,
            replace=replace,
            rng=rng,
            **kwargs
        )

        if isinstance(subsample, tuple):
            for item in subsample:
                yield item
        else:
            yield subsample


def export_stabl_to_csv(stabl, path):
    """
    Export Stabl scores to csv. They can later be used to plot the stabl path again.

    Parameters
    ----------
    stabl: Stabl
        Fitted Stabl instance.

    path: str or Path
        The path where csv files will be saved

    Returns
    -------
    None
    """

    check_is_fitted(stabl, 'stabl_scores_')

    if hasattr(stabl, 'feature_names_in_'):
        X_columns = stabl.feature_names_in_
    else:
        X_columns = [f'x.{i + 1}' for i in range(stabl.n_features_in_)]

    columns = list(ParameterGrid(stabl.lambda_grid))

    df_real = pd.DataFrame(data=stabl.stabl_scores_,
                           index=X_columns, columns=columns)
    df_real.to_csv(Path(path, 'STABL scores.csv'))

    df_max_probs = pd.DataFrame(
        data={"Max Proba": stabl.stabl_scores_.max(axis=1)},
        index=X_columns
    )
    df_max_probs = df_max_probs.sort_values(by='Max Proba', ascending=False)
    df_max_probs.to_csv(Path(path, 'Max STABL scores.csv'))

    if stabl.artificial_type is not None:
        synthetic_index = [
            f'artificial.{i + 1}' for i in range(stabl.X_artificial_.shape[1])]

        df_noise = pd.DataFrame(
            data=stabl.stabl_scores_artificial_,
            index=synthetic_index,
            columns=columns
        )
        df_noise.to_csv(Path(path, 'STABL artificial scores.csv'))

        df_max_probs_noise = pd.DataFrame(
            data={"Max Proba": stabl.stabl_scores_artificial_.max(axis=1)},
            index=synthetic_index
        )
        df_max_probs_noise = df_max_probs_noise.sort_values(
            by='Max Proba', ascending=False)
        df_max_probs_noise.to_csv(
            Path(path, 'Max STABL artificial scores.csv'))


def plot_fdr_graph(
        stabl,
        show_fig=True,
        export_file=False,
        path='./FDR estimate graph.pdf',
        figsize=(8, 4)
):
    """
    Plots the FDR graph.
    The user can also export it to pdf of other format

    Parameters
    ----------
    stabl : Stabl
        Fitted Stabl instance.

    show_fig : bool, default=True
        Whether to display the figure

    export_file: bool, default=False
        If set to True, it will export the plot using the path

    path: str or Path
        Should be the string of the path/name. Use name of the file plus extension

    figsize: tuple
        Size of the Stabl fdr graph

    Returns
    -------
    figure, axis
    """

    check_is_fitted(stabl, 'stabl_scores_')

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    thresh_grid = stabl.fdr_threshold_range

    ax.plot(thresh_grid, stabl.FDRs_, color="#4D4F53",
            label='FDR estimate', lw=2)

    if stabl.min_fdr_ > 1:
        optimal_threshold = 1.
        label = "No optimal threshold minimizing the FDR estimate"
    else:
        optimal_threshold = thresh_grid[np.argmin(stabl.FDRs_)]
        label = f"Optimal threshold={optimal_threshold:.2f}"

    ax.axvline(optimal_threshold, ls='--', lw=1.5,
               color="#C41E3A", label=label)
    ax.set_xlabel('Threshold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1))
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()

    if export_file:
        fig.savefig(path, dpi=95)

    if not show_fig:
        plt.close()

    return fig, ax


def plot_stabl_path(
        stabl,
        new_hard_threshold=None,
        show_fig=True,
        export_file=False,
        path='./Stabl path.pdf',
        figsize=(4, 8)
):
    """Plots Stabl path.
    The user can also export it to pdf or any other format

    Parameters
    ----------
    stabl: Stabl
        Fitted Stabl instance.

    new_hard_threshold: float or None, default=None
        Threshold defining the minimum cutoff value for the
        stabl scores. This is a hard threshold: FDR control
        will be ignored if this is not None.

    show_fig: bool, default=True
        Whether to display the figure

    export_file: bool
        If set to True, it will export the plot using the path

    path: str or Path
        Should be the string of the path/name. Use name of the file plus extension

    figsize: tuple
        Size of the STABL path

    Returns
    -------
    figure, axis
    """

    check_is_fitted(stabl, 'stabl_scores_')

    threshold = stabl.hard_threshold if new_hard_threshold is None else new_hard_threshold

    if isinstance(threshold, float) and not (0.0 < threshold <= 1):
        raise ValueError(
            f'If new_hard_threshold is set, it must be a float in (0, 1], got {threshold}')

    paths_to_highlight = stabl.get_support(new_hard_threshold=threshold)

    x_grid_list = []
    x_padding_list = []
    order_list = []
    different_params = stabl.get_different_parameters()
    nb_different_params = len(different_params)
    if nb_different_params <= 1:
        if 'alpha' in stabl.lambda_grid:
            x_grid_tmp = np.min(stabl.lambda_grid["alpha"]) / stabl.lambda_grid["alpha"]
            x_padding = 0
            order_list = [np.arange(len(stabl.lambda_grid["alpha"]))]
            x_grid_list = [x_grid_tmp]
            x_padding_list = [0]
        elif 'C' in stabl.lambda_grid:
            x_grid_tmp = stabl.lambda_grid["C"] / np.max(stabl.lambda_grid["C"])
            x_padding = 0
            order_list = [np.arange(len(stabl.lambda_grid["C"]))]
            x_grid_list = [x_grid_tmp]
            x_padding_list = [0]
    elif nb_different_params == 2:
        params = list(ParameterGrid(stabl.lambda_grid))
        ordered_params = dict()
        for i, k in enumerate(params):
            l1_ratio = k["l1_ratio"]
            penalty = k["alpha"] if "alpha" in k else k["C"]
            if l1_ratio in ordered_params:
                order = ordered_params[l1_ratio][0]
                x_grid = ordered_params[l1_ratio][1]
            else:
                order = []
                x_grid = []
            order.append(i)
            x_grid.append(penalty)
            ordered_params[l1_ratio] = (order, x_grid)
        figsize = (figsize[0] * len(ordered_params.keys()), figsize[1])
        x_padding = 0
        for l1_ratio in sorted(ordered_params.keys()):
            order = np.array(ordered_params[l1_ratio][0])
            penalties = np.array(ordered_params[l1_ratio][1])
            if "alpha" in different_params:
                x_grid = np.min(penalties) / penalties
            elif "C" in different_params:
                x_grid = penalties / np.max(penalties)
            x_padding += np.max(x_grid) - np.min(x_grid) + 1e-5
            x_grid_list.append(x_grid)
            order_list.append(order)
            x_padding_list.append(x_padding)
    else:
        warnings.warn("Cannot plot the STABL path for more than 2 parameters")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i, o in enumerate(order_list):
        x_grid = x_grid_list[i]
        x_padding = x_padding_list[i]
        x_grid += x_padding
        if not paths_to_highlight.all():
            ax.plot(
                x_grid,
                stabl.stabl_scores_[~paths_to_highlight][:, o].T,
                alpha=1,
                lw=1.5,
                color="#4D4F53",
                label="Noisy features"
            )

        if paths_to_highlight.any():
            ax.plot(
                x_grid,
                stabl.stabl_scores_[paths_to_highlight][:, o].T,
                alpha=1,
                lw=2,
                color="#C41E3A",
                label="Stable features"
            )

        if threshold is not None:
            ax.plot(
                x_grid,
                threshold * np.ones(len(x_grid)),
                c="black",
                ls="--",
                label=f"Hard threshold={threshold: .2f}"
            )

        elif stabl.artificial_type is not None:
            ax.plot(
                x_grid,
                stabl.stabl_scores_artificial_[:, o].T,
                color="gray",
                ls=":",
                alpha=.4,
                lw=1,
                label="Artificial features"
            )

            ax.plot(
                x_grid,
                stabl.fdr_min_threshold_ * np.ones(len(x_grid)),
                c="black",
                ls="--",
                label=f"FDP+ threshold={stabl.fdr_min_threshold_: .2f}"
            )

        if stabl.explore_threshold is not None:
            ax.plot(
                x_grid,
                stabl.explore_threshold * np.ones(len(x_grid)),
                c="#487fad",
                ls="--",
                label=f"Explore threshold={stabl.explore_threshold: .2f}"
            )
        if i != len(order_list) - 1:
            x_vert_gray = np.max(x_grid)
            ax.axvline(x=x_vert_gray, c="gray", ls="--", lw=1)

    ax.tick_params(left=True, right=False, labelleft=True,
                   labelbottom=False, bottom=False)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(f"Frequency of selection")
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1))

    fig.tight_layout()

    if export_file:
        fig.savefig(path, dpi=95)

    if not show_fig:
        plt.close()

    return fig, ax


def save_stabl_results(
        stabl,
        path,
        df_X,
        y,
        figure_fmt='pdf',
        new_hard_threshold=None,
        task_type="binary"
):
    """
    Function to automatically save all the results of a Stabl fitted instance.
    The user must define the input DataFrame and the output to plot the stable
    features.

    Parameters
    ----------
    stabl: Stabl
        Must be a fitted Stabl object.

    path: Path or str
        The path where to save the results. If the path already exists an error will be raised

    figure_fmt: str
        Format of the figures.

    df_X: pd.DataFrame, shape=(n_repeats, n_features)
        input DataFrame

    y: pd.Series, shape=(n_repeats)
        Series of output

    new_hard_threshold: float or None, default=None
        Threshold defining the minimum cutoff value for the
        stability scores. This is a hard threshold: FDR control
        will be ignored if this is not None

    task_type: str, default="binary"
        Type of performed task.
        Choose "binary" for binary classification and "regression" for regression tasks or "multiclass"
    """

    check_is_fitted(stabl)

    path = Path(path, '')

    try:
        os.makedirs(path)
    except FileExistsError:
        raise FileExistsError(f"Folder with path={path} already exists.")

    # Saving the stability scores
    export_stabl_to_csv(stabl=stabl, path=path)

    if stabl.artificial_type is not None:
        plot_fdr_graph(
            stabl=stabl,
            show_fig=False,
            export_file=True,
            path=Path(path, f'FDR Graph.{figure_fmt}'),
            figsize=(8, 4)
        )

    plot_stabl_path(
        stabl=stabl,
        new_hard_threshold=new_hard_threshold,
        show_fig=False,
        export_file=True,
        path=Path(path, f'Stability Path.{figure_fmt}'),
        figsize=(4, 8)
    )

    selected_features = stabl.get_feature_names_out(
        new_hard_threshold=new_hard_threshold)

    nb_selected_features = len(selected_features)
    df_selected_features = pd.DataFrame(
        data={"Feature Name": selected_features},
        index=[f"Feature n°{i + 1}" for i in range(nb_selected_features)]
    )

    Path(path, 'Selected Features').mkdir(parents=True)
    df_selected_features.to_csv(
        Path(path, "Selected Features", "Selected features.csv"))

    if task_type in ["binary", "multiclass"]:
        boxplot_features(
            list_of_features=selected_features,
            df_X=df_X,
            y=y,
            categorical_features=6,
            show_fig=False,
            export_file=True,
            path=Path(path, 'Selected Features'),
            fmt=figure_fmt
        )

    elif task_type == "regression":
        scatterplot_features(
            list_of_features=selected_features,
            df_X=df_X,
            y=y,
            categorical_features=6,
            show_fig=False,
            export_file=True,
            path=Path(path, 'Selected Features'),
            fmt=figure_fmt
        )


def fit_bootstrapped_sample(
        base_estimator,
        X,
        y,
        lambda_val,
        corr_groups=None,
        threshold=None
):
    """
    Fits base_estimator on a bootstrap sample of the original data,
    and returns a mas of the variables that are selected by the fitted model.

    Parameters
    ----------
    base_estimator: estimator
        This is the estimator to be fitted on the data

    X: {array-like, sparse matrix}, shape = [n_repeats, n_features]
        The training input samples.

    y: array-like, shape = [n_repeats]
        The target values.

    lambda_val: dict
        Name of the penalization parameter of base_estimator

    lambda_value: float
        Value of the penalization parameter

    threshold: string or float, default=None
        The hard_threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``hard_threshold`` value is
        the median (resp. the mean) of the feature importance. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the hard_threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    Returns
    -------
    selected_variables: array-like, shape=(n_features, )
        Boolean mask of the selected variables.
    """
    base_estimator.set_params(**lambda_val)
    if hasattr(base_estimator, "groups"):
        base_estimator.set_params(groups=corr_groups)
    base_estimator.fit(X, y)

    features_selection = SelectFromModel(
        estimator=base_estimator,
        threshold=threshold,
        prefit=True
    )

    return features_selection.get_support()


class Stabl(SelectorMixin, BaseEstimator):
    """In a STABL process, the estimator `base_estimator` is fitted
    several time on bootstrap samples of the original data set, for different values of
    the regularization parameter for `base_estimator`. Features that
    get selected significantly by the model in these bootstrap samples are
    considered to be stable variables. This implementation also allows the user
    to use synthetic features to automatically set the hard_threshold of selection by
    FDR control.

    Parameters
    ----------
    base_estimator: sklearn.base_estimator, default=LogisticRegression
        The base estimator used for stability selection. The estimator
        must have either a ``feature_importances_`` or ``coef_``
        attribute after fitting.

    lambda_name: str, default='C'
        The name of the penalization parameter for the estimator
        `base_estimator`. Example for LogisticRegression in scikit-learn: 'C'

    lambda_grid: array-like, default=np.linspace(0.01, 1, 30)
        Grid of values for the penalization parameter to iterate over.

    n_bootstraps: int, default=1000
        Number of bootstrap iterations for each value of lambda.

    artificial_type: str or None
        If str can either be "random_permutation" or "knockoff"
        If None, we do not inject artificial features, the user must therefore define an arbitrary hard_threshold.
        When the artificial_type is none, we fall back into the classic stability selection process.

    artificial_proportion: float
        The proportion of artificial features to

    sample_fraction: float, default=0.5
        The fraction of samples to be used in each bootstrap sample.
        Can be greater than 1 if we replace in the boostrap technique.

    hard_threshold: float, default=None
        Threshold defining the cutoff value for the stability selection.
        If the hard_threshold is defined, the FDRc will be bypassed.
        The default value is None: the user must set a value if no random permutation/knockoff is used.

    fdr_threshold_range: array-like, default=np.arange(0.3, 1, 0.01)
        When using random permutation or knockoff features, the user can change the tested values for the hard_threshold
        For each value, the FDRc will be computed.

    explore : bool, default=False
        If True

    cluster : int, default=None
        Number of clusters to use for the stability selection. If None, no groups are calculated.

    bootstrap_threshold: string or float, default=None
        The hard_threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``hard_threshold`` value is
        the median (resp. the mean) of the feature importance. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the hard_threshold used is 1e-5.
        Otherwise, "mean" is used by default.

    verbose: int, default=0
        Controls the verbosity: the higher, the more messages.

    n_jobs: int, default=-1
        Number of jobs to run in parallel.

    random_state: int or None, default=None
        Random state for reproducibility matters.

    Attributes
    ----------
    n_features_in_: int
        Number of features seen during fit.

    feature_names_in_: ndarray of shape (n_features_in_,)
        Names of features seen during fit. Defined only when X has feature names that are all strings.

    stabl_scores_: array, shape(n_features, n_alphas)
        Array of stability scores for each feature and for each value of the
        penalization parameter.

    stabl_scores_artificial_: array, shape(n_features, n_alphas)
        Array of stability scores for each decoy/knockoff feature and for each value of the
        penalization parameter. Can only be accessed if we used decoy or knockoff in the
        training.

    X_artificial_: array, shape(n_repeats, n_features)
        Array of synthetic features. Can only be returned if we used decoy or knockoffs in the
        training.

    FDRs_: array
        The array of False Discovery Rates.
        Can only be retrieved if we used decoy or knockoffs in the training

    min_fdr_: float
        The Smallest FDR achieved
        Can only be retrieved if we used decoy or knockoffs in the training

    fdr_min_threshold_: float
        The hard_threshold achieving the desired FDR. Can only be retrieved if we used decoy or knockoff
        in the training and if no hard hard_threshold where defined.
    """

    def __init__(
            self,
            base_estimator=LogisticRegression(
                penalty='l1',
                solver='liblinear',
                class_weight='balanced',
                max_iter=int(1e6),
                random_state=42
            ),
            lambda_grid={"C": np.linspace(0.01, 1, 30)},
            n_bootstraps=1000,
            artificial_type="random_permutation",
            artificial_proportion=1.,
            sample_fraction=0.5,
            replace=False,
            hard_threshold=None,
            fdr_threshold_range=list(np.arange(0., 1., .01)),
            explore=False,
            n_explore=5,
            bootstrap_func=classic_bootstrap,
            sample_weight_bootstrap=None,
            bootstrap_threshold=1e-5,
            perc_corr_group_threshold=None,
            sgl_groups=None,
            # cluster=None,
            verbose=0,
            n_jobs=-1,
            random_state=None,
            feat_type=None,
            corr=None,
            auto_ss=False
    ):
        self.base_estimator = base_estimator
        self.lambda_grid = lambda_grid
        self.n_bootstraps = n_bootstraps
        self.artificial_type = artificial_type
        self.artificial_proportion = artificial_proportion
        self.sample_fraction = sample_fraction
        self.hard_threshold = hard_threshold
        self.fdr_threshold_range = fdr_threshold_range
        self.explore = explore
        self.n_explore = n_explore
        self.bootstrap_func = bootstrap_func
        self.sample_weight_bootstrap = sample_weight_bootstrap
        self.bootstrap_threshold = bootstrap_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.replace = replace
        self.random_state = random_state
        self.noise_group = np.array([])
        self.perc_corr_group_threshold = perc_corr_group_threshold
        self.sgl_groups = sgl_groups
        self.stabl_scores_ = None
        self.stabl_scores_artificial_ = None
        self.FDRs_ = None
        self.min_fdr_ = None
        self.fdr_min_threshold_ = None
        self.explore_threshold = None
        self.feat_type = feat_type
        self.corr = corr
        self.auto_ss = auto_ss

    def _validate_input(self):
        if not isinstance(self.n_bootstraps, int) or self.n_bootstraps <= 0:
            raise ValueError(
                f'n_bootstraps should be a positive integer, got {self.n_bootstraps}')

        if not isinstance(self.sample_fraction, float) or not (0.0 < self.sample_fraction):
            raise ValueError(
                f'sample_fraction should be a float in (0, 1], got {self.sample_fraction}')

        if isinstance(self.hard_threshold, float) and not (0.0 < self.hard_threshold <= 1):
            raise ValueError(
                f'If hard_threshold is set, it must be a float in (0, 1], got {self.hard_threshold}')

        if self.hard_threshold is None and self.artificial_type is None:
            raise ValueError(
                f'When not using synthetic features (random permutations, knockoff or gaussian noise), '
                f'the user must define a hard_threshold of selection, got {self.hard_threshold}'
            )

        if self.artificial_type is not None and not (0.0 < self.artificial_proportion <= 1.):
            raise ValueError(
                f"When injecting noise, the noise proportion must be between 0 and 1, "
                f"got {self.artificial_proportion}"
            )

    def _make_groups(self, X):

        nb_real = X.shape[1] - self.noise_group.shape[0]
        X_real = X[:, :nb_real]

        n = X_real.shape[1]

        if self.perc_corr_group_threshold is not None:
            u = UnionFind(elements=range(X.shape[1]))
            corr_mat = pd.DataFrame(X_real).corr().values
            corr_val = corr_mat[np.triu_indices_from(corr_mat, k=1)]
            threshold = np.percentile(corr_val, self.perc_corr_group_threshold) - 0.1

            for i in np.arange(n):
                for j in np.arange(n):
                    if corr_mat[i, j] > threshold:
                        u.union(i, j)
            for idx, i in enumerate(self.noise_group):
                u.union(i, idx + n)

            return list(map(np.array, map(list, u.components())))

        elif self.sgl_groups is not None:
            ng = self.noise_group
            g = [np.concatenate((np.arange(i, i+5), n+ng[i:i+5])) for i in np.arange(0, 996, 5)]
            return g

    def fit(self, X, y, groups=None):
        """Fit the stability selection model on the given data.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_repeats, n_features)
            The training input samples.
        y : array-like, shape=(n_repeats, )
            The target values.
        """
        self._validate_input()
        X_old, y_old = X, y

        X_, y = self._validate_data(
            X=X,
            y=y,
            reset=True,
            validate_separately=False
        )
        X = pd.DataFrame(X_, index=X.index, columns=X.columns)

        n_samples, n_features = X.shape
        n_subsamples = int(np.floor(self.sample_fraction * n_samples))
        param_grid = list(ParameterGrid(self.lambda_grid))
        n_lambdas = len(param_grid)

        # Defining the number of injected noisy features
        n_injected_noise = int(X.shape[1] * self.artificial_proportion)

        base_estimator = clone(self.base_estimator)

        # Initializing the Stabl scores
        self.stabl_scores_ = np.zeros((n_features, n_lambdas))

        # __Synthetic features and coefs__
        if self.artificial_type is not None:
            # Only initialize those score if we use artificial features
            self.stabl_scores_artificial_ = np.zeros(
                (n_injected_noise, n_lambdas))
            X_ = self._make_artificial_features(
                X=X,
                nb_noise=n_injected_noise,
                artificial_type=self.artificial_type,
                random_state=self.random_state
            )
            X = X_
        X = np.array(X)
        corr_groups = None
        if self.perc_corr_group_threshold is not None or self.sgl_groups is not None:
            corr_groups = self._make_groups(X)

        # --Loop--
        leave = (self.verbose > 0)
        for idx, lambda_val in tqdm(
                enumerate(param_grid),
                'Stabl progress',
                total=n_lambdas,
                colour='#001A7B',
                leave=leave,
                file=sys.stdout,
                disable=(not leave)
        ):

            # Generating the bootstrap indices
            bootstrap_indices = _bootstrap_generator(
                n_bootstraps=self.n_bootstraps,
                bootstrap_func=self.bootstrap_func,
                y=y,
                n_subsamples=n_subsamples,
                replace=self.replace,
                groups=groups,
                class_weight=self.sample_weight_bootstrap,
                random_state=self.random_state
            )

            # Computing the frequencies
            selected_variables = Parallel(
                n_jobs=self.n_jobs,
                verbose=0,
                pre_dispatch='2*n_jobs'
            )(delayed(fit_bootstrapped_sample)(
                clone(base_estimator),
                X=X[safe_mask(X, subsample_indices), :],
                y=y[subsample_indices],
                corr_groups=corr_groups,
                lambda_val=lambda_val,
                threshold=self.bootstrap_threshold
            )
                for subsample_indices in bootstrap_indices
            )

            if self.artificial_type is not None:
                self.stabl_scores_artificial_[:, idx] = np.vstack(
                    selected_variables)[:, n_features:].mean(axis=0)
            self.stabl_scores_[:, idx] = np.vstack(selected_variables)[
                :, :n_features].mean(axis=0)

        if self.artificial_type is not None:
            self._compute_FDRc()

        if self.auto_ss:
            self.hard_threshold = ss_cv(X_old, y_old, self)

        return self

    def get_support(self, indices=False, new_hard_threshold=None):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        new_hard_threshold: float or None, default=None
            Threshold defining the minimum cutoff value for the
            stability scores. This is a hard hard_threshold: FDR control
            will be ignored if this is not None

        Returns
        -------
        support : array-like
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self._get_support_mask(new_hard_threshold=new_hard_threshold)
        return mask if not indices else np.where(mask)[0]

    def get_feature_names_out(self, input_features=None, new_hard_threshold=None):
        """Mask feature names according to selected features.

        Parameters
        ----------
        new_hard_threshold: float or None, default=None
            Threshold defining the minimum cutoff value for the
            stability scores. This is a hard threshold: FDR control
            will be ignored if this is not None

        input_features : array-like of str or None, default=None
            Input features.
            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        input_features = _check_feature_names_in(self, input_features)
        return input_features[self.get_support(new_hard_threshold=new_hard_threshold)]

    def transform(self, X, new_hard_threshold=None):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape=(n_repeats, n_features)
            The input array.

        new_hard_threshold: float or None, default=None
            Threshold defining the cutoff value for the
            stabl scores.
            When None the value set during the instantiation will be used.

        Returns
        -------
        X_out : array of shape=(n_repeats, n_selected_features)
            The input samples with only the selected features.
        """
        X = self._validate_data(X, reset=False)

        mask = self.get_support(
            indices=False, new_hard_threshold=new_hard_threshold)

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        if not mask.any():
            warn("No features were selected: either the data is"
                 " too noisy or the selection test too strict.",
                 UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))

        return X[:, safe_mask(X, mask)]

    def get_importances(self):
        """Get the feature importances (stability scores)

        Returns
        -------
        numpy.ndarray of shape (n_features_in_,)
            Feature importances, computed as the max of the stability scores
        """
        check_is_fitted(self, 'stabl_scores_')
        return np.max(self.stabl_scores_, axis=1)

    def _get_support_mask(self, new_hard_threshold=None):
        """Get a mask, or integer index, of the features selected

        Parameters
        ----------
        new_hard_threshold: float or None, default=None
            Threshold defining the cutoff value for the
            stabl scores.
            When None the value set during the instantiation will be used.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            This is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. 
        """
        check_is_fitted(self, 'stabl_scores_')

        new_threshold = self.hard_threshold if new_hard_threshold is None else new_hard_threshold

        if new_threshold is None:
            final_cutoff = self.fdr_min_threshold_
        else:
            final_cutoff = new_threshold

        max_scores = np.max(self.stabl_scores_, axis=1)
        mask = max_scores > final_cutoff

        if np.sum(mask) == 0 and self.explore is True:
            final_cutoff = np.sort(max_scores)[-self.n_explore] - 0.01
            self.explore_threshold = final_cutoff
            mask = max_scores > final_cutoff

        else:
            self.explore_threshold = None

        return mask

    def _make_artificial_features(self, X, artificial_type, nb_noise, random_state=None):
        """
        Function generating the artificial features before the bootstrap process begins.
        The artificial features will be concatenated to the original dataset.

        Parameters
        ----------
        X : array-like, size=(n_repeats, n_features)
            The input array.

        artificial_type: str
            The type of artificial features to generate
            Can either be "random_permutation" or "knockoff"

        nb_noise: int
            Number of artificial features to generate

        Returns
        -------
        X_out : array-like, size=(n_repeats, n_features + n_artificial_features)
            The input array concatenated with the artificial features
        """

        if artificial_type == "random_permutation":
            rng = np.random.default_rng(seed=random_state)
            X_artificial = np.array(X).copy()
            indices = rng.choice(
                a=X_artificial.shape[1], size=nb_noise, replace=False)
            self.noise_group = indices
            X_artificial = X_artificial[:, indices]

            for i in range(X_artificial.shape[1]):
                rng.shuffle(X_artificial[:, i])

        elif artificial_type == "knockoff":

            def generate_noise(X_a):
                corr = self.corr
                feat_type = self.feat_type
                # print(f"\n Generate noise : {feat_type} {corr}\n")
                if feat_type is None or feat_type == "normal":
                    return GaussianSampler(np.array(X_a), method='equicorrelated').sample_knockoffs()

                X_b = pd.read_csv(f"./Norta/normal {corr}.csv", index_col=0).loc[X_a.index, X_a.columns]
                X_b = (X_b - X_b.mean())/X_b.std()
                X_artificial = GaussianSampler(np.array(X_b), method='equicorrelated').sample_knockoffs()
                if "NB" in feat_type:
                    margin = dist.NegativeBinomial(2, 0.1)
                    for i in range(X_artificial.shape[1]):
                        X_artificial[:, i] = bs.normal_to_margin(margin, X_artificial[:, i])

                if "ZI" in feat_type:
                    indices = rng.choice(X_artificial.size, X_artificial.size // 5, replace=False)
                    old_shape = X_artificial.shape
                    X_artificial = X_artificial.reshape(-1)
                    X_artificial[indices] = 0
                    X_artificial = X_artificial.reshape(old_shape)

                X_artificial = (X_artificial - np.mean(X_artificial, axis=0))/np.std(X_artificial, axis=0)
                return np.array(X_artificial)

            np.random.seed(random_state)
            rng = np.random.default_rng(seed=random_state)
            n_features = X.shape[1]
            if n_features > 3000:
                initial_shape = (X.shape[0], (X.shape[1]//3000 + 1) * 3000)
                X_artificial = np.empty(initial_shape)
                for i in range(X.shape[1]//3000 + 1):
                    cols = rng.choice(a=X.shape[1], size=3000, replace=False)
                    X_tmp = X.iloc[:, cols].copy()
                    X_art_tmp = generate_noise(X_tmp)
                    X_artificial[:, i*3000: (i+1)*3000] = X_art_tmp
                X_artificial = X_artificial[:, rng.choice(a=X_artificial.shape[1], size=X.shape[1], replace=False)]

            else:
                X_artificial = generate_noise(X.copy())
            indices = rng.choice(a=X_artificial.shape[1], size=nb_noise, replace=False)
            self.noise_group = indices
            X_artificial = X_artificial[:, indices]

        else:
            raise ValueError("The type of artificial feature must be in ['random_permutation', 'knockoff']."
                             f" Got {artificial_type}")

        self.X_artificial_ = X_artificial

        return np.concatenate([np.array(X), X_artificial], axis=1)

    def _compute_FDRc(self):
        """Function that computes the FDRc at each value of the `thresholds_grid`.
        Also compute the threshold minimizing the FDRc.
        """

        FDPs = []  # Initializing false discovery proportions
        artificial_proportion = self.artificial_proportion
        max_scores_artificial = np.max(self.stabl_scores_artificial_, axis=1)
        max_scores = np.max(self.stabl_scores_, axis=1)

        for thresh in self.fdr_threshold_range:
            num = np.sum((1 / artificial_proportion) *
                         (max_scores_artificial > thresh)) + 1
            denum = max([1, np.sum((max_scores > thresh))])
            FDP = num / denum
            FDPs.append(FDP)

        self.FDRs_ = FDPs
        self.min_fdr_ = np.min(FDPs)

        if self.min_fdr_ > 1.:
            final_cutoff = 1.
        else:
            final_cutoff = np.min(
                [self.fdr_threshold_range[np.argmin(self.FDRs_)], 1])

        self.fdr_min_threshold_ = final_cutoff

    def get_different_parameters(self):
        keys = set()
        for p in ParameterGrid(self.lambda_grid):
            keys.update(p.keys())
        return list(keys)
