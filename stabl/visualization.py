import os
from copy import copy
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, pearsonr
from sklearn import metrics
from sklearn.metrics import roc_curve, r2_score, mean_squared_error, roc_auc_score, auc, \
    precision_recall_curve, mean_absolute_error
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from .utils import nonpartition_cross_val_predict, compute_CI

colors = ['#a8e6ce', '#dcedc2', '#ffd3b5', '#ffaaa6', '#ff8c94', '#e3819d', '#a188b7', '#487fad']
surge_palette = sns.color_palette(colors)


def plot_roc(
        y_true,
        y_preds,
        show_fig=True,
        show_CI=True,
        CI_level=0.95,
        export_file=False,
        path='./ROC Curve.pdf',
        **kwargs
):
    """
    Function to draw the ROC curve from predicted probabilities.

    Parameters
    ----------
    y_true: array-like
        True outcomes for each sample

    y_preds: array-like
        Predicted probabilities for each sample

    show_fig: bool, default=True
        Whether to display the figure

    show_CI: bool, default=True
        If set to True, the confidence interval will be displayed on the figure

    CI_level: float, default=0.95,
        The confidence interval level

    export_file: bool, default=False
        If set to True, the ROC curve will be exported using the path provided
        by the user.

    path: str or Path, default='./ROC curve.pdf'
        The path indicating were we want to save the figure. 
        Should also include the name of the file and the extension.

    **kwargs: additional parameters for the plot function
    """

    fig, ax = plt.subplots(1, 1, **kwargs)

    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    roc_auc = auc(fpr, tpr)

    df_CI, CI = compute_CI(
        y_true=y_true,
        y_preds=y_preds,
        scoring="roc_auc",
        confidence_level=CI_level,
        return_CI_predictions=True
    )

    if show_CI:
        fpr_low, tpr_low, _ = roc_curve(df_CI.target_low, df_CI.preds_low)
        ax.plot(fpr_low, tpr_low, lw=2, ls=":", alpha=.5, color='#e3819d')

        fpr_up, tpr_up, _ = roc_curve(df_CI.target_up, df_CI.preds_up)
        ax.plot(fpr_up, tpr_up, lw=2, ls=":", alpha=.5, color='#e3819d')

    ax.plot(
        fpr,
        tpr,
        lw=2,
        alpha=1,
        color='#C41E3A',
        label=f"ROC (AUC = {roc_auc:.3f} [{CI[0]:.3f}, {CI[1]:.3f}])"
    )
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='#4D4F53', alpha=.8, label="Chance")

    make_beautiful_axis(ax)
    lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", fontsize=8)

    if export_file:
        fig.savefig(path, dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')

    if not show_fig:
        plt.close()

    return fig, ax


def plot_prc(
        y_true,
        y_preds,
        show_fig=True,
        show_CI=True,
        show_iso=True,
        CI_level=0.95,
        export_file=False,
        path='./PR Curve.pdf',
        **kwargs
):
    """
    Function to draw the PR curve from probabilities.

    Parameters
    ----------
    y_true: array-like
        True outcomes for each samples

    y_preds: array-like
        Predicted probabilities for each samples
    
    show_fig : bool, default=True
        Whether to display the figure

    show_iso: bool, default=True,
        Whether to display the iso f1-score lines
        
    show_CI: bool, default=False
        If True the lower and upper CI PR curves will be displayed

    CI_level: float, default=0.95
        The confidence interval level

    export_file: bool, default=False
        If set to True, the ROC curve will be exported using the path provided
        by the user.

    path: str, default='./PR curve.pdf'
        The path indicating were we want to save the figure. 
        Should also include the name of the file and the extension.

    **kwargs: additional parameters for the plot function
    """
    fig, ax = plt.subplots(1, 1, **kwargs)

    precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
    prc_auc = auc(recall, precision)

    df_CI, CI = compute_CI(
        y_true=y_true,
        y_preds=y_preds,
        scoring="prc_auc",
        confidence_level=CI_level,
        return_CI_predictions=True
    )

    if show_iso:
        add_iso_lines(ax)

    if show_CI:
        precision_low, recall_low, _ = precision_recall_curve(
            df_CI.target_low,
            df_CI.preds_low
        )
        ax.plot(recall_low, precision_low, lw=2, ls=":", alpha=.8, color='#e3819d')

        precision_up, recall_up, _ = precision_recall_curve(
            df_CI.target_up,
            df_CI.preds_up
        )
        ax.plot(recall_up, precision_up, lw=2, ls=":", alpha=.8, color='#e3819d')

    ax.plot(
        recall,
        precision,
        lw=2,
        alpha=1,
        color='#C41E3A',
        label=f"PR (AUC = {prc_auc:.3f} [{CI[0]:.3f}, {CI[1]:.3f}])"
    )

    ax = make_beautiful_axis(ax, plot_type="prc")
    lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", fontsize=8)

    if export_file:
        fig.savefig(path, dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')

    if not show_fig:
        plt.close()

    return fig, ax


def boxplot_features(
        list_of_features,
        df_X,
        y,
        show_zero=True,
        show_fig=True,
        export_file=False,
        path='./',
        fmt='pdf'
):
    """Function to boxplot of the features indicated in a list
    given by the user.

    Parameters
    ----------
    list_of_features: list of str
        List of features we want to represent.
        Should be a list of str, each str being a feature name

    df_X : pd.DataFrame
        The input data DataFrame (not preprocessed)

    y: pd.Series
        y used in the fit. Should be a Series to get the name of the outcome.

    show_fig : bool, default=True
        Whether to display the figure

    show_zero: bool, default=True
        If set to true the boxplot will display the zero.

    export_file: bool, default=False
        If set to True, it will export the plot using the path and the format.
        The names of the different file are generated automatically with the name
        of the features.

    path: str or Path, default='./'
        path to the directory. Should not contain the name of the file and the
        extension

    fmt: str, default='pdf'
        Format of the export. Should be string
    """
    palette = ["#4D4F53", "#C41E3A"]

    for feature in list_of_features:
        fig, ax = plt.subplots(1, 1, figsize=(5, 10))

        sns.boxplot(
            ax=ax,
            y=df_X[feature],
            x=y,
            showfliers=False,
            palette=palette,
            boxprops=dict(alpha=.2),
            whiskerprops=dict(alpha=.2),
            width=.4
        )

        sns.stripplot(
            ax=ax,
            y=df_X[feature],
            x=y,
            palette=palette,
            alpha=.5,
            size=4,
            marker="D"
        )

        box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]

        num_patches = len(box_patches)
        lines_per_boxplot = len(ax.lines) // num_patches

        for i, patch in enumerate(box_patches):
            col = patch.get_facecolor()
            patch.set_edgecolor(col)
            patch.set_facecolor(col)

            for line in ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]:
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

        ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis="y")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if show_zero:
            if ax.get_ylim()[0] > 0:
                ax.set_ylim(bottom=0)

            if ax.get_ylim()[1] < 0:
                ax.set_ylim(top=0)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        ax.set_ylabel('')
        ax.set_title(feature, fontsize=5)

        if export_file:
            fig.savefig(Path(path, f"{feature}.{fmt}"), dpi=95)

        if not show_fig:
            plt.close()


def scatterplot_features(
        list_of_features,
        df_X,
        y,
        show_fig=True,
        export_file=False,
        path='./',
        fmt='pdf',
        **kwargs
):
    """Plot the scatter plot of the most stable features given a threshold
    We can also export them to pdf

    Parameters
    ----------
    list_of_features: list of string or int
        The list of features we want to represent the scatter plot for regression

    df_X : pd.DataFrame
        The input DataFrame (not preprocessed)

    y: Series
        y used in the fit. Should be a Series to get the name of the outcome.

    show_fig : bool, default=True
        Whether to display the figure

    export_file: bool, default=False
        If set to True, it will export the plot using the path and the format.
        The names of the different file are generated automatically with the name
        of the features.

    path: str or Path, default='./'
        path to the directory. Should not contain the name of the file and the
        extension

    fmt: str, default='pdf'
        Format of the export. Should be string
    """
    for col in list_of_features:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
        sns.scatterplot(ax=ax, y=df_X[col], color="#C41E3A", x=y, s=10, **kwargs)
        # fig.subplots_adjust(top=0.9)
        ax.set_ylabel('')
        ax.set_title(col, fontsize=5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()

        if export_file:
            fig.savefig(Path(path, f"{col}.{fmt}"), dpi=95)

        if not show_fig:
            plt.close()


def boxplot_binary_predictions(
        y_true,
        y_preds,
        show_fig=True,
        export_file=False,
        path='./Boxplot of predictions.pdf',
        figsize=(5.5, 3.5),
        classes=np.array([0, 1]),
        **kwargs
):
    """Function to plot the boxplot of the binary predictions.
    Also displays the mannwhitney u-test as the statistical test to test
    statistical significance.

    Parameters
    ----------
    classes
    figsize
    y_true: array-like, size=[n_repeats]
        The list of binary labels for each sample we predicted the probability

    y_preds : array-like, size=[n_repeats]
        List of predicted probabilities for each sample

    show_fig : bool, default=True
        Whether to display the figure

    export_file: bool, default=False
        If set to True, it will export the plot using the path and the format.
        The names of the different file are generated automatically with the name
        of the features. 

    path: str, default='./'
        Path to the directory. Should also contain the name of the file and the
        extension (format)

    **kwargs: arguments
        Further arguments to pass to the subplots function
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    palette = ["#4D4F53", "#C41E3A"]

    sns.boxplot(
        ax=ax,
        y=classes[y_true],
        x=y_preds,
        orient='h',
        color='.9',
        showfliers=False,
        palette=palette,
        boxprops=dict(alpha=.25),
        width=.5
    )

    sns.stripplot(ax=ax, y=classes[y_true], x=y_preds, orient='h', palette=palette, **kwargs)

    utest = mannwhitneyu(y_preds[y_true == 0], y_preds[y_true == 1])
    roc_auc = roc_auc_score(y_true, y_preds)
    precision, recall, _ = precision_recall_curve(y_true, y_preds)
    pr_auc = auc(recall, precision)
    plt.title(f"U-test pvalue = {utest.pvalue:.3e}\nAUCs (ROC={roc_auc:.3f}, PR={pr_auc:.3f})")

    plt.tight_layout()
    if export_file:
        fig.savefig(path, dpi=95, bbox_inches="tight")

    if not show_fig:
        plt.close()

    return fig, ax


def scatterplot_regression_predictions(
        y_true,
        y_preds,
        show_fig=True,
        export_file=False,
        path='./Scatterplot of predictions',
        **kwargs
):
    """Function to plot the scatterplot of the regression predictions.

    Parameters
    ----------
    y_true: array-like, size=[n_repeats]
        List of outcomes for each sample 

    y_preds : array-like, size=[n_repeats]
        List of prediction for each sample

    show_fig : bool, default=True
        Whether to display the figure

    export_file: bool, default=False
        If set to True, it will export the plot using the path and the format.
        The names of the different file are generated automatically with the name
        of the features. 

    path: str, default='./'
        Path to the directory. Should also contain the name of the file and the
        extension (format)

    **kwargs: arguments
    """
    fig, ax = plt.subplots(1, 1, **kwargs)
    sns.scatterplot(ax=ax, x=y_true, y=y_preds, color="#487fad", alpha=.9)
    p1 = max(max(y_preds), max(y_true))
    p2 = min(min(y_preds), min(y_true))
    ax.plot([p1, p2], [p1, p2], c='#e3819d', alpha=.9)

    r2 = r2_score(y_true=y_true, y_pred=y_preds)
    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_preds))
    mae = mean_absolute_error(y_true=y_true, y_pred=y_preds)
    pearson_stats, pearson_pvalue = pearsonr(y_true, y_preds)
    ax.set_title(
        f"Pearson r: score = {pearson_stats:.3f}, pvalue= {pearson_pvalue:.3e}\n"
        f"R2-score = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")

    plt.tight_layout()
    if export_file:
        fig.savefig(path, dpi=95)

    if not show_fig:
        plt.close()

    return fig, ax


def make_beautiful_axis(ax, plot_type="roc"):
    """
    Function to make the axis beautiful

    Parameters
    ----------
    ax: matplotlib.Axis

    plot_type:str, default="roc"
        - "roc": make axis beautiful for a ROC plot
        - "prc": make axis beautiful for a PRC plot

    Returns
    -------
    new_ax: matplotlib.Axis
    """

    ax.grid(which='major', color='#DDDDDD', linewidth=0.8, zorder=0)
    ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.7, zorder=0)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(ticks=[0, .25, .5, .75, 1.], labels=[0, .25, .5, .75, 1])
    ax.set_yticks(ticks=[0, .25, .5, .75, 1.], labels=[0, .25, .5, .75, 1])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(size=0, which="both", labelsize=12)

    if plot_type == "roc":
        ax.set_xlabel("1 - Specificity", size=12)
        ax.set_ylabel("Sensitivity", size=12)

    elif plot_type == "prc":
        ax.set_xlabel("Recall", size=12)
        ax.set_ylabel("Precision", size=12)

    return ax


def add_iso_lines(ax, iso_number=4):
    """
    Function to add iso lines in a Precision-Recall Curve plot

    Parameters
    ----------
    ax: matplotlib.Axis
        Axis where to plot the iso lines

    iso_number: int
        Number of iso lines

    Returns
    -------
    None
    """
    f_scores = np.linspace(0.2, 0.8, num=iso_number)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        ax.plot(x[y >= 0], y[y >= 0], color="#487fad", alpha=0.2, lw=1, zorder=0)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(y[45] + 0.02, 1.02), fontsize=7)


def _check_is_permutation(indices, n_samples):
    """Check whether indices are a reordering of the array np.arange(n_repeats)

    Parameters
    ----------
    indices : ndarray
        int array to test

    n_samples : int
        number of expected elements

    Returns
    -------
    is_partition : bool
        True iff sorted(indices) is np.arange(n)
    """
    if len(indices) != n_samples:
        return False
    hit = np.zeros(n_samples, dtype=bool)
    hit[indices] = True
    if not np.all(hit):
        return False
    return True
