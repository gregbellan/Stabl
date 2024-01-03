from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu, pearsonr
from sklearn.metrics import roc_curve, r2_score, mean_squared_error, roc_auc_score, auc, \
    precision_recall_curve, mean_absolute_error
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression

from .utils import compute_CI

colors = ['#a8e6ce', '#dcedc2', '#ffd3b5', '#ffaaa6',
          '#ff8c94', '#e3819d', '#a188b7', '#487fad']
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
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, name="outcomes")

    if not isinstance(y_preds, pd.Series):
        y_preds = pd.Series(y_preds, name="predictions")

    fig, ax = plt.subplots(1, 1, **kwargs)

    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    roc_auc = auc(fpr, tpr)

    # Computing the confidence interval
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
    Function to draw the Precision-Recall curve from predicted probabilities and corresponding true outcomes.

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

    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, name="outcomes")

    if not isinstance(y_preds, pd.Series):
        y_preds = pd.Series(y_preds, name="predictions")

    # Computing the AUPRC
    precision, recall, thresholds = precision_recall_curve(y_true, y_preds)
    prc_auc = auc(recall, precision)

    # Computing the Confidence Interval
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
        precision_low, recall_low, _ = precision_recall_curve(df_CI.target_low, df_CI.preds_low)
        ax.plot(recall_low, precision_low, lw=2, ls=":", alpha=.8, color='#e3819d')

        precision_up, recall_up, _ = precision_recall_curve(df_CI.target_up, df_CI.preds_up)
        ax.plot(recall_up, precision_up, lw=2, ls=":", alpha=.8, color='#e3819d')

    ax.plot(
        recall,
        precision,
        lw=2,
        alpha=1,
        color='#C41E3A',
        label=f"AUPRC={prc_auc:.2f} [{CI[0]:.2f}, {CI[1]:.2f}]"
    )

    ax = make_beautiful_axis(ax, plot_type="prc")  # Make beautiful axes for this plot

    lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower center", fontsize=8)

    if export_file:
        fig.savefig(path, dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')

    if not show_fig:
        plt.close()

    return fig, ax


def boxplot_features(
        features,
        df_X,
        y,
        categorical_features=10,
        cmap='viridis',
        show_zero=True,
        show_fig=True,
        export_file=False,
        path='./',
        fmt='pdf'
):
    """Function to draw the boxplot of the features indicated in the `features` list
    given by the user.

    Parameters
    ----------
    features: list of str
        List of features we want to represent.
        Should be a list of str, each str being a feature name

    df_X : pd.DataFrame
        The input data DataFrame (not preprocessed)

    y: pd.Series
        y used in the fit. Should be a Series to get the name of the outcome.

    categorical_features: list of str or int, default=None
        List of the features that are categorical or upper bound number of unique values to be considered as categorical
        If None, no features are categorical.

    cmap: str, default='viridis'
        The name of the colormap to use.

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

    for feature in features:

        if _is_categorical(df_X, feature, categorical_features):
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            feat = df_X[feature]
            df_plot = pd.DataFrame(columns=np.unique(np.sort(feat.unique())))
            for v in np.unique(y):
                df_plot.loc[v] = feat[y == v].value_counts()
            df_plot.plot.bar(ax=ax, colormap=cmap, alpha=.5, ylabel="Count", rot=45)

            ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',
                      borderaxespad=1, title=f"{feature} values",
                      title_fontsize="small", alignment="left")

            _adjust_box_widths(fig, 0.8, True)
            # make_beautiful_axis(ax, plot_type="barplot")
        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 10))
            if (n_y := np.unique(y).size) > 2:
                cmap_palette = sns.color_palette(cmap, n_y)
            else:
                cmap_palette = palette

            sns.boxplot(
                ax=ax,
                y=df_X[feature],
                x=y,
                showfliers=False,
                palette=cmap_palette,
                hue=y,
                boxprops=dict(alpha=.2),
                whiskerprops=dict(alpha=.2),
                width=.4
            )

            sns.stripplot(
                ax=ax,
                y=df_X[feature],
                x=y,
                palette=cmap_palette,
                hue=y,
                alpha=.5,
                size=4,
                marker="D",
                legend=False,
            )

            # make_beautiful_axis(ax, plot_type="boxplot")

            if show_zero:
                if ax.get_ylim()[0] > 0:
                    ax.set_ylim(bottom=0)

                if ax.get_ylim()[1] < 0:
                    ax.set_ylim(top=0)

            ax.set_ylabel('')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        ax.set_title(feature, fontsize=5)

        if export_file:
            fig.savefig(Path(path, f"{feature}.{fmt}"), dpi=95)

        if not show_fig:
            plt.close()


def scatterplot_features(
        features,
        df_X,
        y,
        categorical_features=10,
        cmap='viridis',
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
    features : list of string or int
        The list of `features` for which we want to represent the scatter plot

    df_X : pd.DataFrame
        The input DataFrame (not preprocessed)

    y : Series
        y used in the fit. Should be a Series to get the name of the outcome.

    categorical_features : list of str or int, default=None
        List of the features that are categorical or upper bound number of unique values to be considered as categorical
        If None, no features are categorical.

    cmap : str, default='viridis'
        The name of the colormap to use.

    show_fig : bool, default=True
        Whether to display the figure

    export_file : bool, default=False
        If set to True, it will export the plot using the path and the format.
        The names of the different file are generated automatically with the name
        of the features.

    path : str or Path, default='./'
        path to the directory. Should not contain the name of the file and the
        extension

    fmt : str, default='pdf'
        Format of the export. Should be string
    """
    for col in features:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

        if _is_categorical(df_X, col, categorical_features):

            cmap_palette = sns.color_palette(cmap, df_X[col].nunique())

            values = df_X[col]
            order = np.sort(np.unique(values))[::-1]
            sns.boxplot(
                ax=ax,
                y=values,
                orient="h",
                x=y,
                order=order,
                showfliers=False,
                palette=cmap_palette,
                boxprops=dict(alpha=.2),
                whiskerprops=dict(alpha=.2),
                width=.4
            )

            sns.stripplot(
                ax=ax,
                y=values,
                orient="h",
                x=y,
                hue=values,
                order=order,
                hue_order=order,
                palette=cmap_palette,
                alpha=.5,
                size=4,
                legend=False,
                marker="D"
            )
            make_beautiful_axis(ax, plot_type="boxplot", gridline_axis="x")

        else:
            sns.scatterplot(ax=ax, y=df_X[col], color="#C41E3A", x=y, s=10, **kwargs)
            make_beautiful_axis(ax, plot_type="scatterplot")
        # fig.subplots_adjust(top=0.9)
        ax.set_ylabel('')
        ax.set_title(col, fontsize=5)

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
        classes=None,
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

    path: str or Path, default='./Boxplot of predictions.pdf'
        Path to the directory. Should also contain the name of the file and the
        extension (format)

    **kwargs: arguments
        Further arguments to pass to the subplots function
    """
    if isinstance(y_true, pd.Series):
        y_axis_title = y_true.name
        y_true = np.array(y_true)
    else:
        y_axis_title = "True Label"

    if isinstance(y_preds, pd.Series):
        x_axis_title = y_preds.name
        y_preds = np.array(y_preds)
    else:
        x_axis_title = "Predictions"

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    palette = ["#4D4F53", "#C41E3A"]

    sns.boxplot(
        ax=ax,
        y=y_true,
        x=y_preds,
        orient='h',
        color='.9',
        showfliers=False,
        palette=palette,
        boxprops=dict(alpha=.25),
        width=.5
    )

    sns.stripplot(ax=ax, y=y_true, x=y_preds, hue=y_true, orient='h', palette=palette, legend=False, **kwargs)
    if classes is not None:
        ax.set_yticklabels(labels=classes)

    roc_auc = roc_auc_score(y_true, y_preds)
    auc_ci_lo, auc_ci_up = compute_CI(y_true, y_preds, scoring="roc_auc")
    precision, recall, _ = precision_recall_curve(y_true, y_preds)
    pr_auc = auc(recall, precision)
    pr_ci_lo, pr_ci_up = compute_CI(y_true, y_preds, scoring="prc_auc")

    plt.title(
        fr"$\bf{{AUROC}}$={roc_auc:.2f} [{auc_ci_lo:.2f}, {auc_ci_up:.2f}]" + '\n'
        fr"$\bf{{AUPRC}}$={pr_auc:.2f} [{pr_ci_lo:.2f}, {pr_ci_up:.2f}]",
        fontsize=10
    )
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)

    make_beautiful_axis(ax, plot_type="boxplot", gridline_axis="x")

    plt.tight_layout()
    if export_file:
        fig.savefig(path, dpi=95, bbox_inches="tight")

    if not show_fig:
        plt.close()

    return fig, ax


def _adjust_box_widths(fig, fac, barplot=False):
    """
    Adjust the widths of a seaborn-generated boxplot and of the barplot.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        The Figure of the plot

    fac : float
        The width scaling factor
    """
    for ax in fig.axes:
        for c in ax.get_children():
            if isinstance(c, mpatches.Rectangle) and barplot:
                # change the width of the rectangle
                c.set_width(fac * c.get_width())

            # searching for PathPatches
            if isinstance(c, mpatches.PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for j in ax.lines:
                    if np.all(j.get_xdata() == [xmin, xmax]):
                        j.set_xdata([xmin_new, xmax_new])


def scatterplot_regression_predictions(
        y_true,
        y_preds,
        show_fig=True,
        export_file=False,
        paths='./Scatterplot of predictions',
        linear_estimation=True,
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

    paths: str or Path or list of Path, default='./'
        Path to the directory. Should also contain the name of the file and the
        extension (format)

    linear_estimation: bool, default=False
        If set to True, it will plot the linear estimation of the predictions

    **kwargs: arguments
    """
    if not isinstance(paths, list):
        paths = [paths]

    fig, ax = plt.subplots(1, 1, **kwargs)
    sns.scatterplot(ax=ax, x=y_true, y=y_preds, color="#d4e0f6", alpha=.9, edgecolor="#ACB4CD", s=50)
    p1 = max(max(y_preds), max(y_true))
    p2 = min(min(y_preds), min(y_true))
    ax.plot([p1, p2], [p1, p2], c='#f23a49', alpha=.7, ls="--")

    r2 = r2_score(y_true=y_true, y_pred=y_preds)
    r2_ci_lo, r2_ci_up = compute_CI(y_true=y_true, y_preds=y_preds, scoring="r2")
    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_preds))
    rmse_ci_up, rmse_ci_lo = compute_CI(y_true=y_true, y_preds=y_preds, scoring="rmse")
    mae = mean_absolute_error(y_true=y_true, y_pred=y_preds)
    mae_ci_up, mae_ci_lo = compute_CI(y_true=y_true, y_preds=y_preds, scoring="mae")
    pearson_stats, pearson_pvalue = pearsonr(y_true, y_preds)

    title = (
        fr"$\bf{{Pearson r}}$: score = {pearson_stats:.2f}, pvalue= {pearson_pvalue:.2e}" + "\n"
        fr"$\bf{{R2-score}}$ = {r2:.2f} [{r2_ci_lo:.2f}, {r2_ci_up:.2f}]" + "\n"
        fr"$\bf{{RMSE}}$ = {rmse:.2f} [{rmse_ci_lo:.2f}, {rmse_ci_up:.2f}]" + "\n"
        fr"$\bf{{MAE}}$ = {mae:.2f} [{mae_ci_lo:.2f}, {mae_ci_up:.2f}]"
    )
    if linear_estimation:
        lin = LinearRegression().fit(X=np.reshape(y_true, (-1, 1)), y=y_preds)
        ax.plot([p1, p2], lin.predict([[p1], [p2]]), c='#1d00c2', alpha=.7)
        title += f"\nLinear regression: slope = {lin.coef_[0]:.2f}, intercept = {lin.intercept_:.2f}"

    ax.set_title(title, fontsize=10)

    make_beautiful_axis(ax, plot_type="scatterplot")

    plt.tight_layout()
    if export_file:
        for path in paths:
            fig.savefig(path, dpi=95)

    if not show_fig:
        plt.close()

    return fig, ax


def make_beautiful_axis(ax, plot_type="roc", gridline_axis="y"):
    """
    Function to make the axis beautiful

    Parameters
    ----------
    gridline_axis
    ax: matplotlib.Axis

    plot_type:str, default="roc"
        - "roc": make axis beautiful for a ROC plot
        - "prc": make axis beautiful for a PRC plot
        - "barplot"
        - "boxplot"

    Returns
    -------
    new_ax: matplotlib.Axis
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    if plot_type == "boxplot":
        box_patches = [patch for patch in ax.patches if type(patch) == mpatches.PathPatch]
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

    if plot_type in ["barplot", "boxplot"]:
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8, axis=gridline_axis)

    if plot_type in ["roc", "prc"]:
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.7, zorder=-10)
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.set_xticks(ticks=[0, .25, .5, .75, 1.], labels=[0, .25, .5, .75, 1])
        ax.set_yticks(ticks=[0, .25, .5, .75, 1.], labels=[0, .25, .5, .75, 1])
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        if plot_type == "roc":
            ax.set_xlabel("1 - Specificity", size=12)
            ax.set_ylabel("Sensitivity", size=12)

        elif plot_type == "prc":
            ax.set_xlabel("Recall", size=12)
            ax.set_ylabel("Precision", size=12)

    ax.tick_params(size=0, which="both", labelsize=12)

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
        ax.plot(x[y >= 0], y[y >= 0], color="#487fad",
                alpha=0.2, lw=1, zorder=0)
        ax.annotate("f1={0:0.1f}".format(f_score),
                    xy=(y[45] + 0.02, 1.02), fontsize=7)


def _is_categorical(df, feature, categorical_features):
    """Determine if a feature is categorical or not.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the feature.
    feature : str
        feature to check.
    categorical_features : list of str or int
        list of categorical features or the maximum number of unique values for a feature to be considered categorical.

    Returns
    -------
    bool
        Whether the feature is categorical or not.
    """
    if isinstance(categorical_features, list) and feature in categorical_features:
        return True
    if isinstance(categorical_features, int) and df[feature].nunique() <= categorical_features:
        return True
    return False
