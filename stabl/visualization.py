import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import pandas as pd


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
        sns.scatterplot(ax=ax, y=df_X[col], color="#C41E3A", x=y, s=10,  **kwargs)
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
