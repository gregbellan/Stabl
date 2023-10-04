from copy import deepcopy
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.base import clone

from .unionfind import UnionFind
from .metrics import jaccard_similarity, fdr_similarity, tpr_similarity, fscore_similarity
from .visualization import make_beautiful_axis
from .stacked_generalization import stacked_multi_omic
from .pipelines_utils import BenchmarkWrapper
from .stabl import Stabl, save_stabl_results
from sklearn.svm import l1_min_c
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

lasso = Lasso(max_iter=int(1e6))
lassocv = LassoCV(n_alphas=30, max_iter=int(1e6), cv=5)

en = ElasticNet(max_iter=int(1e6))
encv = ElasticNetCV(n_alphas=30, cv=5, max_iter=int(1e6))

ridge = Ridge()
ridgecv = RidgeCV(alphas=np.logspace(-2, 2, 30), cv=5, scoring='r2')

feature_metrics = {
    "JACCARD": jaccard_similarity,
    "FDR": fdr_similarity,
    "TPR": tpr_similarity,
}
features_array_metrics = {
    "AUC features": roc_auc_score,
}
binary_prediction_metrics = {
    "AUC": roc_auc_score,
    "AVP": average_precision_score
}
regression_prediction_metrics = {
    "R2": r2_score,
    "RMSE": lambda x, y: mean_squared_error(x, y, squared=False),
    "MSE": mean_squared_error,
    "MAE": mean_absolute_error,
}


def _make_groups(X, percentile):
    n = X.shape[1]
    u = UnionFind(elements=range(n))
    corr_mat = pd.DataFrame(X).corr().values
    corr_val = corr_mat[np.triu_indices_from(corr_mat, k=1)]
    threshold = np.percentile(corr_val, percentile)
    for i in np.arange(n):
        for j in np.arange(n):
            if abs(corr_mat[i, j]) > threshold:
                u.union(i, j)
    res = list(map(list, u.components()))
    res = list(map(np.array, res))
    return res


def fscore_metrics(betas):
    f"""Generate a dictionary of fscore metrics for given betas

    Parameters
    ----------
    betas : list of float
        List of betas for fscore

    Returns
    -------
    dict of [str: function]
        Dictionary of fscore metrics
    """
    return {
        f"F-SCORE {b} features": lambda x, y: fscore_similarity(x, y, beta=b)
        for b in betas
    }


def _linear_output(X, n_informative, snr=2):
    """Generate a linear output from a dataset

    Parameters
    ----------
    X : pd.DataFrame
        Dataset
    n_informative : int
        Number of informative features
    scale : int, optional
        Scale for the random normal function, by default 1

    Returns
    -------
    np.ndarray
        Linear output
    """
    rng = np.random.RandomState(42)
    betas = np.zeros(X.shape[1])
    betas[:n_informative] = rng.uniform(low=-10, high=10, size=n_informative)
    y = X @ betas
    print(f"STD y:", y.std())
    y = (y-y.mean())/y.std()
    # y = y-y.mean()
    scale = np.std(y) / snr
    print(f"SCALE: {scale}")
    y += rng.normal(scale=scale, size=X.shape[0])

    return y


def make_train_test(
    U,
    n_informative,
    snr=2,
    scale_u=2,
    input_type="normal",
    output_type="linear",
    multiomics=False,
    omics_features=None,
    omics_info=None,
):
    """Generate a synthetic dataset with given parameters

    Parameters
    ----------
    n_features : int
        Number of features in the dataset
    n_informative : int
        Number of informative features in the dataset
    s_u : int, optional
        _description_, by default 2
    sigma : int, optional
        Standard deviation of the normal distribution, by default 1
    corr_pow : str, optional
        Correlation power in ["no", "low", "medium", "hard"]. It is used only if input_type is "normal". By default "low"
    output_type : str, optional
        Type of output in ["linear", "sigmoid", "tanh", "square", "complex", "binary"], by default "linear". See _non_linear_output for more details.
    input_type : str, optional
        Type of input in ["normal", "zinb", "nb"], by default "normal"

    Returns
    -------
    tuple of (X_train : pd.DataFrame, X_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series)
        Generated dataset
    """

    rng = np.random.RandomState(42)
    indices = ['Id_' + str(i) for i in range(U.shape[0])]
    features = ['Ft_' + str(i) for i in range(U.shape[1])]
    U = (U - U.mean())/U.std()

    n_features = U.shape[1]

    if input_type == "normal":
        X = rng.normal(size=U.shape)
        for i in range(n_informative):
            X[:, i] += scale_u * U[:, i]

    else:
        X = U

    if output_type == "binary":
        y = _linear_output(U, n_informative, snr=snr)
        y = 1 / (1 + np.exp(-y))
        y = (y > 0.5).astype(int)

    elif output_type == "linear":
        y = _linear_output(U, n_informative, snr=snr)

    else:
        raise ValueError("unrecognized output_type")

    X = pd.DataFrame(data=X, index=indices, columns=features)
    y = pd.Series(data=y, index=indices, name='Outcome')

    if output_type == "binary":
        print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10000, random_state=42)

    if multiomics:
        train_data_dict = dict()
        test_data_dict = dict()

        info_rd = rng.choice(n_informative, n_informative, replace=False)
        nb_noinfo_features = n_features - n_informative
        features_rd = rng.choice(np.arange(n_informative, n_features, 1), nb_noinfo_features, replace=False)

        i = 0
        f = 0
        for idx, (features, info) in enumerate(zip(omics_features, omics_info)):
            info_indices = info_rd[i: i+info]
            features_indices = features_rd[f: f+features]

            tmp_df = pd.concat([X.iloc[:, info_indices], X.iloc[:, features_indices]], axis=1)
            train_data_dict[f"omic {idx+1}"] = tmp_df.loc[X_train.index]
            test_data_dict[f"omic {idx+1}"] = tmp_df.loc[X_test.index]

            i += info
            f += features

        return train_data_dict, test_data_dict, y_train, y_test

    return X_train, X_test, y_train, y_test


def save_metric_graph(df_results, estimators, path, name):
    """Save metric graph using Perf_Median, Perf_Q1 and Perf_Q3 stored in df_results. It generates the graph with and without error bars.

    Parameters
    ----------
    df_results : pd.DataFrame
        dataframe containing the results of the metric for each estimator
    estimators : list of str
        Name of the estimators to plot and compare
    path : pathlib.Path
        Path to the folder where the graphs will be saved
    name : str
        Name of the metric
    """
    os.makedirs(path, exist_ok=True)

    # generate median graph
    fig, ax = plt.subplots(figsize=(10, 10))

    for estimator in estimators:
        ax.plot(df_results.index, df_results[f"{estimator}_{name}_Perf_Median"], linestyle=":", marker="o",
                label=estimator)

    ax.set_xlabel("Number of samples")
    ax.set_ylabel(name+" Median")
    lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                     loc="lower center", fontsize=8)
    fig.savefig(Path(path, name + ".pdf"), dpi=95,
                bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    # generate median graph with error bars
    fig, ax = plt.subplots(figsize=(10, 10))

    for estimator in estimators:
        perf_low = list(
            df_results[f"{estimator}_{name}_Perf_Median"] - df_results[f"{estimator}_{name}_Perf_Q1"])
        perf_up = list(-df_results[f"{estimator}_{name}_Perf_Median"] +
                       df_results[f"{estimator}_{name}_Perf_Q3"])
        ax.errorbar(df_results.index,
                    df_results[f"{estimator}_{name}_Perf_Median"],
                    [perf_low, perf_up],
                    marker='o',
                    capsize=3,
                    fmt=':',
                    capthick=1,
                    label=estimator
                    )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(name+" Median")
    lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                     loc="lower center", fontsize=8)
    fig.savefig(Path(path, name + " performance with errorbar.pdf"),
                dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def save_continuous_scores(dict_results, estimators, path, name, base_estim):
    """Save metric graph using Perf_Mean and Perf_Std stored in df_results. It generates the graph with and without error bars, and with and without log scale.

    Parameters
    ----------
    df_results : pd.DataFrame
        dataframe containing the results of the metric for each estimator
    estimators : list of str
        Name of the estimators to plot and compare
    path : pathlib.Path
        Path to the folder where the graphs will be saved
    name : str
        Name of the metric
    """
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))

    for estimator in base_estim:
        if "Stabl" not in estimator:
            fig, ax = plt.subplots(figsize=(10, 10))
            df_results = pd.DataFrame()
            for n_sample in dict_results.keys():
                df_tmp = dict_results[n_sample][[f"stabl_{estimator}_{name}", f"{estimator}_{name}"]]
                df_tmp = pd.concat([df_tmp, pd.Series(n_sample, index=df_tmp.index, name="n_sample")], axis=1)
                df_results = pd.concat([df_results, df_tmp], axis=0)
            df_results = pd.melt(df_results, id_vars=["n_sample"])
            sns.boxplot(df_results, x="n_sample", y="value", hue="variable", ax=ax)
            # ax.errorbar(df_results.index,
            #             df_results[f"{estimator}_{name}_Mean"],
            #             df_results[f"{estimator}_{name}_Std"],
            #             marker='o',
            #             capsize=3,
            #             fmt=':',
            #             capthick=1,
            #             label=estimator
            #             )
            ax.set_xlabel("Number of samples")
            ax.set_ylabel(f"{name} Score")
            # ax.set_yscale("log")
            lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                             loc="lower center", fontsize=8)
            fig.savefig(Path(path, f"{name} - {estimator} - scores with errorbar.pdf"),
                        dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')
            ax.set_yscale("log")
            fig.savefig(Path(path, f"{name} - {estimator} - ylog scores with errorbar.pdf"),
                        dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.close()

    fig, ax = plt.subplots(figsize=(10, 10))
    medians = pd.DataFrame(index=dict_results.keys())
    for n_sample in dict_results.keys():
        for estimator in estimators:
            medians.loc[n_sample, f"{estimator}_{name}"] = dict_results[n_sample][f"{estimator}_{name}"].median()
    for estimator in estimators:
        ax.plot(medians.index, medians[f"{estimator}_{name}"],
                linestyle=":", marker="o", label=estimator)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel(f"Median {name} score")
    ax.set_yscale("log")
    lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                     loc="lower center", fontsize=8)
    fig.savefig(Path(path, name + " scores.pdf"),
                dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')
    ax.set_xscale("log")
    fig.savefig(Path(path, name + " log scores.pdf"),
                dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def save_roc_values(ground_truth, all_values, path):
    """Genetate ROC curve for each estimator and save it in path.

    Parameters
    ----------
    ground_truth : boolean array with shape (n_features,)
        Boolean array with True for real features and False for fake features
    all_values : dict of str: np.array
        Dict containing the name of the estimator as key and the array of values as value
    path : pathlib.Path
        Name of the file where the graph will be saved
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5,
            color='#4D4F53', alpha=.8, label="Chance")
    for i in all_values.keys():
        mean_values = np.median(all_values[i], axis=0)
        auc = roc_auc_score(ground_truth, mean_values)
        roc = roc_curve(ground_truth, mean_values)
        ax.plot(roc[0], roc[1], lw=2, alpha=1,
                label=f"{i} ROC (AUC={auc:.3f})")
    make_beautiful_axis(ax)
    lgd = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
                     loc="lower center", fontsize=8)
    fig.savefig(path, dpi=95, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()


def _get_stabl(model):
    """Return the Stabl model if it is or contains one, else return None. 
    It extracts the Stabl object from a Pipeline if it is one.
    Its purpose is to extract the Stabl object from any estimator.

    Parameters
    ----------
    model : sklearn estimator
        Estimator to check

    Returns
    -------
    Stabl or None
        Stabl model or None
    """
    if isinstance(model, BenchmarkWrapper):
        return _get_stabl(model.model)
    if isinstance(model, Pipeline):
        for step in model:
            if isinstance(step[1], Stabl):
                return step[1]
    if isinstance(model, Stabl):
        return model
    return None


def perform_prediction(estimator, X_train, y_train, X_test, output_type, features=None):
    if isinstance(estimator, Stabl):
        if np.sum(features) == 0:
            mean_score = 0.5 if output_type == "binary" else np.mean(y_train)
            preds = np.zeros(X_test.shape[0]) + mean_score
        else:
            if output_type == "binary":
                model = LogisticRegression(max_iter=int(1e6))
                model.fit(X_train.iloc[:, features], y_train)
                preds = model.predict_proba(X_test.iloc[:, features])[:, 1]
            else:
                model = LinearRegression()
                model.fit(X_train.iloc[:, features], y_train)
                preds = model.predict(X_test.iloc[:, features])
    else:
        if output_type == "binary":
            preds = estimator.predict_proba(X_test)[:, 1]
        else:
            preds = estimator.predict(X_test)

    return preds


def synthetic_benchmark_feature_selection(
        X,
        estimators,
        preprocess_transformer=StandardScaler(),
        n_informative_list=[25],
        n_samples_list=[30, 40, 50, 75, 100, 150, 250, 350, 500, 750, 1000],
        n_experiments=50,
        result_folder_title="Synthetic Results",
        f_number=[0.1, 0.5, 1],
        output_type="linear",
        input_type="normal",
        verbose=0,
        snr=2,
        scale_u=2,
        sgl_corr_percentile=99,
        sgl_groups=None,
        base_estim=["alasso", "lasso", "en", "sgl"]
):
    """Run benchmark on synthetic data in regression task

    Parameters
    ----------
    estimators : dict[str, sklearn estimator]
        Dict of feature selectors to benchmark with their name as key. They must implement `fit`, `get_support(indices=True)` and `get_importances` methods.
        You can use the surge_library.pipelines_utils.BenchmarkWrapper object to wrap your estimator.
    preprocess_transformer : sklearn transformer, optional
        Preprocessing transformer to apply on the data, by default StandardScaler()
    n_features_list : list, optional
        number of features in the synthetic data, by default [200, 500, 1000]
    n_informative_list : list, optional
        number of informative features, by default [10, 25, 50]
    n_samples_list : list, optional
        number of sample in the experiment, by default [30, 40, 50, 75, 100, 150, 250, 350, 500, 750, 1000]
    n_experiments : int, optional
        number of experiment to create statistics, by default 40
    result_folder_title : str, optional
        Output folder, by default "Synthetic Results"
    f_number : list, optional
        f number used in the f-score, by default [0.1, 0.5, 1]
    corr_pow : str, optional
        Correlation power in ["no", "low", "medium", "hard"], by default "low"
    output_type : str, optional
        Type of output in ["linear", "sigmoid", "tanh", "square", "complex", "binary"], by default "linear". See _non_linear_output for more details.
    verbose : int, optional
        Verbosity level, by default 0.
        It will use the binary representation of the number to select the verbosity level and will use the last 5 bits :
        - 10000 : print the tqdm for the estimators
        - 01000 : print the tqdm for the experiments
        - 00100 : print the tqdm for the sample's list
        - 00010 : print the tqdm for the informative features list
        - 00001 : print the tqdm for the number of features list
        We can combine multiple levels by adding the corresponding bits. For example, `31 = Ob11111` will print all the tqdm.
    """

    if isinstance(f_number, int):
        f_number = [f_number]
    verbose_array = bin(verbose + 64)[-5:]  # 64 = 100000

    for n_info in (pbar_info := tqdm(n_informative_list, total=len(n_informative_list), disable=verbose_array[3] == '0')):
        pbar_info.set_description_str(f"[n_informative={n_info}]")

        X_train, X_test, y_train, y_test = make_train_test(
            U=X,
            n_informative=n_info,
            output_type=output_type,
            snr=snr,
            scale_u=scale_u,
            input_type=input_type
        )

        task_type = "binary" if output_type == "binary" else "regression"

        groud_truth_features_array = np.zeros(X.shape[1])
        groud_truth_features_array[:n_info] = 1
        groud_truth_features = list(np.arange(0, n_info, 1))

        fdr_min_thresholds = {k: {} for k in estimators.keys()}
        for k in estimators.keys():
            fdr_min_thresholds[k] = {n: [] for n in n_samples_list}

        df_results = pd.DataFrame(data=None, index=n_samples_list)
        dict_results = {}
        df_results.index.name = 'nb_samples'

        for n_samples in (pbar_sample := tqdm(n_samples_list, total=len(n_samples_list), disable=verbose_array[2] == '0')):
            pbar_sample.set_description_str(f"[n_informative={n_info}, n_samples={n_samples}]")

            all_values = {k: [] for k in estimators.keys()}
            all_preds = {k: [] for k in estimators.keys()}

            f_metrics = {**feature_metrics, **fscore_metrics(f_number)}
            f_array_metrics = features_array_metrics

            if output_type == "binary":
                p_metrics = binary_prediction_metrics
            else:
                p_metrics = regression_prediction_metrics

            scores = {
                k: [] for k in (
                    list(f_metrics.keys()) +
                    list(p_metrics.keys()) +
                    list(f_array_metrics.keys()) +
                    ["NB_FEATURES"]
                )
            }

            estimators_scores = {k: deepcopy(scores) for k in estimators.keys()}
            rng = np.random.RandomState(42)

            best_param_en = pd.DataFrame(index=range(n_experiments), columns=["alpha", "l1_ratio"])

            for iteration in (pbar_exp := tqdm(range(n_experiments), total=n_experiments, disable=verbose_array[1] == '0')):
                pbar_exp.set_description_str(f"[n_informative={n_info}, n_samples={n_samples}, iteration={iteration + 1}]")
                X_subtrain, y_subtrain = resample(
                    X_train,
                    y_train,
                    n_samples=n_samples,
                    replace=False,
                    random_state=rng
                )

                X_subtrain_p = preprocess_transformer.fit_transform(X_subtrain)

                X_subtrain = pd.DataFrame(
                    data=X_subtrain_p,
                    index=X_subtrain.index,
                    columns=preprocess_transformer.get_feature_names_out()
                )

                X_test_std = preprocess_transformer.transform(X_test)

                X_test_std = pd.DataFrame(
                    data=X_test_std,
                    index=X_test.index,
                    columns=preprocess_transformer.get_feature_names_out()
                )

                for k, v in (pbar_estim := tqdm(estimators.items(), total=len(estimators), disable=verbose_array[0] == '0')):
                    pbar_estim.set_description_str(f"[n_informative={n_info}, n_samples={n_samples}, iteration={iteration + 1}, estimator={k}]")
                    v = clone(v)
                    if isinstance(v, Stabl):
                        stabl_model = v
                        prev_lambda_grid = stabl_model.lambda_grid
                        if prev_lambda_grid is None:
                            if task_type == "binary":
                                min_C = l1_min_c(X_subtrain, y_subtrain)
                                lambda_grid = np.linspace(min_C, min_C * 100, 5)
                                prev_lambda_grid = {"C": lambda_grid}
                            else:
                                l_max = np.linalg.norm(X_subtrain.values.T@y_subtrain, np.inf)/X_subtrain.shape[0]
                                lambda_grid = np.geomspace(l_max/30, l_max + 5, 5)
                                prev_lambda_grid = {"alpha": lambda_grid}
                            stabl_model.set_params(lambda_grid=prev_lambda_grid)

                        if n_samples <= 75:
                            fdr_threshold_range = np.arange(0.1, 1, 0.01)
                        else:
                            fdr_threshold_range = np.arange(0.1, 1, 0.01)

                        stabl_model.set_params(fdr_threshold_range=fdr_threshold_range)

                        v = stabl_model
                    if k == "SGL":
                        if sgl_groups is None:
                            groups_sgl = _make_groups(X_subtrain, sgl_corr_percentile)
                        else:
                            groups_sgl = sgl_groups

                        setattr(model.estimator, "groups", groups_sgl)

                    fitted_est = v.fit(X_subtrain, y_subtrain)

                    if isinstance(v, Stabl):
                        stabl_path = Path(f"./{result_folder_title}/NbInfo={n_info}/Stabl results n_samples={n_samples}/iteration {iteration}/{k}")
                        save_stabl_results(
                            stabl=stabl_model,
                            path=stabl_path,
                            df_X=X_subtrain,
                            y=y_subtrain,
                            task_type=task_type
                        )

                        fdr_min_thresholds[k][n_samples].append(v.fdr_min_threshold_)
                        features = list(fitted_est.get_support(indices=True))
                        # print(features)
                        features_importances = fitted_est.stabl_scores_.max(1)

                    else:
                        # if isinstance(fitted_est, GridSearchCV) and k == "en":
                        if isinstance(fitted_est, GridSearchCV):
                            # model_path = Path(f"./{result_folder_title}/NbInfo={n_info}/Model results n_samples={n_samples}/iteration {iteration}/{k}")
                            # os.makedirs(model_path, exist_ok=True)
                            for el_k, el_v in fitted_est.best_params_.items():
                                best_param_en.loc[iteration, el_k] = el_v
                        features = list(np.where(fitted_est.best_estimator_.coef_.flatten())[0])
                        features_importances = fitted_est.best_estimator_.coef_.flatten()

                    all_values[k].append(features_importances)
                    for n, f in f_metrics.items():
                        estimators_scores[k][n].append(f(features, groud_truth_features))

                    for n, f in f_array_metrics.items():
                        estimators_scores[k][n].append(f(groud_truth_features_array, features_importances))

                    estimators_scores[k]["NB_FEATURES"].append(len(features))

                    preds = perform_prediction(fitted_est, X_subtrain, y_subtrain, X_test_std, output_type, features=features)
                    all_preds[k].append(preds)
                    for n, f in p_metrics.items():
                        estimators_scores[k][n].append(f(y_test, preds))
                os.makedirs(Path(f"./{result_folder_title}/NbInfo={n_info}/Model results n_samples={n_samples}"), exist_ok=True)
                best_param_en.to_csv(Path(f"./{result_folder_title}/NbInfo={n_info}/Model results n_samples={n_samples}/en_params.csv"))
            # -----------------------------------------------------------------------------------
            # SAVING RAW SCORES
            scores_path = f"./{result_folder_title}/NbInfo={n_info}/Scores"
            os.makedirs(scores_path, exist_ok=True)
            scores_df = pd.DataFrame(data={
                f"{m}_{k}": v for m in estimators.keys() for k, v in estimators_scores[m].items()
            },
                index=[
                f"Experiment {i + 1}" for i in range(n_experiments)]
            )
            scores_df.to_csv(Path(scores_path, f"Raw_scores_NbSamples={n_samples}.csv"))

            # -----------------------------------------------------------------------------------
            # SAVING RAW WEIGHTS
            values_path = f"./{result_folder_title}/NbInfo={n_info}/Values/Raw_values_NbSamples={n_samples}"
            os.makedirs(values_path, exist_ok=True)
            for i in estimators.keys():
                values_df = pd.DataFrame(data=all_values[i],
                                         index=[
                                             f"Experiment {j + 1}" for j in range(n_experiments)]
                                         )
                values_df.to_csv(
                    Path(values_path, f"Raw_values_{i}_NbSamples={n_samples}.csv"))

            # SAVING ROC CURVES
            roc_path = f"./{result_folder_title}/NbInfo={n_info}/ROC curves"
            os.makedirs(roc_path, exist_ok=True)
            save_roc_values(groud_truth_features_array, all_values, Path(
                roc_path, f"ROC_NbSamples={n_samples}.pdf"))

            # -----------------------------------------------------------------------------------
            # SAVING RAW PREDICTIONS
            folder = f"./{result_folder_title}/NbInfo={n_info}/Predictions/" \
                     f"Raw_predictions_NbSamples={n_samples}"
            os.makedirs(folder, exist_ok=True)
            for name, preds in all_preds.items():
                df_predictions = pd.DataFrame(data=np.array(preds).T,
                                              columns=[
                                                  f"Experiment {i}" for i in range(n_experiments)]
                                              )
                df_predictions["Outcome"] = np.array(y_test)
                df_predictions.to_csv(
                    Path(folder, f"{name}_predictions_NbSamples={n_samples}.csv"))

            # -----------------------------------------------------------------------------------
            df_results_sample = pd.Series(name=n_samples)

            dict_results[n_samples] = pd.DataFrame()

            # Storing scores
            for i in estimators.keys():
                for j in list(p_metrics.keys()) + ["NB_FEATURES"]:
                    df_results_sample.loc[f"{i}_{j}_Mean"] = np.mean(
                        scores_df[f"{i}_{j}"])
                    df_results_sample.loc[f"{i}_{j}_Std"] = np.std(
                        scores_df[f"{i}_{j}"])
                    dict_results[n_samples][f"{i}_{j}"] = scores_df[f"{i}_{j}"]

            # Storing other metrics
            for i in estimators.keys():
                for j in (list(f_metrics.keys()) + list(f_array_metrics.keys()) + list(p_metrics.keys()) + ["NB_FEATURES"]):
                    iqr_ground = np.percentile(
                        scores_df[f"{i}_{j}"], [75, 25])
                    df_results_sample.loc[f'{i}_{j}_Perf_Median'] = np.median(
                        scores_df[f"{i}_{j}"])
                    df_results_sample.loc[f'{i}_{j}_Perf_Q1'] = iqr_ground[1]
                    df_results_sample.loc[f'{i}_{j}_Perf_Q3'] = iqr_ground[0]

            df_results.loc[df_results_sample.name, df_results_sample.index] = df_results_sample
            print(df_results.shape)

        # -----------------------------------------------------------------------------------
        # SAVING FDR MIN THRESHOLDS
        folder = f"./{result_folder_title}/NbInfo={n_info}/Predictions/" \
            f"Raw_predictions_NbSamples={n_samples}"
        print(fdr_min_thresholds)
        for model_name, v in estimators.items():
            if isinstance(v, Stabl):
                df_threshold = pd.DataFrame(data=fdr_min_thresholds[model_name])
                df_threshold.index = [f"Exp{j}" for j in range(n_experiments)]
                df_threshold.columns = [f"NbSamples={j}" for j in n_samples_list]
                df_threshold.to_csv(f"./{result_folder_title}/NbInfo={n_info}/df_thresholds_{model_name}.csv")

        metrics_path = f'./{result_folder_title}/NbInfo={n_info}/Metrics'
        os.makedirs(metrics_path, exist_ok=True)

        df_results.to_csv(Path(metrics_path, "df_results.csv"))
        os.makedirs(Path(metrics_path, "dict_results"), exist_ok=True)
        for i, j in dict_results.items():
            j.to_csv(Path(metrics_path, "dict_results", f"dict_results_sample_{i}.csv"))

        for i in (list(f_metrics.keys()) + list(f_array_metrics.keys())):
            path = Path(metrics_path, i)
            os.makedirs(path, exist_ok=True)
            save_metric_graph(df_results, estimators.keys(), path, name=i)

        for i in list(p_metrics.keys()) + ["NB_FEATURES"]:
            path = Path(metrics_path, i)
            os.makedirs(path, exist_ok=True)
            save_continuous_scores(
                dict_results, estimators.keys(), path, i, base_estim=base_estim)


def synthetic_benchmark_feature_selection_multiomics(
        X,
        estimators,
        preprocess_transformer=StandardScaler(),
        n_informative_list=[25],
        n_samples_list=[30, 40, 50, 75, 100, 150, 250, 350, 500, 750, 1000],
        n_experiments=30,
        result_folder_title="Synthetic Results",
        f_number=[0.1, 0.5, 1],
        output_type="linear",
        input_type="normal",
        verbose=0,
        snr=2,
        scale_u=2,
        sgl_corr_percentile=99,
        sgl_groups=None,
        base_estim=["alasso", "lasso", "en", "sgl"],
        multiomics=False,
        omics_features=None,
        omics_info=None,
        n_iter=1000,

):
    """Run benchmark on synthetic data in regression task

    Parameters
    ----------
    estimators : dict[str, sklearn estimator]
        Dict of feature selectors to benchmark with their name as key. They must implement `fit`, `get_support(indices=True)` and `get_importances` methods.
        You can use the surge_library.pipelines_utils.BenchmarkWrapper object to wrap your estimator.
    preprocess_transformer : sklearn transformer, optional
        Preprocessing transformer to apply on the data, by default StandardScaler()
    n_features_list : list, optional
        number of features in the synthetic data, by default [200, 500, 1000]
    n_informative_list : list, optional
        number of informative features, by default [10, 25, 50]
    n_samples_list : list, optional
        number of sample in the experiment, by default [30, 40, 50, 75, 100, 150, 250, 350, 500, 750, 1000]
    n_experiments : int, optional
        number of experiment to create statistics, by default 40
    result_folder_title : str, optional
        Output folder, by default "Synthetic Results"
    f_number : list, optional
        f number used in the f-score, by default [0.1, 0.5, 1]
    corr_pow : str, optional
        Correlation power in ["no", "low", "medium", "hard"], by default "low"
    output_type : str, optional
        Type of output in ["linear", "sigmoid", "tanh", "square", "complex", "binary"], by default "linear". See _non_linear_output for more details.
    verbose : int, optional
        Verbosity level, by default 0.
        It will use the binary representation of the number to select the verbosity level and will use the last 5 bits :
        - 10000 : print the tqdm for the estimators
        - 01000 : print the tqdm for the experiments
        - 00100 : print the tqdm for the sample's list
        - 00010 : print the tqdm for the informative features list
        - 00001 : print the tqdm for the number of features list
        We can combine multiple levels by adding the corresponding bits. For example, `31 = Ob11111` will print all the tqdm.
    """

    if isinstance(f_number, int):
        f_number = [f_number]
    verbose_array = bin(verbose + 64)[-5:]  # 64 = 100000

    for n_info in (pbar_info := tqdm(n_informative_list, total=len(n_informative_list), disable=verbose_array[3] == '0')):
        pbar_info.set_description_str(f"[n_informative={n_info}]")

        X_train, X_test, y_train, y_test = make_train_test(
            U=X,
            n_informative=n_info,
            output_type=output_type,
            snr=snr,
            scale_u=scale_u,
            input_type=input_type,
            multiomics=multiomics,
            omics_features=omics_features,
            omics_info=omics_info
        )

        task_type = "binary" if output_type == "binary" else "regression"

        X_train_tot = pd.concat(X_train.values(), axis=1)
        groud_truth_features = ['Ft_' + str(i) for i in range(25)]
        groud_truth_features_array = np.zeros(X.shape[1])
        for f in groud_truth_features:
            groud_truth_features_array[X_train_tot.columns.get_loc(f)] = 1

        df_results = pd.DataFrame(data=None, index=n_samples_list)
        dict_results = {}
        df_results.index.name = 'nb_samples'

        for n_samples in (pbar_sample := tqdm(n_samples_list, total=len(n_samples_list), disable=verbose_array[2] == '0')):
            pbar_sample.set_description_str(f"[n_informative={n_info}, n_samples={n_samples}]")

            all_values = {k: [] for k in estimators.keys()}
            all_preds = {k: [] for k in estimators.keys()}

            f_metrics = {**feature_metrics, **fscore_metrics(f_number)}
            f_array_metrics = features_array_metrics

            if output_type == "binary":
                p_metrics = binary_prediction_metrics
            else:
                p_metrics = regression_prediction_metrics

            scores = {
                k: [] for k in (
                    list(f_metrics.keys()) +
                    list(p_metrics.keys()) +
                    ["NB_FEATURES"]
                )
            }

            estimators_scores = {k: deepcopy(scores) for k in estimators.keys()}
            rng = np.random.RandomState(42)

            for iteration in (pbar_exp := tqdm(range(n_experiments), total=n_experiments, disable=verbose_array[1] == '0')):
                pbar_exp.set_description_str(f"[n_informative={n_info}, n_samples={n_samples}, iteration={iteration + 1}]")

                y_subtrain = resample(
                    y_train,
                    n_samples=n_samples,
                    replace=False,
                    random_state=rng
                )

                X_train_tot = pd.concat(X_train.values(), axis=1).loc[y_subtrain.index]
                X_test_tot = pd.concat(X_test.values(), axis=1)

                iter_features = {}
                train_predictions = {}
                test_predictions = {}
                for k, v in estimators.items():
                    iter_features[k] = []
                    if not isinstance(v, Stabl):
                        train_predictions[k] = pd.DataFrame(columns=X_train.keys())
                        test_predictions[k] = pd.DataFrame(columns=X_train.keys(), index=X_test_tot.index)

                for omic_name, omic_df in X_train.items():
                    X_subtrain = omic_df.loc[y_subtrain.index]
                    X_subtest = X_test[omic_name]
                    X_subtrain_p = preprocess_transformer.fit_transform(X_subtrain)

                    X_subtrain = pd.DataFrame(
                        data=X_subtrain_p,
                        index=X_subtrain.index,
                        columns=preprocess_transformer.get_feature_names_out()
                    )
                    X_subtest = pd.DataFrame(
                        data=preprocess_transformer.transform(X_subtest),
                        index=X_subtest.index,
                        columns=preprocess_transformer.get_feature_names_out()
                    )

                    for k, v in (pbar_estim := tqdm(estimators.items(), total=len(estimators), disable=verbose_array[0] == '0')):
                        pbar_estim.set_description_str(f"[n_informative={n_info}, n_samples={n_samples}, iteration={iteration + 1}, estimator={k}]")
                        v = clone(v)
                        if isinstance(v, Stabl):
                            stabl_model = v
                            prev_lambda_grid = stabl_model.lambda_grid
                            if prev_lambda_grid is None:
                                if task_type == "binary":
                                    min_C = l1_min_c(X_subtrain, y_subtrain)
                                    lambda_grid = np.linspace(min_C, min_C * 100, 5)
                                    prev_lambda_grid = {"C": lambda_grid}
                                else:
                                    l_max = np.linalg.norm(X_subtrain.values.T@y_subtrain, np.inf)/X_subtrain.shape[0]
                                    lambda_grid = np.geomspace(l_max/30, l_max + 5, 5)
                                    prev_lambda_grid = {"alpha": lambda_grid}
                                stabl_model.set_params(lambda_grid=prev_lambda_grid)

                            if n_samples <= 75:
                                fdr_threshold_range = np.arange(0.1, 1, 0.01)
                            else:
                                fdr_threshold_range = np.arange(0.1, 1, 0.01)

                            stabl_model.set_params(fdr_threshold_range=fdr_threshold_range)

                            v = stabl_model
                        if k == "SGL":
                            if sgl_groups is None:
                                groups_sgl = _make_groups(X_subtrain, sgl_corr_percentile)
                            else:
                                groups_sgl = sgl_groups

                            setattr(model.estimator, "groups", groups_sgl)

                        fitted_est = v.fit(X_subtrain, y_subtrain)

                        if isinstance(v, Stabl):
                            stabl_path = Path(f"./{result_folder_title}/NbInfo={n_info}/Stabl results {omic_name} n_samples={n_samples}/iteration {iteration}/{k}")
                            save_stabl_results(
                                stabl=stabl_model,
                                path=stabl_path,
                                df_X=X_subtrain,
                                y=y_subtrain,
                                task_type=task_type
                            )

                            features = list(fitted_est.get_feature_names_out())
                            iter_features[k] += features
                            # print(f"{omic_name}", iter_features)

                        else:
                            features = list(np.where(fitted_est.best_estimator_.coef_.flatten())[0])
                            print(features)
                            iter_features[k] += list(X_subtrain.iloc[:, features].columns)
                            train_predictions[k][omic_name] = fitted_est.predict_proba(X_subtrain)[:, 1] if task_type == "binary" else fitted_est.predict(X_subtrain)
                            test_predictions[k][omic_name] = fitted_est.predict_proba(X_subtest)[:, 1] if task_type == "binary" else fitted_est.predict(X_subtest)

                for k, v in estimators.items():
                    X_train_std = preprocess_transformer.fit_transform(X_train_tot[iter_features[k]])

                    X_train_std = pd.DataFrame(
                        data=X_train_std,
                        index=X_train_tot.index,
                        columns=preprocess_transformer.get_feature_names_out()
                    )

                    X_test_std = preprocess_transformer.transform(X_test_tot[iter_features[k]])

                    X_test_std = pd.DataFrame(
                        data=X_test_std,
                        index=X_test_tot.index,
                        columns=preprocess_transformer.get_feature_names_out()
                    )

                    # all_values[k].append(features_importances)
                    for n, f in f_metrics.items():
                        estimators_scores[k][n].append(f(iter_features[k], groud_truth_features))

                    # for n, f in f_array_metrics.items():
                    #     estimators_scores[k][n].append(f(groud_truth_features_array, features_importances))

                    estimators_scores[k]["NB_FEATURES"].append(len(iter_features[k]))

                    if isinstance(v, Stabl):
                        if len(iter_features[k]) == 0:
                            mean_score = 0.5 if output_type == "binary" else np.mean(y_subtrain)
                            preds = np.zeros(X_test.shape[0]) + mean_score
                        else:
                            if output_type == "binary":
                                model = LogisticRegression(penalty=None, max_iter=int(1e6), n_jobs=-1)
                                model.fit(X_train_std, y_subtrain)
                                preds = model.predict_proba(X_test_std)[:, 1]
                            else:
                                model = LinearRegression(n_jobs=-1)
                                model.fit(X_train_std, y_subtrain)
                                preds = model.predict(X_test_std)
                    else:
                        print(f"Late Fusion {k}")
                        model_lf_path = Path(f"./{result_folder_title}/NbInfo={n_info}/Late Fusion results n_samples={n_samples}/iteration {iteration}/{k}")
                        os.makedirs(model_lf_path, exist_ok=True)
                        predictions = train_predictions[k]
                        stacked_df, weights = stacked_multi_omic(predictions, y_subtrain, task_type, n_iter=n_iter)
                        weights.to_csv(Path(model_lf_path, f"Associated weights LF {k}.csv"))
                        stacked_df.to_csv(
                            Path(model_lf_path, f"Stacked Generalization predictions train LF {k}.csv"))
                        valid_preds = test_predictions[k]
                        preds = (pd.DataFrame(valid_preds) @ pd.DataFrame(weights)).sum(axis=1) / weights.sum().values
                        ###

                        # run en late fusion

                        ###
                        # if output_type == "binary":
                        #    preds = fitted_est.predict_proba(X_test_std)[:, 1]
                        # else:
                        #    preds = fitted_est.predict(X_test_std)
                    print(k, preds.shape)
                    all_preds[k].append(preds)
                    for n, f in p_metrics.items():
                        estimators_scores[k][n].append(f(y_test, preds))

            # -----------------------------------------------------------------------------------
            # SAVING RAW SCORES
            scores_path = f"./{result_folder_title}/NbInfo={n_info}/Scores"
            os.makedirs(scores_path, exist_ok=True)
            scores_df = pd.DataFrame(data={
                f"{m}_{k}": v for m in estimators.keys() for k, v in estimators_scores[m].items()
            },
                index=[
                f"Experiment {i + 1}" for i in range(n_experiments)]
            )
            scores_df.to_csv(Path(scores_path, f"Raw_scores_NbSamples={n_samples}.csv"))

            # -----------------------------------------------------------------------------------
            # SAVING RAW WEIGHTS
#             values_path = f"./{result_folder_title}/NbInfo={n_info}/Values/Raw_values_NbSamples={n_samples}"
#             os.makedirs(values_path, exist_ok=True)
#             for i in estimators.keys():
#                 values_df = pd.DataFrame(data=all_values[i],
#                                          index=[
#                                              f"Experiment {j + 1}" for j in range(n_experiments)]
#                                          )
#                 values_df.to_csv(
#                     Path(values_path, f"Raw_values_{i}_NbSamples={n_samples}.csv"))

#             # SAVING ROC CURVES
#             roc_path = f"./{result_folder_title}/NbInfo={n_info}/ROC curves"
#             os.makedirs(roc_path, exist_ok=True)
#             save_roc_values(groud_truth_features_array, all_values, Path(
#                 roc_path, f"ROC_NbSamples={n_samples}.pdf"))

            # -----------------------------------------------------------------------------------
            # SAVING RAW PREDICTIONS
            folder = f"./{result_folder_title}/NbInfo={n_info}/Predictions/" \
                     f"Raw_predictions_NbSamples={n_samples}"
            os.makedirs(folder, exist_ok=True)
            for name, preds in all_preds.items():
                df_predictions = pd.DataFrame(data=np.array(preds).T,
                                              columns=[
                                                  f"Experiment {i}" for i in range(n_experiments)]
                                              )
                df_predictions["Outcome"] = np.array(y_test)
                df_predictions.to_csv(
                    Path(folder, f"{name}_predictions_NbSamples={n_samples}.csv"))

            # -----------------------------------------------------------------------------------
            df_results_sample = pd.Series(name=n_samples)

            dict_results[n_samples] = pd.DataFrame()

            # Storing scores
            for i in estimators.keys():
                for j in list(p_metrics.keys()) + ["NB_FEATURES"]:
                    df_results_sample.loc[f"{i}_{j}_Mean"] = np.mean(
                        scores_df[f"{i}_{j}"])
                    df_results_sample.loc[f"{i}_{j}_Std"] = np.std(
                        scores_df[f"{i}_{j}"])
                    dict_results[n_samples][f"{i}_{j}"] = scores_df[f"{i}_{j}"]

            # Storing other metrics
            for i in estimators.keys():
                for j in (list(f_metrics.keys()) + list(p_metrics.keys()) + ["NB_FEATURES"]):
                    iqr_ground = np.percentile(
                        scores_df[f"{i}_{j}"], [75, 25])
                    df_results_sample.loc[f'{i}_{j}_Perf_Median'] = np.median(
                        scores_df[f"{i}_{j}"])
                    df_results_sample.loc[f'{i}_{j}_Perf_Q1'] = iqr_ground[1]
                    df_results_sample.loc[f'{i}_{j}_Perf_Q3'] = iqr_ground[0]

            df_results.loc[df_results_sample.name, df_results_sample.index] = df_results_sample
            print(df_results.shape)

        # -----------------------------------------------------------------------------------
        # SAVING FDR MIN THRESHOLDS
#         folder = f"./{result_folder_title}/NbInfo={n_info}/Predictions/" \
#                      f"Raw_predictions_NbSamples={n_samples}"
#         print(fdr_min_thresholds)
#         for model_name, v in estimators.items():
#             if isinstance(v, Stabl):
#                 df_threshold = pd.DataFrame(data = fdr_min_thresholds[model_name])
#                 df_threshold.index = [f"Exp{j}" for j in range(n_experiments)]
#                 df_threshold.columns = [f"NbSamples={j}" for j in n_samples_list]
#                 df_threshold.to_csv(f"./{result_folder_title}/NbInfo={n_info}/df_thresholds_{model_name}.csv")

        metrics_path = f'./{result_folder_title}/NbInfo={n_info}/Metrics'
        os.makedirs(metrics_path, exist_ok=True)

        df_results.to_csv(Path(metrics_path, "df_results.csv"))
        os.makedirs(Path(metrics_path, "dict_results"), exist_ok=True)
        for i, j in dict_results.items():
            j.to_csv(Path(metrics_path, "dict_results", f"dict_results_sample_{i}.csv"))

        for i in (list(f_metrics.keys())):
            path = Path(metrics_path, i)
            os.makedirs(path, exist_ok=True)
            save_metric_graph(df_results, estimators.keys(), path, name=i)

        for i in list(p_metrics.keys()) + ["NB_FEATURES"]:
            path = Path(metrics_path, i)
            os.makedirs(path, exist_ok=True)
            save_continuous_scores(
                dict_results, estimators.keys(), path, i, base_estim=base_estim)
