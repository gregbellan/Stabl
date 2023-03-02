import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.transforms as transforms

from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.base import clone

from .stabl import Stabl
from .metrics import jaccard_matrix, jaccard_similarity

cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

lasso = Lasso(max_iter=int(1e6))
lassocv = LassoCV(n_alphas=30, max_iter=int(1e6), cv=5)

en = ElasticNet(max_iter=int(1e6))
encv = ElasticNetCV(n_alphas=30, cv=5, max_iter=int(1e6))

ridge = Ridge()
ridgecv = RidgeCV(alphas=np.logspace(-2, 2, 30), cv=5, scoring='r2')


def compute_est_FDR(stability_selection):
    FDPs = []  # Initializing false discovery proportions
    thresholds_grid = np.arange(0., 1., 0.01)
    artificial_proportion = stability_selection.artificial_proportion
    max_scores_artificial = np.max(stability_selection.stabl_scores_artificial_, axis=1)
    max_scores = np.max(stability_selection.stabl_scores_, axis=1)

    for thresh in thresholds_grid:
        num = np.sum((1 / artificial_proportion) * (max_scores_artificial > thresh))
        denum = np.max([1, np.sum((max_scores > thresh))])
        FDP = (num + 1) / denum
        FDPs.append(FDP)

    return FDPs


def compute_true_FDR(stability_selection, true_features_indices):
    max_scores = stability_selection.stabl_scores_.max(axis=1)
    thresh_grid = np.arange(0, 1, 0.01)
    FDRs = []
    tFDRs = []
    for thresh in thresh_grid:
        set_selected_features = set(np.where(max_scores > thresh)[0])
        if len(set_selected_features) == 0:
            FDR = 1.
            tFDR = 1.
        else:
            FP = len(set_selected_features - set(true_features_indices))
            FDR = (FP + 1) / len(set_selected_features)
            tFDR = FP / len(set_selected_features)
        tFDRs.append(tFDR)
        FDRs.append(FDR)
    return FDRs, tFDRs


def make_train_test(n_features, n_informative, noise=2, s_u=2, sigma=1):
    n_samples = 50000
    rng = np.random.RandomState(42)

    X = rng.normal(size=(n_samples, n_features))
    U = rng.normal(scale=s_u, size=(n_samples, n_features))

    for i in range(n_informative):
        X[:, i] += noise * U[:, i]

    betas = np.zeros(n_features)
    betas[:n_informative] = rng.uniform(low=-10, high=10, size=n_informative)
    y = U @ betas + np.random.normal(scale=sigma, size=n_samples)

    indices = ['Id.' + str(i) for i in range(X.shape[0])]
    features = ['Ft.' + str(i) for i in range(X.shape[1])]

    X = pd.DataFrame(data=X, index=indices, columns=features)
    y = pd.Series(data=y, index=indices, name='Outcome')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=10000, random_state=42)

    return X_train, X_test, y_train, y_test


def save_fdr_figures(
        true_FDRs_decoy,
        estimated_FDRs_decoy,
        path
):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(0, 1, 0.01)

    plt.plot(x, np.median(true_FDRs_decoy, axis=0), label="FDR", linewidth=2, color="#4D4F53")
    plt.fill_between(x,
                     np.percentile(true_FDRs_decoy, [25], axis=0)[0],
                     np.percentile(true_FDRs_decoy, [75], axis=0)[0],
                     alpha=.09,
                     color="#4D4F53"
                     )

    plt.plot(x, np.median(estimated_FDRs_decoy, axis=0), label="FDP +", linewidth=2, color="#C41E3A")
    plt.fill_between(
        x,
        np.percentile(estimated_FDRs_decoy, [25], axis=0)[0],
        np.percentile(estimated_FDRs_decoy, [75], axis=0)[0],
        alpha=.09,
        color="#C41E3A"
    )

    min_FDP = x[np.argmin(np.median(estimated_FDRs_decoy, axis=0))]
    plt.axvline(
        min_FDP,
        label="min FDP +",
        lw=2,
        ls="--",
        color="#C41E3A"
    )
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    plt.text(min_FDP, 1.07, "{:.2f}".format(min_FDP), color="#C41E3A", transform=trans)
    plt.legend()
    plt.ylim(bottom=-0.05, top=1.05)
    plt.ylabel("FDR")
    plt.xlabel("Threshold of selection")
    plt.savefig(path)
    plt.close()


def save_jaccard_decoy_lasso(df_results, path):
    # Jaccard stability for decoy
    decoy_stab_low = list(df_results["Decoy_Jaccard_Stab_Median"] - df_results["Decoy_Jaccard_Stab_Q1"])
    decoy_stab_up = list(-df_results["Decoy_Jaccard_Stab_Median"] + df_results["Decoy_Jaccard_Stab_Q3"])

    # Jaccard performance for decoy
    decoy_perf_low = list(df_results["Decoy_Jaccard_Perf_Median"] - df_results["Decoy_Jaccard_Perf_Q1"])
    decoy_perf_up = list(-df_results["Decoy_Jaccard_Perf_Median"] + df_results["Decoy_Jaccard_Perf_Q3"])

    # Jaccard stability for lasso
    lasso_stab_low = list(df_results["Lasso_Jaccard_Stab_Median"] - df_results["Lasso_Jaccard_Stab_Q1"])
    lasso_stab_up = list(-df_results["Lasso_Jaccard_Stab_Median"] + df_results["Lasso_Jaccard_Stab_Q3"])

    # Jaccard performance for lasso
    lasso_perf_low = list(df_results["Lasso_Jaccard_Perf_Median"] - df_results["Lasso_Jaccard_Perf_Q1"])
    lasso_perf_up = list(-df_results["Lasso_Jaccard_Perf_Median"] + df_results["Lasso_Jaccard_Perf_Q3"])

    plt.figure(figsize=(8, 8))

    plt.errorbar(df_results.index,
                 df_results["Decoy_Jaccard_Stab_Median"],
                 [decoy_stab_low, decoy_stab_up],
                 marker='o',
                 capsize=3,
                 fmt=':',
                 capthick=1,
                 label="Decoy"
                 )

    plt.errorbar(df_results.index,
                 df_results["Lasso_Jaccard_Stab_Median"],
                 [lasso_stab_low, lasso_stab_up],
                 marker='o',
                 capsize=3,
                 capthick=1,
                 fmt=':',
                 label="Lasso"
                 )

    plt.xlabel("Number of samples")
    plt.ylabel("Jaccard Median")
    plt.legend()
    plt.savefig(path + "Jaccard stability decoy vs lasso.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))

    plt.errorbar(df_results.index,
                 df_results["Decoy_Jaccard_Perf_Median"],
                 [decoy_perf_low, decoy_perf_up],
                 marker='o',
                 capsize=3,
                 fmt=':',
                 capthick=1,
                 label="Decoy"
                 )

    plt.errorbar(df_results.index,
                 df_results["Lasso_Jaccard_Perf_Median"],
                 [lasso_perf_low, lasso_perf_up],
                 marker='o',
                 capsize=3,
                 capthick=1,
                 fmt=':',
                 label="Lasso"
                 )

    plt.xlabel("Number of samples")
    plt.ylabel("Jaccard Median")
    plt.legend()
    plt.savefig(path + "Jaccard performance decoy vs lasso.pdf")
    plt.close()


def save_jaccard_decoy_stab(df_results, path):
    plt.figure(figsize=(8, 8))
    plt.plot(df_results.index, df_results.Decoy_Jaccard_Stab_Median, linestyle=":", marker="o", color="C0",
             label="Decoy")
    plt.plot(df_results.index, df_results.SS_03_Jaccard_Stab_Median, linestyle=":", marker="o", color="C1",
             label="Stab 0.3")
    plt.plot(df_results.index, df_results.SS_05_Jaccard_Stab_Median, linestyle=":", marker="o", color="C2",
             label="Stab 0.5")
    plt.plot(df_results.index, df_results.SS_08_Jaccard_Stab_Median, linestyle=":", marker="o", color="C3",
             label="Stab 0.8")
    plt.xlabel("Number of samples")
    plt.ylabel("Jaccard Median")
    plt.legend()
    plt.savefig(path + "Jaccard stability decoy vs stab.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(df_results.index, df_results.Decoy_Jaccard_Perf_Median, linestyle=":", marker="o", color="C0",
             label="Decoy")
    plt.plot(df_results.index, df_results.SS_03_Jaccard_Perf_Median, linestyle=":", marker="o", color="C1",
             label="Stab 0.3")
    plt.plot(df_results.index, df_results.SS_05_Jaccard_Perf_Median, linestyle=":", marker="o", color="C2",
             label="Stab 0.5")
    plt.plot(df_results.index, df_results.SS_08_Jaccard_Perf_Median, linestyle=":", marker="o", color="C3",
             label="Stab 0.8")
    plt.xlabel("Number of samples")
    plt.ylabel("Jaccard Median")
    plt.legend()
    plt.savefig(path + "Jaccard performace decoy vs stab.pdf")
    plt.close()


def save_R2_scores(df_results, path):
    plt.figure(figsize=(8, 8))

    plt.errorbar(df_results.index,
                 df_results["Decoy_R2_Mean"],
                 df_results["Decoy_R2_Std"],
                 marker='o',
                 capsize=3,
                 fmt=':',
                 capthick=1,
                 label="Decoy"
                 )

    plt.errorbar(df_results.index,
                 df_results["Lasso_R2_Mean"],
                 df_results["Lasso_R2_Std"],
                 marker='o',
                 capsize=3,
                 fmt=':',
                 capthick=1,
                 label="Lasso"
                 )

    plt.xlabel("Number of samples")
    plt.ylabel("Mean R2 Score")
    plt.legend()
    plt.savefig(path + "LassovsDecoy R2 scores.pdf")
    plt.xscale("log")
    plt.savefig(path + "LassovsDecoy R2 scores log.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(df_results.index, df_results["Decoy_R2_Mean"], marker='o', linestyle=":", label="Decoy")
    plt.plot(df_results.index, df_results["SS_03_R2_Mean"], marker='o', linestyle=":", label="Stab 0.3")
    plt.plot(df_results.index, df_results["SS_05_R2_Mean"], marker='o', linestyle=":", label="Stab 0.5")
    plt.plot(df_results.index, df_results["SS_08_R2_Mean"], marker='o', linestyle=":", label="Stab 0.8")
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Mean R2 score")
    plt.savefig(path + "StabvsDecoy R2 scores.pdf")
    plt.xscale("log")
    plt.savefig(path + "StabvsDecoy R2 scores log.pdf")
    plt.close()


def save_MSE_scores(df_results, path):
    plt.figure(figsize=(8, 8))

    plt.errorbar(
        df_results.index,
        df_results["Decoy_MSE_Mean"],
        df_results["Decoy_MSE_Std"],
        marker='o',
        capsize=3,
        fmt=':',
        capthick=1,
        label="Decoy"
    )

    plt.errorbar(
        df_results.index,
        df_results["Lasso_MSE_Mean"],
        df_results["Lasso_MSE_Std"],
        marker='o',
        capsize=3,
        fmt=':',
        capthick=1,
        label="Lasso"
    )

    plt.xlabel("Number of samples")
    plt.ylabel("Mean MSE Score")
    plt.legend()
    plt.savefig(path + "LassovsDecoy MSE scores.pdf")
    plt.xscale("log")
    plt.savefig(path + "LassovsDecoy MSE scores log.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(df_results.index, df_results["Decoy_MSE_Mean"], marker='o', linestyle=":", label="Decoy")
    plt.plot(df_results.index, df_results["SS_03_MSE_Mean"], marker='o', linestyle=":", label="Stab 0.3")
    plt.plot(df_results.index, df_results["SS_05_MSE_Mean"], marker='o', linestyle=":", label="Stab 0.5")
    plt.plot(df_results.index, df_results["SS_08_MSE_Mean"], marker='o', linestyle=":", label="Stab 0.8")
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Mean MSE score")
    plt.savefig(path + "StabvsDecoy MSE scores.pdf")
    plt.xscale("log")
    plt.savefig(path + "StabvsDecoy MSE scores log.pdf")
    plt.close()


def save_MAE_scores(df_results, path):
    plt.figure(figsize=(8, 8))

    plt.errorbar(df_results.index,
                 df_results["Decoy_MAE_Mean"],
                 df_results["Decoy_MAE_Std"],
                 marker='o',
                 capsize=3,
                 fmt=':',
                 capthick=1,
                 label="Decoy"
                 )

    plt.errorbar(df_results.index,
                 df_results["Lasso_MAE_Mean"],
                 df_results["Lasso_MAE_Std"],
                 marker='o',
                 capsize=3,
                 fmt=':',
                 capthick=1,
                 label="Lasso"
                 )

    plt.xlabel("Number of samples")
    plt.ylabel("Mean MAE Score")
    plt.legend()
    plt.savefig(path + "LassovsDecoy MAE scores.pdf")
    plt.xscale("log")
    plt.savefig(path + "LassovsDecoy MAE scores log.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(df_results.index, df_results["Decoy_MAE_Mean"], marker='o', linestyle=":", label="Decoy")
    plt.plot(df_results.index, df_results["SS_03_MAE_Mean"], marker='o', linestyle=":", label="Stab 0.3")
    plt.plot(df_results.index, df_results["SS_05_MAE_Mean"], marker='o', linestyle=":", label="Stab 0.5")
    plt.plot(df_results.index, df_results["SS_08_MAE_Mean"], marker='o', linestyle=":", label="Stab 0.8")
    plt.legend()
    plt.xlabel("Number of samples")
    plt.ylabel("Mean MAE score")
    plt.savefig(path + "StabvsDecoy MAE scores.pdf")
    plt.xscale("log")
    plt.savefig(path + "StabvsDecoy MAE scores log.pdf")
    plt.close()


def save_nb_features_plot(df_results, path, nb_info):
    plt.figure(figsize=(8, 8))
    plt.plot(df_results.index, df_results.Avg_nb_features_Lasso, linestyle=":", marker="o", label="Lasso")
    plt.plot(df_results.index, df_results.Avg_nb_features_Decoy, linestyle=":", marker="o", label="Decoy")
    plt.xlabel("Number of samples")
    plt.ylabel("Average number of selected features")
    plt.legend()
    plt.savefig(path + "Average number of selected featues Lasso vs MOB.pdf")
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.figure(figsize=(8, 8))
    plt.plot(df_results.index, df_results.Avg_nb_features_Decoy, linestyle=":", marker="o", label="Decoy")
    plt.plot(df_results.index, df_results.Avg_nb_features_SS_03, linestyle=":", marker="o", label="Stab 0.3")
    plt.plot(df_results.index, df_results.Avg_nb_features_SS_05, linestyle=":", marker="o", label="Stab 0.5")
    plt.plot(df_results.index, df_results.Avg_nb_features_SS_08, linestyle=":", marker="o", label="Stab 0.8")
    plt.xlabel("Number of samples")
    plt.ylim(top=2 * nb_info, bottom=-1)
    plt.ylabel("Average number of selected features")
    plt.legend()
    plt.savefig(path + "Average number of selected featues MOB vs Stability.pdf")
    plt.close()


def create_results_folder(result_folder_title, nb_features, nb_info):
    os.makedirs(f"{result_folder_title}/NbFeatures={nb_features}_NbInfo={nb_info}/FDR graphs", exist_ok=True)
    os.makedirs(f"{result_folder_title}/NbFeatures={nb_features}_NbInfo={nb_info}/Avg features graphs", exist_ok=True)
    os.makedirs(f"{result_folder_title}/NbFeatures={nb_features}_NbInfo={nb_info}/Predictions", exist_ok=True)
    os.makedirs(f"{result_folder_title}/NbFeatures={nb_features}_NbInfo={nb_info}/Scores/MSE", exist_ok=True)
    os.makedirs(f"{result_folder_title}/NbFeatures={nb_features}_NbInfo={nb_info}/Scores/MAE", exist_ok=True)
    os.makedirs(f"{result_folder_title}/NbFeatures={nb_features}_NbInfo={nb_info}/Scores/R2", exist_ok=True)
    os.makedirs(f"{result_folder_title}/NbFeatures={nb_features}_NbInfo={nb_info}/Jaccard", exist_ok=True)


def synthetic_benchmark_regression(
        base_estimator,
        lambda_name,
        n_features_list,
        n_informative_list,
        n_samples_list,
        n_experiments,
        artificial_type,
        result_folder_title
):
    for n_features in n_features_list:
        for n_info in n_informative_list:

            create_results_folder(result_folder_title, n_features, n_info)

            print('=======================================================================')
            print(f'Computation for {n_features} features with {n_info} informative ones')
            print('=======================================================================')

            X_train, X_test, y_train, y_test = make_train_test(
                n_features=n_features,
                n_informative=n_info
            )

            print('Starting the training for different subtraining sizes')
            print('-----------------------------------------------------------------------')
            df_results = pd.DataFrame(data=None, index=n_samples_list)
            df_results.index.name = 'nb_samples'

            df_thresholds = pd.DataFrame(index=[f"Exp.{i}" for i in range(n_experiments)])

            for n_samples in n_samples_list:

                print('n_samples = %d' % n_samples)
                print('***********************************************************************')

                all_stable_features_decoy = []
                all_lasso_features = []
                all_stable_features_ss_03 = []
                all_stable_features_ss_05 = []
                all_stable_features_ss_08 = []

                all_threshold_stabl = []

                all_lasso_preds = []
                all_stabl_preds = []
                all_ss_03_preds = []
                all_ss_05_preds = []
                all_ss_08_preds = []

                lasso_r2_scores = []
                lasso_mse_scores = []
                lasso_mae_scores = []

                decoy_r2_scores = []
                decoy_mse_scores = []
                decoy_mae_scores = []

                ss_03_r2_scores = []
                ss_03_mse_scores = []
                ss_03_mae_scores = []

                ss_05_r2_scores = []
                ss_05_mse_scores = []
                ss_05_mae_scores = []

                ss_08_r2_scores = []
                ss_08_mse_scores = []
                ss_08_mae_scores = []

                groud_truth_features = list(np.arange(0, n_info, 1))

                jaccard_ground_truth_decoy = []  # vector of jaccard metric at each repetition of the experiment
                jaccard_ground_truth_lasso = []
                jaccard_ground_truth_ss_03 = []
                jaccard_ground_truth_ss_05 = []
                jaccard_ground_truth_ss_08 = []

                true_FDRs_ss = []
                true_FDRs_decoy = []
                true_FDRs_estimates_decoy = []
                estimated_FDRs_decoy = []

                for iteration in range(n_experiments):
                    if iteration % 10 == 0:
                        print('Repetition of this experiment %d/%d' % (iteration + 1, n_experiments))

                    X_subtrain, y_subtrain = resample(X_train,
                                                      y_train,
                                                      n_samples=n_samples,
                                                      replace=False
                                                      )

                    scaler = StandardScaler()
                    X_subtrain = scaler.fit_transform(X_subtrain)
                    X_test_std = scaler.transform(X_test)

                    l_max = np.linalg.norm(X_subtrain.T @ y_subtrain, np.inf) / X_subtrain.shape[0]
                    LAMBDA_GRID_ = np.geomspace(l_max / 100, l_max + 5, 5)

                    # Stability Selection without noise
                    stab_sel = Stabl(base_estimator=base_estimator,
                                     lambda_name=lambda_name,
                                     n_bootstraps=100,
                                     lambda_grid=LAMBDA_GRID_,
                                     artificial_type=None,
                                     sample_fraction=.5,
                                     hard_threshold=.1,
                                     replace=False,
                                     verbose=0,
                                     n_jobs=-1,
                                     random_state=42
                                     ).fit(X_subtrain, y_subtrain)

                    true_FDRs_ss.append(compute_true_FDR(stab_sel, groud_truth_features))

                    ss_features_03 = list(stab_sel.get_support(indices=True, new_hard_threshold=.3))
                    jaccard_ground_truth_ss_03.append(jaccard_similarity(ss_features_03, groud_truth_features))
                    all_stable_features_ss_03.append(ss_features_03)

                    if len(ss_features_03) == 0:
                        ss_03_preds = [np.median(y_subtrain)] * len(y_test)

                    else:
                        ss_03_ridge = clone(ridgecv).fit(X_subtrain[:, ss_features_03], y_subtrain)
                        ss_03_preds = ss_03_ridge.predict(X_test_std[:, ss_features_03])
                    all_ss_03_preds.append(ss_03_preds)
                    ss_03_r2_scores.append(r2_score(y_test, ss_03_preds))
                    ss_03_mse_scores.append(mean_squared_error(y_test, ss_03_preds))
                    ss_03_mae_scores.append(mean_absolute_error(y_test, ss_03_preds))

                    ss_features_05 = list(stab_sel.get_support(indices=True, new_hard_threshold=.5))
                    jaccard_ground_truth_ss_05.append(jaccard_similarity(ss_features_05, groud_truth_features))
                    all_stable_features_ss_05.append(ss_features_05)

                    if len(ss_features_05) == 0:
                        ss_05_preds = [np.median(y_subtrain)] * len(y_test)

                    else:
                        ss_05_ridge = clone(ridgecv).fit(X_subtrain[:, ss_features_05], y_subtrain)
                        ss_05_preds = ss_05_ridge.predict(X_test_std[:, ss_features_05])
                    all_ss_05_preds.append(ss_05_preds)
                    ss_05_r2_scores.append(r2_score(y_test, ss_05_preds))
                    ss_05_mse_scores.append(mean_squared_error(y_test, ss_05_preds))
                    ss_05_mae_scores.append(mean_absolute_error(y_test, ss_05_preds))

                    ss_features_08 = list(stab_sel.get_support(indices=True, new_hard_threshold=.8))
                    jaccard_ground_truth_ss_08.append(jaccard_similarity(ss_features_08, groud_truth_features))
                    all_stable_features_ss_08.append(ss_features_08)

                    if len(ss_features_08) == 0:
                        ss_08_preds = [np.median(y_subtrain)] * len(y_test)

                    else:
                        ss_08_ridge = clone(ridgecv).fit(X_subtrain[:, ss_features_08], y_subtrain)
                        ss_08_preds = ss_08_ridge.predict(X_test_std[:, ss_features_08])
                    all_ss_08_preds.append(ss_08_preds)
                    ss_08_r2_scores.append(r2_score(y_test, ss_08_preds))
                    ss_08_mse_scores.append(mean_squared_error(y_test, ss_08_preds))
                    ss_08_mae_scores.append(mean_absolute_error(y_test, ss_08_preds))

                    # Stability Selection with decoy
                    stab_with_decoy = Stabl(base_estimator=base_estimator,
                                            lambda_name=lambda_name,
                                            n_bootstraps=100,
                                            lambda_grid=LAMBDA_GRID_,
                                            artificial_type=artificial_type,
                                            sample_fraction=.5,
                                            fdr_threshold_range=np.arange(0.1, 1., 0.01),
                                            replace=False,
                                            verbose=0,
                                            n_jobs=-1,
                                            random_state=42
                                            ).fit(X_subtrain, y_subtrain)

                    _ = stab_with_decoy.get_support()
                    all_threshold_stabl.append(stab_with_decoy.fdr_min_threshold_)
                    true_FDRs_estimates, true_FDRs = compute_true_FDR(stab_with_decoy, groud_truth_features)
                    true_FDRs_estimates_decoy.append(true_FDRs_estimates)
                    true_FDRs_decoy.append(true_FDRs)
                    estimated_FDRs_decoy.append(compute_est_FDR(stab_with_decoy))

                    decoy_features = list(stab_with_decoy.get_support(indices=True))
                    jaccard_ground_truth_decoy.append(jaccard_similarity(decoy_features, groud_truth_features))
                    all_stable_features_decoy.append(decoy_features)

                    if len(decoy_features) == 0:
                        decoy_preds = [np.median(y_subtrain)] * len(y_test)

                    else:
                        decoy_ridge = clone(ridgecv).fit(X_subtrain[:, decoy_features], y_subtrain)
                        decoy_preds = decoy_ridge.predict(X_test_std[:, decoy_features])
                    all_stabl_preds.append(decoy_preds)
                    decoy_r2_scores.append(r2_score(y_test, decoy_preds))
                    decoy_mse_scores.append(mean_squared_error(y_test, decoy_preds))
                    decoy_mae_scores.append(mean_absolute_error(y_test, decoy_preds))

                    # Training a Lasso
                    model = clone(lassocv).fit(X_subtrain, y_subtrain)
                    lasso_features = list(np.where(model.coef_)[0])
                    jaccard_ground_truth_lasso.append(jaccard_similarity(lasso_features, groud_truth_features))
                    all_lasso_features.append(lasso_features)

                    lasso_preds = model.predict(X_test_std)
                    all_lasso_preds.append(lasso_preds)
                    lasso_r2_scores.append(r2_score(y_test, lasso_preds))
                    lasso_mse_scores.append(mean_squared_error(y_test, lasso_preds))
                    lasso_mae_scores.append(mean_absolute_error(y_test, lasso_preds))

                df_thresholds[f"NbSamples={n_samples}"] = all_threshold_stabl
                fdr_path = f'./{result_folder_title}/NbFeatures={n_features}_NbInfo={n_info}/FDR graphs/' \
                           f'NbSamples={n_samples}.pdf'
                save_fdr_figures(true_FDRs_decoy, estimated_FDRs_decoy, fdr_path)

                scores_path = f"./{result_folder_title}/NbFeatures={n_features}_NbInfo={n_info}/Scores"
                scores_df = pd.DataFrame(data={"STABL_R2": decoy_r2_scores, "STABL_MSE": decoy_mse_scores,
                                               "STABL_MAE": decoy_mae_scores, "SS_03_R2": ss_03_r2_scores,
                                               "SS_03_MAE": ss_03_mae_scores, "SS_03_MSE": ss_03_mse_scores,
                                               "SS_05_R2": ss_05_r2_scores, "SS_05_MAE": ss_05_mae_scores,
                                               "SS_05_MSE": ss_05_mse_scores, "SS_08_R2": ss_08_r2_scores,
                                               "SS_08_MAE": ss_08_mae_scores, "SS_08_MSE": ss_08_mse_scores,
                                               "Lasso_R2": lasso_r2_scores, "Lasso_MAE": lasso_mae_scores,
                                               "Lasso_MSE": lasso_mse_scores},
                                         index=[f"Experiment {i + 1}" for i in range(n_experiments)]
                                         )

                scores_df.to_csv(Path(scores_path, f"Raw_scores_NbSamples={n_samples}.csv"))

                # -----------------------------------------------------------------------------------
                # SAVING RAW PREDICTIONS
                folder = f"./{result_folder_title}/NbFeatures={n_features}_NbInfo={n_info}/Predictions/" \
                         f"Raw_predictions_NbSamples={n_samples}"

                dict_preds = {"Lasso": all_lasso_preds, "STABL": all_stabl_preds,
                              "SS_03": all_ss_03_preds, "SS_05": all_ss_05_preds,
                              "SS_08": all_ss_08_preds
                              }

                for name, preds in dict_preds.items():
                    df_predictions = pd.DataFrame(data=np.array(preds).T,
                                                  columns=[f"Experiment {i}" for i in range(n_experiments)]
                                                  )
                    df_predictions["Outcome"] = np.array(y_test)
                    os.makedirs(Path(folder, f"{name}"), exist_ok=True)
                    df_predictions.to_csv(Path(folder, f"{name}_predictions_NbSamples={n_samples}.csv"))

                # -----------------------------------------------------------------------------------

                # Storing decoy scores
                df_results.loc[n_samples, 'Decoy_R2_Mean'] = np.mean(decoy_r2_scores)
                df_results.loc[n_samples, 'Decoy_R2_Std'] = np.std(decoy_r2_scores)
                df_results.loc[n_samples, 'Decoy_MSE_Mean'] = np.mean(decoy_mse_scores)
                df_results.loc[n_samples, 'Decoy_MSE_Std'] = np.std(decoy_mse_scores)
                df_results.loc[n_samples, 'Decoy_MAE_Mean'] = np.mean(decoy_mae_scores)
                df_results.loc[n_samples, 'Decoy_MAE_Std'] = np.std(decoy_mae_scores)

                # Storing ss scores
                df_results.loc[n_samples, 'SS_03_R2_Mean'] = np.mean(ss_03_r2_scores)
                df_results.loc[n_samples, 'SS_03_R2_Std'] = np.std(ss_03_r2_scores)
                df_results.loc[n_samples, 'SS_03_MSE_Mean'] = np.mean(ss_03_mse_scores)
                df_results.loc[n_samples, 'SS_03_MSE_Std'] = np.std(ss_03_mse_scores)
                df_results.loc[n_samples, 'SS_03_MAE_Mean'] = np.mean(ss_03_mae_scores)
                df_results.loc[n_samples, 'SS_03_MAE_Std'] = np.std(ss_03_mae_scores)

                df_results.loc[n_samples, 'SS_05_R2_Mean'] = np.mean(ss_05_r2_scores)
                df_results.loc[n_samples, 'SS_05_R2_Std'] = np.std(ss_05_r2_scores)
                df_results.loc[n_samples, 'SS_05_MSE_Mean'] = np.mean(ss_05_mse_scores)
                df_results.loc[n_samples, 'SS_05_MSE_Std'] = np.std(ss_05_mse_scores)
                df_results.loc[n_samples, 'SS_05_MAE_Mean'] = np.mean(ss_05_mae_scores)
                df_results.loc[n_samples, 'SS_05_MAE_Std'] = np.std(ss_05_mae_scores)

                df_results.loc[n_samples, 'SS_08_R2_Mean'] = np.mean(ss_08_r2_scores)
                df_results.loc[n_samples, 'SS_08_R2_Std'] = np.std(ss_08_r2_scores)
                df_results.loc[n_samples, 'SS_08_MSE_Mean'] = np.mean(ss_08_mse_scores)
                df_results.loc[n_samples, 'SS_08_MSE_Std'] = np.std(ss_08_mse_scores)
                df_results.loc[n_samples, 'SS_08_MAE_Mean'] = np.mean(ss_08_mae_scores)
                df_results.loc[n_samples, 'SS_08_MAE_Std'] = np.std(ss_08_mae_scores)

                # Storing Lasso scores
                df_results.loc[n_samples, 'Lasso_R2_Mean'] = np.mean(lasso_r2_scores)
                df_results.loc[n_samples, 'Lasso_R2_Std'] = np.std(lasso_r2_scores)
                df_results.loc[n_samples, 'Lasso_MSE_Mean'] = np.mean(lasso_mse_scores)
                df_results.loc[n_samples, 'Lasso_MSE_Std'] = np.std(lasso_mse_scores)
                df_results.loc[n_samples, 'Lasso_MAE_Mean'] = np.mean(lasso_mae_scores)
                df_results.loc[n_samples, 'Lasso_MAE_Std'] = np.std(lasso_mae_scores)

                # -----------------------------------------------------------------------------------
                # SAVING RAW NB FEATURES

                df_features = pd.DataFrame(data={"# features STABL": [len(i) for i in all_stable_features_decoy],
                                                 "# features SS_03": [len(i) for i in all_stable_features_ss_03],
                                                 "# features SS_05": [len(i) for i in all_stable_features_ss_05],
                                                 "# features SS_08": [len(i) for i in all_stable_features_ss_08],
                                                 "# features Lasso": [len(i) for i in all_lasso_features],
                                                 },
                                           index=[f"Experiment {i}" for i in range(n_experiments)]
                                           )
                df_features.to_csv(
                    f"./{result_folder_title}/NbFeatures={n_features}_NbInfo={n_info}/Avg features graphs/"
                    f"Raw_features_NbSamples={n_samples}.csv")

                # -----------------------------------------------------------------------------------

                # -----------------------------------------------------------------------------------
                # SAVING RAW PERFORMANCE JACCARDS
                folder = f"./{result_folder_title}/NbFeatures={n_features}_NbInfo={n_info}/Jaccard/" \
                         f"Raw_Jaccards_Performance"
                os.makedirs(folder, exist_ok=True)

                df_jaccard_groud_truth = pd.DataFrame(data={"STABL": jaccard_ground_truth_decoy,
                                                            "SS_03": jaccard_ground_truth_ss_03,
                                                            "SS_05": jaccard_ground_truth_ss_05,
                                                            "SS_08": jaccard_ground_truth_ss_08,
                                                            "Lasso": jaccard_ground_truth_lasso
                                                            },
                                                      index=[f"Experiment {i}" for i in range(n_experiments)]
                                                      )
                df_jaccard_groud_truth.to_csv(Path(folder, f"Jaccard_ground_truth_NbSamples={n_samples}.csv"))
                # -----------------------------------------------------------------------------------

                # SAVING RAW STABILITY JACCARDS
                folder = f"./{result_folder_title}/NbFeatures={n_features}_NbInfo={n_info}/" \
                         f"Jaccard/Raw_Jaccards_Stability/"
                os.makedirs(folder, exist_ok=True)
                df_jaccard_stability = pd.DataFrame()

                jaccard_value_decoy = jaccard_matrix(all_stable_features_decoy, remove_diag=False)
                jaccard_value_decoy = np.array(
                    jaccard_value_decoy[np.triu_indices_from(jaccard_value_decoy, k=1)]).flatten()

                jaccard_value_ss_03 = jaccard_matrix(all_stable_features_ss_03, remove_diag=False)
                jaccard_value_ss_03 = np.array(
                    jaccard_value_ss_03[np.triu_indices_from(jaccard_value_ss_03, k=1)]).flatten()

                jaccard_value_ss_05 = jaccard_matrix(all_stable_features_ss_05, remove_diag=False)
                jaccard_value_ss_05 = np.array(
                    jaccard_value_ss_05[np.triu_indices_from(jaccard_value_ss_05, k=1)]).flatten()

                jaccard_value_ss_08 = jaccard_matrix(all_stable_features_ss_08, remove_diag=False)
                jaccard_value_ss_08 = np.array(
                    jaccard_value_ss_08[np.triu_indices_from(jaccard_value_ss_08, k=1)]).flatten()

                jaccard_value_lasso = jaccard_matrix(all_lasso_features, remove_diag=False)
                jaccard_value_lasso = np.array(
                    jaccard_value_lasso[np.triu_indices_from(jaccard_value_lasso, k=1)]).flatten()

                df_jaccard_stability["STABL"] = jaccard_value_decoy
                df_jaccard_stability["SS_03"] = jaccard_value_ss_03
                df_jaccard_stability["SS_05"] = jaccard_value_ss_05
                df_jaccard_stability["SS_08"] = jaccard_value_ss_08
                df_jaccard_stability["Lasso"] = jaccard_value_lasso

                df_jaccard_stability.to_csv(Path(folder, f"Jaccard_stability_NbSamples={n_samples}.csv"))

                # Decoy Jaccard metrics for stability
                jaccard_matrix_decoy = jaccard_matrix(all_stable_features_decoy, )
                # df_jaccard_stab = pd.DataFrame(jaccard_matrix_decoy)
                # df_jaccard_stab.to_csv(Path(folder,f"Jaccard_stability_STABL_{n_samples}.csv"))
                iqr_jaccard_decoy = np.percentile(jaccard_matrix_decoy, [75, 25])
                df_results.loc[n_samples, 'Decoy_Jaccard_Stab_Median'] = np.median(jaccard_matrix_decoy)
                df_results.loc[n_samples, 'Decoy_Jaccard_Stab_Q1'] = iqr_jaccard_decoy[1]
                df_results.loc[n_samples, 'Decoy_Jaccard_Stab_Q3'] = iqr_jaccard_decoy[0]

                # Decoy Jaccard metrics for performance
                iqr_jaccard_ground_decoy = np.percentile(jaccard_ground_truth_decoy, [75, 25])
                df_results.loc[n_samples, 'Decoy_Jaccard_Perf_Median'] = np.median(jaccard_ground_truth_decoy)
                df_results.loc[n_samples, 'Decoy_Jaccard_Perf_Q1'] = iqr_jaccard_ground_decoy[1]
                df_results.loc[n_samples, 'Decoy_Jaccard_Perf_Q3'] = iqr_jaccard_ground_decoy[0]

                # Lasso Jaccard metrics for stability
                jaccard_matrix_lasso = jaccard_matrix(all_lasso_features)
                # df_jaccard_stab = pd.DataFrame(jaccard_matrix_lasso)
                # df_jaccard_stab.to_csv(Path(folder,f"Jaccard_stability_Lasso_{n_samples}.csv"))
                iqr_jaccard_lasso = np.percentile(jaccard_matrix_lasso, [75, 25])
                df_results.loc[n_samples, 'Lasso_Jaccard_Stab_Median'] = np.median(jaccard_matrix_lasso)
                df_results.loc[n_samples, 'Lasso_Jaccard_Stab_Q1'] = iqr_jaccard_lasso[1]
                df_results.loc[n_samples, 'Lasso_Jaccard_Stab_Q3'] = iqr_jaccard_lasso[0]

                # Lasso Jaccard metric for performance
                iqr_jaccard_ground_lasso = np.percentile(jaccard_ground_truth_lasso, [75, 25])
                df_results.loc[n_samples, 'Lasso_Jaccard_Perf_Median'] = np.median(jaccard_ground_truth_lasso)
                df_results.loc[n_samples, 'Lasso_Jaccard_Perf_Q1'] = iqr_jaccard_ground_lasso[1]
                df_results.loc[n_samples, 'Lasso_Jaccard_Perf_Q3'] = iqr_jaccard_ground_lasso[0]

                # SS03 Jaccard metrics for stability
                jaccard_matrix_ss_03 = jaccard_matrix(all_stable_features_ss_03)
                # df_jaccard_stab = pd.DataFrame(jaccard_matrix_ss_03)
                # df_jaccard_stab.to_csv(Path(folder,f"Jaccard_stability_SS_03_{n_samples}.csv"))
                iqr_jaccard_ss_03 = np.percentile(jaccard_matrix_ss_03, [75, 25])
                df_results.loc[n_samples, 'SS_03_Jaccard_Stab_Median'] = np.median(jaccard_matrix_ss_03)
                df_results.loc[n_samples, 'SS_03_Jaccard_Stab_Q1'] = iqr_jaccard_ss_03[1]
                df_results.loc[n_samples, 'SS_03_Jaccard_Stab_Q3'] = iqr_jaccard_ss_03[0]

                # SS03 Jaccard metric for performance
                iqr_jaccard_ground_ss_03 = np.percentile(jaccard_ground_truth_ss_03, [75, 25])
                df_results.loc[n_samples, 'SS_03_Jaccard_Perf_Median'] = np.median(jaccard_ground_truth_ss_03)
                df_results.loc[n_samples, 'SS_03_Jaccard_Perf_Q1'] = iqr_jaccard_ground_ss_03[1]
                df_results.loc[n_samples, 'SS_03_Jaccard_Perf_Q3'] = iqr_jaccard_ground_ss_03[0]

                # SS05 Jaccard metrics for stability
                jaccard_matrix_ss_05 = jaccard_matrix(all_stable_features_ss_05)
                # df_jaccard_stab = pd.DataFrame(jaccard_matrix_ss_05)
                # df_jaccard_stab.to_csv(Path(folder,f"Jaccard_stability_SS_05_{n_samples}.csv"))
                iqr_jaccard_ss_05 = np.percentile(jaccard_matrix_ss_05, [75, 25])
                df_results.loc[n_samples, 'SS_05_Jaccard_Stab_Median'] = np.median(jaccard_matrix_ss_05)
                df_results.loc[n_samples, 'SS_05_Jaccard_Stab_Q1'] = iqr_jaccard_ss_05[1]
                df_results.loc[n_samples, 'SS_05_Jaccard_Stab_Q3'] = iqr_jaccard_ss_05[0]

                # SS05 Jaccard metric for performance
                iqr_jaccard_ground_ss_05 = np.percentile(jaccard_ground_truth_ss_05, [75, 25])
                df_results.loc[n_samples, 'SS_05_Jaccard_Perf_Median'] = np.median(jaccard_ground_truth_ss_05)
                df_results.loc[n_samples, 'SS_05_Jaccard_Perf_Q1'] = iqr_jaccard_ground_ss_05[1]
                df_results.loc[n_samples, 'SS_05_Jaccard_Perf_Q3'] = iqr_jaccard_ground_ss_05[0]

                # SS08 Jaccard metrics for stability
                jaccard_matrix_ss_08 = jaccard_matrix(all_stable_features_ss_08)
                # df_jaccard_stab = pd.DataFrame(jaccard_matrix_ss_08)
                # df_jaccard_stab.to_csv(Path(folder,f"Jaccard_stability_SS_08_{n_samples}.csv"))
                iqr_jaccard_ss_08 = np.percentile(jaccard_matrix_ss_08, [75, 25])
                df_results.loc[n_samples, 'SS_08_Jaccard_Stab_Median'] = np.median(jaccard_matrix_ss_08)
                df_results.loc[n_samples, 'SS_08_Jaccard_Stab_Q1'] = iqr_jaccard_ss_08[1]
                df_results.loc[n_samples, 'SS_08_Jaccard_Stab_Q3'] = iqr_jaccard_ss_08[0]

                # SS08 Jaccard metric for performance
                iqr_jaccard_ground_ss_08 = np.percentile(jaccard_ground_truth_ss_08, [75, 25])
                df_results.loc[n_samples, 'SS_08_Jaccard_Perf_Median'] = np.median(jaccard_ground_truth_ss_08)
                df_results.loc[n_samples, 'SS_08_Jaccard_Perf_Q1'] = iqr_jaccard_ground_ss_08[1]
                df_results.loc[n_samples, 'SS_08_Jaccard_Perf_Q3'] = iqr_jaccard_ground_ss_08[0]

                # Lasso average nb features
                avg_selected_features_lasso = np.mean([len(i) for i in all_lasso_features])
                df_results.loc[n_samples, "Avg_nb_features_Lasso"] = avg_selected_features_lasso

                # Decoy average nb features
                avg_selected_features_decoy = np.mean([len(i) for i in all_stable_features_decoy])
                df_results.loc[n_samples, "Avg_nb_features_Decoy"] = avg_selected_features_decoy

                # SS03 average nb features
                avg_selected_features_ss_03 = np.mean([len(i) for i in all_stable_features_ss_03])
                df_results.loc[n_samples, "Avg_nb_features_SS_03"] = avg_selected_features_ss_03

                # SS05 average nb features
                avg_selected_features_ss_05 = np.mean([len(i) for i in all_stable_features_ss_05])
                df_results.loc[n_samples, "Avg_nb_features_SS_05"] = avg_selected_features_ss_05

                # SS08 average nb features
                avg_selected_features_ss_08 = np.mean([len(i) for i in all_stable_features_ss_08])
                df_results.loc[n_samples, "Avg_nb_features_SS_08"] = avg_selected_features_ss_08

            folder_name = f'./{result_folder_title}/NbFeatures={n_features}_NbInfo={n_info}/'
            df_results.to_csv(folder_name + "df_results.csv")
            df_thresholds.to_csv(folder_name + "df_threshold.csv")

            save_jaccard_decoy_lasso(df_results, folder_name + 'Jaccard/')
            save_jaccard_decoy_stab(df_results, folder_name + 'Jaccard/')
            save_MAE_scores(df_results, folder_name + 'Scores/MAE/')
            save_MSE_scores(df_results, folder_name + 'Scores/MSE/')
            save_R2_scores(df_results, folder_name + 'Scores/R2/')
            save_nb_features_plot(df_results, folder_name + 'Avg features graphs/', nb_info=n_info)
