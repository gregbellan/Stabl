import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LogisticRegression, LinearRegression, ElasticNetCV,\
    Lasso
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

from .metrics import jaccard_matrix
from .preprocessing import LowInfoFilter, remove_low_info_samples
from .stabl import save_stabl_results

from scipy import stats
from scipy.stats import mannwhitneyu

from .pipelines_utils import save_plots, compute_scores_table

lasso = Lasso(max_iter=int(1e6))
lasso_cv = LassoCV(n_alphas=50, max_iter=int(1e6), n_jobs=-1)
en_cv = ElasticNetCV(n_alphas=50, max_iter=int(1e6), n_jobs=-1, l1_ratio=.5)

logit_lasso_cv = LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=np.logspace(-2, 2, 50),
                                      max_iter=int(1e6), class_weight="balanced", scoring="roc_auc",
                                      n_jobs=-1
                                      )

logit_en_cv = LogisticRegressionCV(penalty="elasticnet", solver="saga", Cs=np.logspace(-2, 2, 50),
                                   max_iter=int(1e6), class_weight="balanced", scoring="roc_auc",
                                   n_jobs=-1, l1_ratios=[.5]
                                   )

logit = LogisticRegression(penalty=None, class_weight="balanced", max_iter=int(1e6))
linreg = LinearRegression()

preprocessing = Pipeline(
    steps=[
        ("variance", VarianceThreshold(0.01)),
        ("lif", LowInfoFilter()),
        ("impute", SimpleImputer(strategy="median")),
        ("std", StandardScaler())
    ]
)


def single_omic_stabl_cv(
        X,
        y,
        outer_splitter,
        stabl,
        stability_selection,
        task_type,
        save_path,
        outer_groups=None
):
    """

    Parameters
    ----------
    X
    stability_selection: Stabl
    data_dict: dict
        Dictionary containing the input omic-files.

    y: pd.Series
        pandas Series containing the outcomes for the use case. Note that y should contains the union of outcomes for
        the data_dict.

    outer_splitter: sklearn.model_selection._split.BaseCrossValidator
        Outer cross validation splitter

    stabl: SurgeLibrary.stability_selection.StabilitySelection
        STABL used to select features at each fold of the cross validation and for each omic.

    task_type: str
        Can either be "binary" for binary classification or "regression" for regression tasks.

    save_path: Path or str
        Where to save the results

    outer_groups: pd.Series, default=None
        If used, should be the same size as y and should indicate the groups of the samples.

    Returns
    -------

    """
    models = ["STABL", "SS 03", "SS 05", "SS 08", "Lasso", "Lasso 1SE", "ElasticNet"]

    os.makedirs(Path(save_path, "Training CV"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    # Initializing the df containing the data of all omics
    predictions_dict = dict()
    selected_features_dict = dict()

    for model in models:
        predictions_dict[model] = pd.DataFrame(data=None, index=y.index)
        selected_features_dict[model] = []

    i = 1
    for train, test in outer_splitter.split(X, y, groups=outer_groups):
        print(f" Iteration {i} over {outer_splitter.get_n_splits()} ".center(80, '*'), "\n")
        train_idx, test_idx = y.iloc[train].index, y.iloc[test].index

        fold_selected_features = dict()
        for model in models:
            fold_selected_features[model] = []

        print(f"{len(train_idx)} train samples, {len(test_idx)} test samples")

        X_tmp = X.drop(index=test_idx, errors="ignore")

        # Preprocessing of X_tmp
        X_tmp = remove_low_info_samples(X_tmp)
        y_tmp = y.loc[X_tmp.index]

        X_tmp_std = pd.DataFrame(
            data=preprocessing.fit_transform(X_tmp),
            index=X_tmp.index,
            columns=preprocessing.get_feature_names_out()
        )

        # __STABL__
        if task_type == "binary":
            min_C = l1_min_c(X_tmp_std, y_tmp)
            lambda_grid = np.linspace(min_C, min_C * 100, 10)
            stabl.set_params(lambda_grid=lambda_grid)
            stability_selection.set_params(lambda_grid=lambda_grid)

        stabl.fit(X_tmp_std, y_tmp)
        tmp_sel_features = list(stabl.get_feature_names_out())
        fold_selected_features["STABL"] = tmp_sel_features

        print(
            f"STABL finished ({X_tmp.shape[0]} samples);"
            f" {len(tmp_sel_features)} features selected\n"
        )

        # __SS__
        stability_selection.fit(X_tmp_std, y_tmp)
        fold_selected_features["SS 03"] = list(stability_selection.get_feature_names_out(new_hard_threshold=.3))
        fold_selected_features["SS 05"] = list(stability_selection.get_feature_names_out(new_hard_threshold=.5))
        fold_selected_features["SS 08"] = list(stability_selection.get_feature_names_out(new_hard_threshold=.8))

        selected_features_dict["STABL"].append(fold_selected_features["STABL"])
        selected_features_dict[f"SS 03"].append(fold_selected_features["SS 03"])
        selected_features_dict[f"SS 05"].append(fold_selected_features["SS 05"])
        selected_features_dict[f"SS 08"].append(fold_selected_features["SS 08"])

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"This fold: {len(fold_selected_features['STABL'])} features selected for STABL")
        print(f"This fold: {len(fold_selected_features['SS 03'])} features selected for SS 03")
        print(f"This fold: {len(fold_selected_features['SS 05'])} features selected for SS 05")
        print(f"This fold: {len(fold_selected_features['SS 08'])} features selected for SS 08")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

        for model in ["STABL", "SS 03", "SS 05", "SS 08"]:

            X_train = X.loc[train_idx, fold_selected_features[model]]
            X_test = X.loc[test_idx, fold_selected_features[model]]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            if len(fold_selected_features[model]) > 0:
                # Standardization
                std_pipe = Pipeline(
                    steps=[
                        ('imputer', SimpleImputer(strategy="median")),
                        ('std', StandardScaler())
                    ]
                )

                X_train = pd.DataFrame(
                    data=std_pipe.fit_transform(X_train),
                    index=X_train.index,
                    columns=X_train.columns
                )
                X_test = pd.DataFrame(
                    data=std_pipe.transform(X_test),
                    index=X_test.index,
                    columns=X_test.columns
                )

                # __Final Models__
                if task_type == "binary":
                    predictions = clone(logit).fit(X_train, y_train).predict_proba(X_test)[:, 1].flatten()

                elif task_type == "regression":
                    predictions = clone(linreg).fit(X_train, y_train).predict(X_test)

                else:
                    raise ValueError("task_type not recognized.")

                predictions_dict[model].loc[test_idx, f'Fold n°{i}'] = predictions

            else:
                if task_type == "binary":
                    predictions_dict[model].loc[test_idx, f'Fold n°{i}'] = [0.5] * len(test_idx)

                elif task_type == "regression":
                    predictions_dict[model].loc[test_idx, f'Fold n°{i}'] = [np.mean(y_train)] * len(test_idx)

                else:
                    raise ValueError("task_type not recognized.")

        # __other models__
        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        X_train = pd.DataFrame(
            data=preprocessing.fit_transform(X_train),
            columns=preprocessing.get_feature_names_out(),
            index=X_train.index
        )

        X_test = pd.DataFrame(
            data=preprocessing.transform(X_test),
            columns=preprocessing.get_feature_names_out(),
            index=X_test.index
        )

        # __Lasso__
        if task_type == "binary":
            inner_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
            model = clone(logit_lasso_cv).set_params(cv=inner_splitter)
            predictions = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        else:
            inner_splitter = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
            model = clone(lasso_cv).set_params(cv=inner_splitter)
            predictions = model.fit(X_train, y_train).predict(X_test)

        selected_features_dict["Lasso"].append(list(X_train.columns[np.where(model.coef_.flatten())]))
        predictions_dict["Lasso"].loc[test_idx, f"Fold n°{i}"] = predictions

        # __Lasso 1SE__
        if task_type == "binary":
            best_c_corr = model.C_[0] - model.scores_[True].std() / np.sqrt(inner_splitter.get_n_splits())
            model = LogisticRegression(penalty='l1', solver='liblinear', C=best_c_corr, class_weight='balanced',
                                       max_iter=1_000_000)

            predictions = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]

        selected_features_dict["Lasso 1SE"].append(list(X_train.columns[np.where(model.coef_.flatten())]))
        predictions_dict["Lasso 1SE"].loc[test_idx, f"Fold n°{i}"] = predictions

        # __EN__
        if task_type == "binary":
            model = clone(logit_en_cv).set_params(cv=inner_splitter)
            predictions = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]

        else:
            model = clone(en_cv).set_params(cv=inner_splitter)
            predictions = model.fit(X_train, y_train).predict(X_test)

        selected_features_dict["ElasticNet"].append(list(X_train.columns[np.where(model.coef_.flatten())]))
        predictions_dict["ElasticNet"].loc[test_idx, f"Fold n°{i}"] = predictions

        i += 1

    # __SAVING_RESULTS__

    if y.name is None:
        y.name = "outcome"

    summary_res_path = Path(save_path, "Summary")
    cv_res_path = Path(save_path, "Training CV")

    jaccard_matrix_dict = dict()
    formatted_features_dict = dict()

    for model in models:

        jaccard_matrix_dict[model] = jaccard_matrix(selected_features_dict[model])

        formatted_features_dict[model] = pd.DataFrame(
            data={
                "Fold selected features": selected_features_dict[model],
                "Fold nb of features": [len(el) for el in selected_features_dict[model]]
            },
            index=[f"Fold {i}" for i in range(outer_splitter.get_n_splits())]
        )
        formatted_features_dict[model].to_csv(Path(cv_res_path, f"Selected Features {model}.csv"))

    predictions_dict = {model: predictions_dict[model].median(axis=1) for model in predictions_dict.keys()}

    table_of_scores = compute_scores_table(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        selected_features_dict=formatted_features_dict
    )

    table_of_scores.to_csv(Path(summary_res_path, "Scores training CV.csv"))
    table_of_scores.to_csv(Path(cv_res_path, "Scores training CV.csv"))

    save_plots(
        predictions_dict=predictions_dict,
        y=y,
        task_type=task_type,
        save_path=cv_res_path
    )

    return predictions_dict


def single_omic_stabl(
        X,
        y,
        stabl,
        stability_selection,
        task_type,
        save_path,
        X_test=None,
        y_test=None
):
    """

    Parameters
    ----------
    stability_selection
    X

    y

    stabl: SurgeLibrary.stability_selection.StabilitySelection

    task_type

    save_path

    X_test: pd.DataFrame, default=None

    y_test: pd.Series, default=None

    Returns
    -------

    """
    models = ["STABL", "SS 03", "SS 05", "SS 08", "Lasso", "Lasso 1SE", "ElasticNet"]

    os.makedirs(Path(save_path, "Training-Validation"), exist_ok=True)
    os.makedirs(Path(save_path, "Summary"), exist_ok=True)

    predictions_dict = dict()
    selected_features_dict = dict()
    for model in models:
        selected_features_dict[model] = []

    X_omic_std = pd.DataFrame(
        data=preprocessing.fit_transform(X),
        index=X.index,
        columns=preprocessing.get_feature_names_out()
    )
    y_omic = y.loc[X_omic_std.index]

    stabl.fit(X_omic_std, y_omic)
    omic_selected_features = stabl.get_feature_names_out()
    selected_features_dict["STABL"] += list(omic_selected_features)

    print(f"STABL finished ; {len(omic_selected_features)} features selected")

    save_stabl_results(
        stabl=stabl,
        path=Path(save_path, "Training-Validation", f"STABL results"),
        df_X=X,
        y=y_omic,
        task_type=task_type
    )

    stability_selection.fit(X_omic_std, y_omic)
    selected_features_dict["SS 03"] += list(stability_selection.get_feature_names_out(new_hard_threshold=.3))
    save_stabl_results(stabl=stability_selection,
                       path=Path(save_path, "Training-Validation", f"SS 03 results"), df_X=X,
                       y=y_omic, task_type=task_type, new_hard_threshold=.3)
    selected_features_dict["SS 05"] += list(stability_selection.get_feature_names_out(new_hard_threshold=.5))
    save_stabl_results(stabl=stability_selection,
                       path=Path(save_path, "Training-Validation", f"SS 05 results"), df_X=X,
                       y=y_omic, task_type=task_type, new_hard_threshold=.5)
    selected_features_dict["SS 08"] += list(stability_selection.get_feature_names_out(new_hard_threshold=.8))
    save_stabl_results(stabl=stability_selection,
                       path=Path(save_path, "Training-Validation", f"SS 08 results"), df_X=X,
                       y=y_omic, task_type=task_type, new_hard_threshold=.8)

    final_prepro = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("std", StandardScaler())]
    )

    for model in ["STABL", "SS 03", "SS 05", "SS 08"]:
        if len(selected_features_dict[model]) > 0:
            X_train = X[selected_features_dict[model]]
            X_train_std = pd.DataFrame(
                data=final_prepro.fit_transform(X_train),
                index=X.index,
                columns=final_prepro.get_feature_names_out()
            )

            if task_type == "binary":
                base_linear_model = logit

            else:
                base_linear_model = linreg

            base_linear_model.fit(X_train_std, y)
            base_linear_model_coef = pd.DataFrame(
                {"Feature": selected_features_dict[model],
                 "Associated weight": base_linear_model.coef_.flatten()
                 }
            ).set_index("Feature")
            base_linear_model_coef.to_csv(Path(save_path, "Training-Validation", f"{model} coefficients.csv"))

            if X_test is not None:
                X_test_std = pd.DataFrame(
                    data=final_prepro.transform(X_test[selected_features_dict[model]]),
                    index=X_test.index,
                    columns=final_prepro.get_feature_names_out()
                )
                if task_type == "binary":
                    model_preds = base_linear_model.predict_proba(X_test_std)[:, 1]
                else:
                    model_preds = base_linear_model.predict(X_test_std)

                predictions_dict[model] = pd.Series(
                    model_preds,
                    index=y_test.index,
                    name=f"{model} predictions"
                )
        else:
            if X_test is not None:
                if task_type == "binary":
                    model_preds = [0.5]*len(y_test)
                else:
                    model_preds = [np.mean(y)] * len(y_test)

                predictions_dict[model] = pd.Series(
                    model_preds,
                    index=y_test.index,
                    name=f"{model} predictions"
                )

    # __Lasso__
    X_train_std = pd.DataFrame(
        data=preprocessing.fit_transform(X),
        index=X.index,
        columns=preprocessing.get_feature_names_out()
    )

    if task_type == "binary":
        inner_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        model_lasso = clone(logit_lasso_cv).set_params(cv=inner_splitter).fit(X_train_std, y)
    else:
        inner_splitter = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        model_lasso = clone(lasso_cv).set_params(cv=inner_splitter).fit(X_train_std, y)

    selected_features_dict["Lasso"] += list(X_train_std.columns[np.where(model_lasso.coef_.flatten())])

    lasso_coef = pd.DataFrame(
        {"Feature": selected_features_dict["Lasso"],
         "Associated weight": model_lasso.coef_.flatten()[np.where(model_lasso.coef_.flatten())]
         }
    ).set_index("Feature")
    lasso_coef.to_csv(Path(save_path, "Training-Validation", f"Lasso coefficients.csv"))

    # __Lasso 1SE__
    if task_type == "binary":
        best_c_corr = model_lasso.C_[0] - model_lasso.scores_[True].std() / np.sqrt(inner_splitter.get_n_splits())
        model_lasso1se = LogisticRegression(penalty='l1', solver='liblinear', C=best_c_corr,
                                            class_weight='balanced', max_iter=1_000_000
                                            ).fit(X_train_std, y)

    selected_features_dict["Lasso 1SE"] += list(X_train_std.columns[np.where(model_lasso1se.coef_.flatten())])

    lasso1se_coef = pd.DataFrame(
        {"Feature": selected_features_dict["Lasso 1SE"],
         "Associated weight": model_lasso1se.coef_.flatten()[np.where(model_lasso1se.coef_.flatten())]
         }
    ).set_index("Feature")
    lasso1se_coef.to_csv(Path(save_path, "Training-Validation", f"Lasso 1SE coefficients.csv"))

    # __EN__

    if task_type == "binary":
        model_en = clone(logit_en_cv).set_params(cv=inner_splitter).fit(X_train_std, y)
    else:
        model_en = clone(en_cv).set_params(cv=inner_splitter).fit(X_train_std, y)

    selected_features_dict["ElasticNet"] += list(X_train_std.columns[np.where(model_en.coef_.flatten())])

    en_coef = pd.DataFrame(
        {"Feature": selected_features_dict["ElasticNet"],
         "Associated weight": model_en.coef_.flatten()[np.where(model_en.coef_.flatten())]
         }
    ).set_index("Feature")
    en_coef.to_csv(Path(save_path, "Training-Validation", f"ElasticNet coefficients.csv"))

    if X_test is not None:
        X_test_std = pd.DataFrame(
            data=preprocessing.transform(X_test),
            index=X_test.index,
            columns=preprocessing.get_feature_names_out()
        )

        predictions_dict["Lasso"] = pd.Series(
            data=model_lasso.predict(X_test_std) if task_type == "regression" else model_lasso.predict_proba(X_test_std
                                                                                                             )[:, 1],
            index=y_test.index,
            name="Lasso predictions"
        )

        predictions_dict["Lasso 1SE"] = pd.Series(
            data=model_lasso1se.predict(X_test_std) if task_type == "regression" else model_lasso1se.predict_proba(
                X_test_std)[:, 1],
            index=y_test.index,
            name="Lasso 1SE predictions"
        )

        predictions_dict["ElasticNet"] = pd.Series(
            data=model_en.predict(X_test_std) if task_type == "regression" else model_en.predict_proba(
                X_test_std)[:, 1],
            index=y_test.index,
            name="ElasticNet predictions"
        )

        save_plots(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type,
            save_path=Path(save_path, "Training-Validation"),
        )

        validation_scores = compute_scores_table(
            predictions_dict=predictions_dict,
            y=y_test,
            task_type=task_type,
        )
        validation_scores.to_csv(Path(save_path, "Training-Validation", "Scores on Validation.csv"))
        validation_scores.to_csv(Path(save_path, "Summary", "Scores on Validation.csv"))

    return predictions_dict


def compute_features_table_stabl_vs_lasso(
        all_features_Lasso,
        all_features_STABL,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        task_type="binary"
):
    """

    Parameters
    ----------
    all_features_Lasso: array-like,
        Array of feature names selected by the Lasso

    all_features_STABL: array-like
        Array of feature names selected by STABL

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
    all_features = list(set(all_features_Lasso).union(set(all_features_STABL)))
    df_out = pd.DataFrame(data=False, index=all_features, columns=["Selected by Lasso", "Selected by STABL"])
    for feature in all_features:
        if feature in set(all_features_Lasso):
            df_out.loc[feature, "Selected by Lasso"] = True
        else:
            df_out.loc[feature, "Selected by Lasso"] = False

        if feature in set(all_features_STABL):
            df_out.loc[feature, "Selected by STABL"] = True
        else:
            df_out.loc[feature, "Selected by STABL"] = False

    if task_type == "binary":
        df_out["Train Mannwithney pvalues"] = [mannwhitneyu(X_train.loc[y_train == 0, i],
                                                            X_train.loc[y_train == 1, i],
                                                            nan_policy="omit")[1]
                                               for i in all_features]

        df_out["Train T-test pvalues"] = [stats.ttest_ind(X_train.loc[y_train == 1, i],
                                                          X_train.loc[y_train == 0, i],
                                                          nan_policy="omit")[1]
                                          for i in all_features]
    elif task_type == "regression":
        df_out["Train Pearson-r pvalues"] = [stats.pearsonr(X_train.loc[:, i].dropna(),
                                                            y_train.loc[X_train.loc[:, i].dropna().index]
                                                            )[1]
                                             for i in all_features]

        df_out["Train Spearman-r pvalues"] = [stats.spearmanr(X_train.loc[:, i].dropna(),
                                                              y_train.loc[X_train.loc[:, i].dropna().index]
                                                              )[1]
                                              for i in all_features]

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
