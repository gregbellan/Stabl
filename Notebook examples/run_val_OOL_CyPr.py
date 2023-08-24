import numpy as np
import pandas as pd
from stabl import data
from stabl.multi_omic_pipelines import multi_omic_stabl_cv, multi_omic_stabl
from sklearn.model_selection import RepeatedStratifiedKFold, GroupShuffleSplit, GridSearchCV, RepeatedKFold
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from stabl.stabl import Stabl, group_bootstrap
from stabl.asgl import ALogitLasso, ALasso
from groupyr import SGL, LogisticSGL
from sklearn.base import clone

chosen_inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
outter_cv = GroupShuffleSplit(n_splits=25, test_size=0.2, random_state=42)

artificial_type = "knockoff"
task_type = "regression"

# Regression

# Lasso
lasso = Lasso(max_iter=int(1e6), random_state=42)
lasso_cv = GridSearchCV(
    lasso, scoring="r2", param_grid={"alpha": np.logspace(-2, 2, 30)}, cv=chosen_inner_cv, n_jobs=-1
)

# ElasticNet
en = ElasticNet(max_iter=int(1e6), random_state=42)
en_params = {"alpha": np.logspace(-2, 2, 10), "l1_ratio": [.5, .7, .9]}
en_cv = GridSearchCV(en, param_grid=en_params, scoring="r2", cv=chosen_inner_cv, n_jobs=-1)

# ALasso
alasso = ALasso(max_iter=int(1e6), random_state=42)
alasso_cv = GridSearchCV(alasso, scoring="r2", param_grid={"alpha": np.logspace(-2, 2, 30)}, cv=chosen_inner_cv, n_jobs=-1)

# SGL
sgl = SGL(max_iter=int(1e3), l1_ratio=0.5)
sgl_cv = GridSearchCV(sgl, scoring="r2", param_grid={"alpha": np.logspace(-1, 2, 5), "l1_ratio": [.5, .7, .9]}, cv=chosen_inner_cv, n_jobs=-1)

# Stabl
stabl = Stabl(
    lasso,
    n_bootstraps=2000,
    artificial_type=artificial_type,
    artificial_proportion=1.,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=42,
    lambda_grid={"alpha": np.logspace(0, 2, 30)},
    verbose=1
)

stabl_alasso = clone(stabl).set_params(
    base_estimator=alasso,
    lambda_grid={"alpha": np.logspace(0, 2, 30)},
    verbose=1
)
stabl_en = clone(stabl).set_params(
    base_estimator=en,
    lambda_grid=[
        {"alpha": np.logspace(1, 2, 10), "l1_ratio": [.5]},
        {"alpha": np.logspace(0.5, 2, 10), "l1_ratio": [.7]},
        {"alpha": np.logspace(0.5, 2, 10), "l1_ratio": [.9]},
    ],
    verbose=1)

stabl_sgl = clone(stabl).set_params(
    base_estimator=sgl,
    n_bootstraps=50,
    perc_corr_group_threshold=99,
    lambda_grid=[
        {"alpha": np.logspace(1, 2, 10), "l1_ratio": [.5]},
        {"alpha": np.logspace(1, 2, 10), "l1_ratio": [.7]},
        {"alpha": np.logspace(1, 2, 10), "l1_ratio": [.9]}
    ],
    verbose=1
)

estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "sgl": sgl_cv,
    "stabl_lasso": stabl,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
    "stabl_sgl": stabl_sgl,
}

models = [
    "STABL Lasso",
    "Lasso",
    "STABL ALasso",
    "ALasso",
    "STABL ElasticNet",
    "ElasticNet",
    # "STABL SGL-90",
    # "STABL SGL-95",
    # "SGL-90",
    # "SGL-95",
]

X_train, X_valid, y_train, y_valid, ids, task_type = data.load_onset_of_labor("../Sample Data/Onset of Labor")

print("Run Validation on onset of labor dataset")
multi_omic_stabl(
    X_train,
    y_train,
    splitter=outter_cv,
    estimators=estimators,
    task_type=task_type,
    save_path="./Results OOL/CyPr",
    groups=ids,
    early_fusion=True,
    X_test=X_valid,
    y_test=y_valid,
    n_iter_lf=1000,
    models=models,
    sgl_corr_percentile=[90, 95]
)
