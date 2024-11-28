import os
import shutil
import numpy as np
from stabl import data
from stabl.multi_omic_pipelines import multi_omic_stabl_cv
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, RepeatedKFold
from sklearn.linear_model import Lasso, ElasticNet
from stabl.stabl import Stabl
from stabl.adaptive import ALasso
from sklearn.base import clone
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd

np.random.seed(42)

# Defining outer Cross-Validation (CV) loop and inner CV loop
# The outer loop is used as the general evaluation framework whereas the inner loop is used to tune models at each fold

outter_cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
chosen_inner_cv = RepeatedKFold(n_splits=5, n_repeats=5,  random_state=42)

artificial_type = "knockoff"  # or "random_permutation"

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
alasso_cv = GridSearchCV(
    alasso, scoring="r2", param_grid={"alpha": np.logspace(-2, 2, 30)}, cv=chosen_inner_cv, n_jobs=-1
)

# RandomForest
rf = RandomForestRegressor(random_state=42)
rf_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, 7]}
rf_cv = GridSearchCV(
    rf, scoring='r2', param_grid=rf_grid, cv=chosen_inner_cv, n_jobs=-1
)

# XGBoost
xgb = XGBRegressor(random_state=42)
xgb_grid = {"min_child_weight": [1, 2], "max_depth": [3, 5, 7]}
xgb_cv = GridSearchCV(
    xgb, scoring='r2', param_grid=xgb_grid, cv=chosen_inner_cv, n_jobs=-1
)

# Stabl
stabl_lasso = Stabl(
    base_estimator=lasso,
    n_bootstraps=1000,
    artificial_type=artificial_type,
    artificial_proportion=1.,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=42,
    lambda_grid={"alpha": np.logspace(-2, 2, 30)},
    verbose=1
)

stabl_alasso = clone(stabl_lasso).set_params(
    base_estimator=alasso,
    lambda_grid={"alpha": np.logspace(-2, 2, 30)},
    verbose=1
)

stabl_en = clone(stabl_lasso).set_params(
    base_estimator=en,
    n_bootstraps=100,
    lambda_grid=[
        {"alpha": np.logspace(-2, 2, 10), "l1_ratio": [.5, .7, .9]}
    ],
    verbose=1
)

stabl_rf = clone(stabl_lasso).set_params(
    base_estimator=rf,
    n_bootstraps=100,
    lambda_grid=[rf_grid],
    verbose=1
)

stabl_xgb = clone(stabl_lasso).set_params(
    base_estimator=xgb,
    n_bootstraps=100,
    lambda_grid=[xgb_grid],
    verbose=1
)

estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "rf": rf_cv,
    "xgb": xgb_cv,
    
    "stabl_lasso": stabl_lasso,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
    "stabl_rf": stabl_rf,
    "stabl_xgb": stabl_xgb
}
models = [
    "Lasso",
    "ALasso",
    "ElasticNet",
    "RandomForest",
    "XGBoost",
    
    "STABL Lasso",
    "STABL ALasso",
    "STABL ElasticNet",
    "STABL RandomForest",
    "STABL XGBoost"
]

X_train, X_valid, y_train, y_valid, ids, task_type = data.load_onset_of_labor("./Sample Data/Onset of Labor")

save_path = "./Benchmarks results/Regression Onset of Labor"

if os.path.exists(save_path):
    shutil.rmtree(save_path)

print("Run CV on Onset of Labor Regression dataset")
print(ids)
multi_omic_stabl_cv(
    data_dict=X_train,
    y=y_train,
    outer_splitter=outter_cv,
    estimators=estimators,
    task_type=task_type,
    save_path=save_path,
    outer_groups=ids,
    early_fusion=False,
    late_fusion=True,
    n_iter_lf=1000,
    models=models
)