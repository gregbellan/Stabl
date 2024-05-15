import numpy as np
from stabl import data
from stabl.multi_omic_pipelines import multi_omic_stabl_cv
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, RepeatedKFold
from sklearn.linear_model import Lasso, ElasticNet
from stabl.stabl import Stabl
from stabl.adaptive import ALasso
from sklearn.base import clone

outter_cv = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
chosen_inner_cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

artificial_type = "knockoff"

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
alasso_cv = GridSearchCV(
    alasso, scoring="r2", param_grid={"alpha": np.logspace(-2, 2, 30)}, cv=chosen_inner_cv, n_jobs=-1
)

# Stabl
stabl = Stabl(
    base_estimator=lasso,
    n_bootstraps=300,
    artificial_type=artificial_type,
    artificial_proportion=1.,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=42,
    lambda_grid={"alpha": np.logspace(0, 2, 10)},
    verbose=1
)

stabl_alasso = clone(stabl).set_params(
    base_estimator=alasso,
    lambda_grid={"alpha": np.logspace(0, 2, 10)},
    verbose=1
)
stabl_en = clone(stabl).set_params(
    base_estimator=en,
    lambda_grid=[
        {"alpha": np.logspace(0.5, 2, 5), "l1_ratio": [.9]}
    ],
    verbose=1)

estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "stabl_lasso": stabl,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en
}

models = [
    "STABL Lasso", "Lasso",
    "STABL ALasso", "ALasso",
    "STABL ElasticNet", "ElasticNet"
]

X_train, X_valid, y_train, y_valid, ids, task_type = data.load_onset_of_labor_cv("./Sample Data/Onset of Labor")

print("Run CV on onset of labor dataset with 3 omics")
multi_omic_stabl_cv(
    data_dict=X_train,
    y=y_train,
    outer_splitter=outter_cv,
    estimators=estimators,
    task_type=task_type,
    save_path="./Benchmarks results/Results OOL/CyPrMe",
    outer_groups=ids,
    early_fusion=True,
    late_fusion=True,
    models=models,
    n_iter_lf=1000
)
