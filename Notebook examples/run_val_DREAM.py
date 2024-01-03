import numpy as np
from stabl import data
from stabl.multi_omic_pipelines import multi_omic_stabl
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from stabl.stabl import Stabl
from stabl.adaptive import ALogitLasso
from sklearn.base import clone

random_seed = 42
np.random.seed(random_seed)
chosen_inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

artificial_type = "knockoff"
X_train, X_valid, y_train, y_valid, ids, task_type = data.load_dream("./Sample Data/Dream")
y_train = y_train.astype(int)
for name, df in X_train.items():
    df.columns = df.columns.str.replace("/", "_", regex=True)
    X_train[name] = df

# Lasso
lasso = LogisticRegression(
    penalty="l1", class_weight="balanced", max_iter=int(1e6), solver="liblinear", random_state=random_seed
)
lasso_cv = GridSearchCV(
    lasso, param_grid={"C": np.logspace(-3, 0, 30)}, scoring="roc_auc", cv=chosen_inner_cv, n_jobs=-1
)


# ElasticNet
en = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    class_weight='balanced',
    max_iter=int(1e4),
    random_state=random_seed
)
en_params = {"C": np.logspace(-3, 0, 10), "l1_ratio": [.5, .7, .9]}
en_cv = GridSearchCV(en, param_grid=en_params, scoring="roc_auc", cv=chosen_inner_cv, n_jobs=-1)

# ALasso
alasso = ALogitLasso(
    penalty="l1", solver="liblinear", max_iter=int(1e6), class_weight='balanced', random_state=random_seed
)
alasso_cv = GridSearchCV(
    alasso, scoring='roc_auc', param_grid={"C": np.logspace(-3, 0, 30)}, cv=chosen_inner_cv, n_jobs=-1
)

# Stabl
stabl = Stabl(
    base_estimator=lasso,
    n_bootstraps=100,
    artificial_type=artificial_type,
    artificial_proportion=.5,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=random_seed,
    lambda_grid={"C": np.linspace(0.004, 0.4, 30)},
    verbose=1
)
stabl_rp = clone(stabl).set_params(artificial_type="random_permutation")

stabl_alasso = clone(stabl).set_params(
    base_estimator=alasso,
    lambda_grid={"C": np.linspace(0.004, 4, 30)},
    verbose=1
)

stabl_en = clone(stabl).set_params(
    base_estimator=en,
    n_bootstraps=50,
    lambda_grid=[
        {"C": np.logspace(-3, -1, 5), "l1_ratio": [.9]}
    ],
    verbose=1
)

estimators = {
    "lasso": lasso_cv,
    "alasso": alasso_cv,
    "en": en_cv,
    "stabl_lasso": stabl,
    "stabl_alasso": stabl_alasso,
    "stabl_en": stabl_en,
}

models = [
    "STABL Lasso",
    "Lasso",
    "STABL ALasso", "ALasso",
    "STABL ElasticNet", "ElasticNet"
]

print("Run TV on dream dataset")
multi_omic_stabl(
    data_dict=X_train,
    y=y_train,
    estimators=estimators,
    task_type=task_type,
    save_path="./Benchmarks results/Results Dream",
    groups=ids,
    early_fusion=True,
    X_test=X_valid,
    y_test=y_valid,
    n_iter_lf=1000,
    models=models,
)
