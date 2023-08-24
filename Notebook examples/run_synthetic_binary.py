import numpy as np
import pandas as pd
from stabl import data
from stabl.synthetic import synthetic_benchmark_feature_selection
from sklearn.model_selection import RepeatedStratifiedKFold, GroupShuffleSplit, GridSearchCV, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from stabl.stabl import Stabl
from sklearn.base import clone
from stabl.asgl import ALogitLasso, ALasso

from stabl.metrics import jaccard_matrix
from stabl.stacked_generalization import stacked_multi_omic

from stabl.pipelines_utils import save_plots, compute_scores_table, compute_pvalues_table, BenchmarkWrapper
from tqdm.autonotebook import tqdm
from sklearn.cluster import KMeans

chosen_inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

artificial_type = "knockoff"
task_type = "linear"

# Lasso
lasso = LogisticRegression(penalty="l1", max_iter=int(1e6), random_state=42, solver="liblinear")
lasso_cv = GridSearchCV(
    lasso, scoring="roc_auc", param_grid={"C": np.logspace(-2, 2, 30)}, cv=chosen_inner_cv, n_jobs=-1
)

# ElasticNet
en = LogisticRegression(penalty="elasticnet", max_iter=int(1e4), random_state=42, solver="saga")
en_params = {"C": np.logspace(-2, 1, 5), "l1_ratio": [.5, .7, .9]}
en_cv = GridSearchCV(en, param_grid=en_params, scoring="roc_auc", cv=chosen_inner_cv, n_jobs=-1)

# ALasso
alasso = ALogitLasso(penalty="l1", solver="liblinear", max_iter=int(1e6), random_state=42)
alasso_cv = GridSearchCV(alasso, scoring="roc_auc", param_grid={"C": np.logspace(-2, 2, 30)}, cv=chosen_inner_cv, n_jobs=-1)

# Stabl
stabl = Stabl(
    lasso, 
    n_bootstraps=100, 
    artificial_type=artificial_type, 
    artificial_proportion=1., 
    replace=False, 
    fdr_threshold_range=np.arange(0.1, 1, 0.01), 
    sample_fraction=0.5, 
    random_state=42, 
    lambda_grid=None, 
    verbose=0
)

stabl_rp = clone(stabl).set_params(artificial_type="random_permutation")

stabl_alasso = clone(stabl).set_params(
    base_estimator=alasso, 
    lambda_grid=None, 
    verbose=0
)
stabl_alasso_rp = clone(stabl_alasso).set_params(artificial_type="random_permutation")

stabl_en = clone(stabl).set_params(
    base_estimator=en, 
    lambda_grid=[
        {"C": np.logspace(-2, 1, 5), "l1_ratio": [.5]},
        {"C": np.logspace(-2, 0, 5), "l1_ratio": [.7]},
        {"C": np.logspace(-2, 0, 5), "l1_ratio": [.9]},
    ], 
    verbose=0)
stabl_en_rp = clone(stabl_en).set_params(artificial_type="random_permutation")

estimators = {
    "lasso" : lasso_cv,
    "stabl_lasso" : stabl,
    "stabl_lasso_rp" : stabl_rp,
    "alasso" : alasso_cv,
    "stabl_alasso" : stabl_alasso,
    "stabl_alasso_rp" : stabl_alasso_rp,
    "en" : en_cv,
    "stabl_en" : stabl_en,
    "stabl_en_rp" : stabl_en_rp,
}


for feat_type in ["normal", "NB", "ZINB"]:
    for corr in ["no_corr", "low_corr", "medium_corr", "high_corr"]:
        X = pd.read_csv(f"./Norta/{feat_type} {corr}.csv", index_col=0)

        for i,j in estimators.items():
            if isinstance(j,Stabl):
                j.set_params(**{
                    "feat_type" : feat_type,
                    "corr" : corr
                })

        synthetic_benchmark_feature_selection(
                X=np.array(X),
                estimators=estimators,
                n_informative_list=[10, 25, 50],
                n_samples_list=[50, 75, 100, 150, 200, 300, 400, 600, 800, 1000],
                n_experiments=30,
                result_folder_title=f"./Synthetic Binary/Synthetic_{task_type}_{feat_type}_{corr}",
                f_number=[0.1, 0.5, 1],
                output_type = "binary",
                verbose=0b01111,
                input_type=feat_type,
                snr=2,
                scale_u=2,
                base_estim = ["alasso", "lasso", "en"],
            )
