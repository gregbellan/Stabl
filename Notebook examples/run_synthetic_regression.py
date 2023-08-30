from julia.api import Julia
jl = Julia(compiled_modules=False)
import numpy as np
import pandas as pd
from stabl import data
from stabl.synthetic import synthetic_benchmark_feature_selection
from sklearn.model_selection import RepeatedStratifiedKFold, GroupShuffleSplit, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from stabl.stabl import Stabl
from sklearn.base import clone
from stabl.asgl import ALogitLasso, ALasso

from stabl.metrics import jaccard_matrix
from stabl.stacked_generalization import stacked_multi_omic

from stabl.pipelines_utils import save_plots, compute_scores_table, compute_pvalues_table, BenchmarkWrapper
from tqdm.autonotebook import tqdm
from synthetic_data import load_data

chosen_inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

artificial_type = "knockoff"
task_type = "linear"

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
        {"alpha": np.logspace(-1, 2, 5), "l1_ratio": [.5]},
        {"alpha": np.logspace(-1, 2, 5), "l1_ratio": [.7]},
        {"alpha": np.logspace(-2, 2, 5), "l1_ratio": [.9]},
    ],
    verbose=0)
stabl_en_rp = clone(stabl_en).set_params(artificial_type="random_permutation")


# stabl_ss_3 = clone(stabl).set_params(artificial_type=None, hard_threshold=0.3)
# stabl_ss_5 = clone(stabl).set_params(artificial_type=None, hard_threshold=0.5)
# stabl_ss_8 = clone(stabl).set_params(artificial_type=None, hard_threshold=0.8)

estimators = {
    "lasso": lasso_cv,
    "stabl_lasso": stabl,
    "stabl_lasso_rp": stabl_rp,
    "alasso": alasso_cv,
    "stabl_alasso": stabl_alasso,
    "stabl_alasso_rp": stabl_alasso_rp,
    "en": en_cv,
    "stabl_en": stabl_en,
    "stabl_en_rp": stabl_en_rp,
    # "stabl_lasso_ss_3" : stabl_ss_3,
    # "stabl_lasso_ss_5" : stabl_ss_5,
    # "stabl_lasso_ss_8" : stabl_ss_8,
}

for feat_type in ["normal", "NB", "ZINB"]:
    for corr in ["no_corr", "low_corr", "medium_corr", "high_corr"]:
        for n_info in tqdm([10, 25, 50], desc="n_info"):
            X = load_data(1000, n_info, feat_type, corr, use_blocks=False)

            for i, j in estimators.items():
                if isinstance(j, Stabl):
                    j.set_params(**{
                        "feat_type": feat_type,
                        "corr": corr
                    })

            synthetic_benchmark_feature_selection(
                X=np.array(X),
                estimators=estimators,
                n_informative_list=[n_info],
                n_samples_list=[50, 75, 100, 150, 200, 300, 400, 600, 800, 1000],
                n_experiments=30,
                result_folder_title=f"./Synthetic Linear/Synthetic_{task_type}_{feat_type}_{corr}",
                f_number=[0.1, 0.5, 1],
                output_type="linear",
                verbose=0b01101,
                input_type=feat_type,
                snr=2,
                scale_u=2,
                base_estim=["alasso", "lasso", "en"]
            )
