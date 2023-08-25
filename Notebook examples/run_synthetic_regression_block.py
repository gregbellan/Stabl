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
from groupyr import LogisticSGL, SGL
from tqdm.autonotebook import tqdm
from sklearn.cluster import KMeans
from synthetic_data import load_data

chosen_inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

artificial_type = "knockoff"
task_type = "linear"

# SGL
sgl = SGL(max_iter=int(1e3), l1_ratio=0.5)
sgl_cv = GridSearchCV(sgl, scoring="r2", param_grid={"alpha": np.logspace(-2, 2, 5), "l1_ratio": [.5, .7, .9]}, cv=chosen_inner_cv, n_jobs=-1)

sgl_groups = [np.arange(i, i+5) for i in np.arange(0, 996, 5)]
stabl_sgl = Stabl(
    base_estimator=sgl,
    n_bootstraps=50,
    artificial_type=artificial_type,
    artificial_proportion=1.,
    replace=False,
    fdr_threshold_range=np.arange(0.1, 1, 0.01),
    sample_fraction=0.5,
    random_state=42,
    perc_corr_group_threshold=None,
    sgl_groups=sgl_groups,
    lambda_grid=[
        {"alpha": np.logspace(-1.5, 1, 5), "l1_ratio": [.5]},
        {"alpha": np.logspace(-2, 1, 5), "l1_ratio": [.7]},
        {"alpha": np.logspace(-2, 1, 5), "l1_ratio": [.9]}
    ],
    verbose=0
)
stabl_sgl_rp = clone(stabl_sgl).set_params(artificial_type="random_permutation")

estimators = {
    "sgl": sgl_cv,
    "stabl_sgl": stabl_sgl,
    "stabl_sgl_rp": stabl_sgl_rp,
}

for feat_type in ["normal", "NB", "ZINB"]:
    for corr in ["no_corr", "low_corr", "medium_corr", "high_corr"]:
        for n_info in tqdm([10, 25, 50], desc="n_info"):
            X = load_data(1000, n_info, feat_type, corr, use_blocks=True)

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
                result_folder_title=f"./Synthetic Linear (Block)/Synthetic_{task_type}_{feat_type}_{corr}",
                f_number=[0.1, 0.5, 1],
                output_type="linear",
                verbose=0b01101,
                input_type=feat_type,
                snr=2,
                scale_u=2,
                sgl_groups=sgl_groups,
                sgl_corr_percentile=None,
                base_estim=["sgl"]
            )
