import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression


class ALasso(Lasso):
    """_summary_

    Parameters
    ----------
    BaseEstimator : _type_
        _description_
    """

    def __init__(
        self,
        n_iter_lasso=2,
        alpha=1.0,
        *,
        fit_intercept=True,
        precompute=False,
        copy_X=True,
        max_iter=int(1e6),
        tol=1e-4,
        warm_start=False,
        positive=False,
        random_state=None,
        selection="cyclic",
    ):
        self.n_iter_lasso = n_iter_lasso
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection

        )

    def fit(self, X, y):
        """

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        """
        n_features = X.shape[1]
        weights = np.ones(n_features)

        for _ in range(self.n_iter_lasso):
            X_w = X / weights
            super().fit(X_w, y)
            self.coef_ = self.coef_ / weights
            weights = 1. / \
                (2. * np.sqrt(np.abs(self.coef_)) + np.finfo(float).eps)
        return self


class ALogitLasso(LogisticRegression):
    """
    """

    def __init__(
        self,
        n_iter_lasso=2,
        penalty="l1",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight="balanced",
        random_state=None,
        solver="liblinear",
        max_iter=int(1e6),
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self.n_iter_lasso = n_iter_lasso
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio
        )

    def fit(self, X, y):
        """

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        """
        n_features = X.shape[1]
        weights = np.ones(n_features)
        
        for _ in range(self.n_iter_lasso):
            X_w = X / weights
            super().fit(X_w, y)
            self.coef_ = self.coef_ / weights
            weights = 1. / \
                (2. * np.sqrt(np.abs(self.coef_)) + np.finfo(float).eps)
                
        return self
