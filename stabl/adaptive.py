import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression


class ALasso(Lasso):
    """Implementation of the adaptive lasso algorithm (ALasso).
    This weighted Lasso updated weights iteratively.
    The implementation structure is inspired by the sklearn implementation of the Lasso algorithm.

    Parameters
    ----------
    n_iter_lasso : int, optional
        Number of iteration where a lasso is performed, by default 2

    alpha : float, default=1.0
        Constant that multiplies the L1 term, controlling regularization
        strength. `alpha` must be a non-negative float i.e. in `[0, inf)`.

        When `alpha = 0`, the objective is equivalent to ordinary least
        squares, solved by the `LinearRegression` object. For numerical
        reasons, using `alpha = 0` with the `ALasso` object is not advised.
        Instead, you should use the `LinearRegression` object.

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    precompute : bool or array-like of shape (n_features, n_features),\
                 default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. The Gram matrix can also be passed as argument.
        For sparse input this option is always ``False`` to preserve sparsity.

    copy_X : bool, default=True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, default=1_000_000
        The maximum number of iterations.

    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    positive : bool, default=False
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, default=None
        The seed of the pseudo random number generator that selects a random
        feature to update. Used when ``selection`` == 'random'.
        Pass an int for reproducible output across multiple function calls.

    selection : {'cyclic', 'random'}, default='cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
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
        self.coef_ = None
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

    def fit(self, X, y, **kwargs):
        """
        Fit model with coordinate descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        """
        n_features = X.shape[1]
        weights = np.ones(n_features)

        for _ in range(self.n_iter_lasso):
            X_w = X / weights
            super().fit(X_w, y)
            self.coef_ = self.coef_ / weights
            weights = 1. / (2. * np.sqrt(np.abs(self.coef_)) + np.finfo(float).eps)
        return self


class ALogitLasso(LogisticRegression):
    """
    Implementation of the adaptive lasso algorithm (ALasso) for logistic regression.
    It is a logistic regression with penalty with weights that are updated at each iteration.
    The implementation structure is inspired by the sklearn implementation of the LogisticRegression algorithm.

    Parameters
    ----------

    n_iter_lasso : int, optional
        Number of iteration where a lasso is performed, by default 2

    penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
        Specify the norm of the penalty:

        - `None`: no penalty is added;
        - `'l2'`: add a L2 penalty term, and it is the default choice;
        - `'l1'`: add a L1 penalty term;
        - `'elasticnet'`: both L1 and L2 penalty terms are added.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is only implemented for
        l2 penalty with liblinear solver. Prefer dual=False when
        n_samples > n_features.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default=1
        Useful only when the solver 'liblinear' is used
        and self.fit_intercept is set to True. In this case, x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equal to
        intercept_scaling is appended to the instance vector.
        The intercept becomes ``intercept_scaling * synthetic_feature_weight``.

        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    random_state : int, RandomState instance, default=None
        Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
        data.

    solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, default='lbfgs'
        Algorithm to use in the optimization problem. Default is 'lbfgs'.
        To choose a solver, you might want to consider the following aspects:
            - For small datasets, 'liblinear' is a good choice, whereas 'sag'
              and 'saga' are faster for large ones;
            - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
              'lbfgs' handle multinomial loss;
            - 'liblinear' is limited to one-versus-rest schemes.
            - 'newton-cholesky' is a good choice for `n_samples` >> `n_features`,
              especially with one-hot encoded categorical features with rare
              categories. Note that it is limited to binary classification and the
              one-versus-rest reduction for multiclass classification. Be aware that
              the memory usage of this solver has a quadratic dependency on
              `n_features` because it explicitly computes the Hessian matrix.

        .. warning::
           The choice of the algorithm depends on the penalty chosen.
           Supported penalties by solver:
           - 'lbfgs'           -   ['l2', None]
           - 'liblinear'       -   ['l1', 'l2']
           - 'newton-cg'       -   ['l2', None]
           - 'newton-cholesky' -   ['l2', None]
           - 'sag'             -   ['l2', None]
           - 'saga'            -   ['elasticnet', 'l1', 'l2', None]

        .. note::
           'sag' and 'saga' fast convergence is only guaranteed on features
           with approximately the same scale. You can preprocess the data with
           a scaler from :mod:`sklearn.preprocessing`.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

    verbose : int, default=0
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        Useless for liblinear solver.

    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". This parameter is ignored when the ``solver`` is
        set to 'liblinear' regardless of whether 'multi_class' is specified or
        not. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    l1_ratio : float, default=None
        The Elastic-Net mixing parameter, with ``0 <= l1_ratio <= 1``. Only
        used if ``penalty='elasticnet'``. Setting ``l1_ratio=0`` is equivalent
        to using ``penalty='l2'``, while setting ``l1_ratio=1`` is equivalent
        to using ``penalty='l1'``. For ``0 < l1_ratio <1``, the penalty is a
        combination of L1 and L2.

    norm_order : non-zero int, inf, -inf, default=1
        Order of the norm used to normalize the coef to calculate the weights in the case
        where the ``coef_`` attribute of the estimator is of dimension 2 (multiclass case).
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
        self.coef_ = None
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

    def fit(self, X, y, **kwargs):
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
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
