import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


class LowInfoFilter(SelectorMixin, BaseEstimator):
    """Feature selector that removes all low-information features.

    This feature selection algorithm looks only at the features (X), not the
    desired outputs (y), and can thus be used for unsupervised learning.

    A feature is considered to be a low info one if the proportion of nan
    values is above a given threshold set by the user.

    Parameters
    ----------
    max_nan_fraction : float, default=0.2
        Features with a proportion of nan values greater than this threshold will
        be removed. By default, the proportion is set to 0.2.

    Attributes
    ----------
    nan_counts_ : array-like, shape=(n_features_in_,)
        Count of nan values for each individual feature.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : array-like, shape=(n_features_in_,)
        Names of features seen during the fit. Defined only when X
        has feature names that are all strings.

    Notes
    -----
    Allows NaN in the input.
    Raises ValueError if no feature in X meets the low info threshold.
    """

    def __init__(self, max_nan_fraction=0.2):
        self.max_nan_fraction = max_nan_fraction
        self.n_samples = None
        self.nan_counts_ = None

    def fit(self, X, y=None):
        """Learn empirical Nan proportion in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute NaN proportion, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            force_all_finite="allow-nan",
        )

        if self.max_nan_fraction > 1 or self.max_nan_fraction < 0:
            raise ValueError(
                f"Nan fraction must be between 0 and 1 Got: {self.max_nan_fraction}")

        self.n_samples = X.shape[0]
        self.nan_counts_ = np.isnan(np.array(X)).sum(0)

        if np.all(~np.isfinite(self.nan_counts_) | (
                self.nan_counts_ > self.max_nan_fraction * self.n_samples)):
            msg = "No feature in X meets the low info threshold {0:.5f}"
            if self.n_samples == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.max_nan_fraction))

        return self

    def _get_support_mask(self):
        """Get a mask, or integer index, of the features selected
            
        Returns
        -------
        support : array-like
            An index that selects the retained features from a feature vector.
            This is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. 
        """
        check_is_fitted(self)

        return self.nan_counts_ <= self.max_nan_fraction * self.n_samples

    def _more_tags(self):
        # Useful to allow the use of nan values
        return {"allow_nan": True}
