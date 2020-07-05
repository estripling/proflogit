import patsy
import numpy as np
from sklearn.utils import check_array, check_X_y
from .rga import RGA
from .empc import EMPChurn
from .utils import default_formula
from .utils import FLOAT_TYPES, NUM_TYPE


class ProfLogitCCP(object):
    """
    ProfLogit for Customer Churn Prediction
    =======================================

    Maximizing empirical EMP for churn by optimizing
    the regression coefficients of the logistic model through
    a Real-coded Genetic Algorithm (RGA).

    Parameters
    ----------
    rga_kws : dict (default: None)
        Parameters passed to ``RGA``.
        If None, the following parameters are used:
        {
            'niter': 1000,
            'niter_diff': 250,
        }
        *Note*: When manually specifying ``rga_kws``, at least one of the
        following three parameters must to be in the dict:
        ``niter``, ``niter_diff``, or ``nfev``.
        See help(proflogit.rga.RGA) for more information.

    reg_kws : dict (default: None)
        Parameters for regularization.
        If None, the following parameters are used:
        {
            'lambda': 0.1,  # Arbitrary value, needs to be tuned
            'alpha': 1.0,  # By default, applying lasso penalty
            'soft-thd': True,  # Apply soft-thresholding
        }

    empc_kws : dict (default: None)
        Parameters passed to ``EMPChurn``.
        If None, the default parameters are used.

    intercept : bool (default: True)
        If True, intercept in the logistic model is taken into account.
        The corresponding all-ones vector should be in
        the first column of ``x``.

    default_bounds: tuple, default: (-3, 3)
        Bounds for every regression parameter. Use the ``bounds`` parameter
        through ``rga_kws`` for individual specifications.

    Attributes
    ----------
    rga : `proflogit.rga.RGA`
        RGA instance. The optimization result is stored under its ``res``
        attribute, which is represented as a `scipy.optimize.OptimizeResult`
        object with attributes:
          * ``x`` (ndarray) The solution of the optimization.
          * ``fun`` (float) Objective function value of the solution array.
          * ``success`` (bool) Whether the optimizer exited successfully.
          * ``message`` (str) Description of the cause of the termination.
          * ``nit`` (int) Number of iterations performed by the optimizer.
          * ``nit_diff`` (int) Number of consecutive non-improvements.
          * ``nfev`` (int) Number of evaluations of the objective functions.
        See `OptimizeResult` for a description of other attributes.

    References
    ----------
    [1] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2017). Profit Maximizing Logistic Model for
        Customer Churn Prediction Using Genetic Algorithms.
        Swarm and Evolutionary Computation.
    [2] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2015). Profit Maximizing Logistic Regression Modeling for
        Customer Churn Prediction. IEEE International Conference on
        Data Science and Advanced Analytics (DSAA) (pp. 1–10). Paris, France.

    """

    def __init__(
        self,
        rga_kws=None,
        reg_kws=None,
        empc_kws=None,
        intercept=True,
        default_bounds=(-3, 3),
    ):
        # Check regularization parameters
        default_reg_kws = {"lambda": 0.1, "alpha": 1.0, "soft-thd": True}
        if reg_kws is None:
            self.reg_kws = default_reg_kws
        else:
            assert isinstance(reg_kws, dict), (
                "``reg_kws`` must be a dict, "
                "where keys are parameters. See help(ProfLogitCCPDFO)."
            )
            for k, v in reg_kws.items():
                if k == "lambda":
                    assert isinstance(v, FLOAT_TYPES) and v >= 0.0
                elif k == "alpha":
                    assert isinstance(v, FLOAT_TYPES) and 0.0 <= v <= 1.0
                elif k == "soft-thd":
                    assert isinstance(v, bool)
                else:
                    raise KeyError(
                        "'{}' key is not available in reg_kws. ".format(k)
                        + "See help(ProfLogitCCPDFO)."
                    )
            default_reg_kws.update(reg_kws)
            self.reg_kws = default_reg_kws

        # Check EMPC parameters
        if empc_kws is None:
            empc_kws = {}
        assert isinstance(empc_kws, dict), (
            "``empc_kws`` must be a dict, "
            "where keys are parameters. See help(proflogit.empc.EMPChurn)."
        )
        self.empc_kws = empc_kws

        # Check RGA parameters
        if rga_kws is None:
            rga_kws = {"niter": 1000, "niter_diff": 250}
        assert isinstance(rga_kws, dict), (
            "``rga_kws`` must be a dict, "
            "where keys are parameters. See help(proflogit.rga.RGA)."
        )
        self.rga_kws = rga_kws

        # Check intercept
        assert isinstance(intercept, bool)
        self.intercept = intercept

        # Check bounds
        assert isinstance(default_bounds, tuple) and len(default_bounds) == 2
        assert all(isinstance(v, NUM_TYPE) for v in default_bounds)
        self.default_bounds = default_bounds

        # Attributes
        self.n_dim = None
        self.rga = None
        self.emp = None

    def fit(self, X, y):
        """
        Train ProfLogitCCP.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)
            Standardized design matrix (zero mean and unit variance).

        y : 1D array-like or label indicator array, shape=(n_samples,)
            Binary target values. Churners have a ``case_label`` value.

        Returns
        -------
        self : object
            Returns self.

        """
        X, y = check_X_y(X, y)
        self.n_dim = X.shape[1]

        if self.intercept:
            assert all(x == 1 for x in X[:, 0]), "First column must be all 1s."

        if "bounds" not in self.rga_kws:
            self.rga_kws["bounds"] = [self.default_bounds] * self.n_dim

        # Init
        self.emp = EMPChurn(y_true=y, **self.empc_kws)
        func_args = (X, self.reg_kws, self.emp, self.intercept)
        self.rga = RGA(func=proflogit_fobj, args=func_args, **self.rga_kws)

        # Do optimization
        self.rga.solve()

        return self

    def predict_proba(self, X):
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)
            Standardized design matrix (zero mean and unit variance).

        Returns
        -------
        y_score : 1D array-like, shape=(n_samples,)
            Predicted probabilities for churners.

        """
        X = check_array(X)
        assert X.ndim == 2
        assert X.shape[1] == self.n_dim
        theta = self.rga.res.x
        logits = np.dot(X, theta)
        y_score = 1 / (1 + np.exp(-logits))  # Invert logit transformation
        return y_score

    def score(self, X, y_true):
        """
        Compute EMPC performance score.

        Parameters
        ----------
        X : 2D numpy.ndarray, shape=(n_samples, n_dim)
            Standardized design matrix (zero mean and unit variance).

        y_true : 1D array-like or label indicator array, shape=(n_samples,)
            Binary target values. Churners have a ``case_label`` value.

        Returns
        -------
        score : float
            EMPC score.

        """
        X, y_true = check_X_y(X, y_true)
        y_score = self.predict_proba(X)
        empc = EMPChurn(y_true, y_score, **self.empc_kws).empc()
        return empc


def proflogit_fobj(theta, *args):
    """ProfLogit's objective function (maximization problem)."""
    # Get function arguments
    X, reg_kws, emp, intercept = args
    # X: (numpy.ndarray) standardized model matrix
    # reg_kws: (dict) regularization parameters
    # emp: (EMPChurn) object to compute EMPC
    # intercept: (bool) include intercept

    # Check theta
    # b refers to elements in theta; modifying b, will modify the corresponding
    # elements in theta
    # b is the vector holding the regression coefficients (no intercept)
    b = theta[1:] if intercept else theta

    def soft_thresholding_func(bvec, regkws):
        """Apply soft-thresholding."""
        bool_nonzero = (np.abs(bvec) - regkws["lambda"]) > 0
        if np.sum(bool_nonzero) > 0:
            bvec[bool_nonzero] = np.sign(bvec[bool_nonzero]) * (
                np.abs(bvec[bool_nonzero]) - regkws["lambda"]
            )
        if np.sum(~bool_nonzero) > 0:
            bvec[~bool_nonzero] = 0
        return bvec

    def reg_term(bvec, regkws):
        """Elastic net regularization."""
        return 0.5 * (1 - regkws["alpha"]) * np.sum(bvec ** 2) + regkws[
            "alpha"
        ] * np.sum(np.abs(bvec))

    if reg_kws["soft-thd"]:
        b = soft_thresholding_func(b, reg_kws)
    logits = np.dot(X, theta)
    y_score = 1 / (1 + np.exp(-logits))  # Invert logit transformation
    empc = emp.empc(y_score)
    penalty = reg_kws["lambda"] * reg_term(b, reg_kws)
    fitness = empc - penalty
    return fitness


class ProfLogit(ProfLogitCCP):
    """
    ProfLogit for churn
    ===================

    This is a convenient wrapper for ProfLogitCCP to make use of
    patsy's powerful ``formula`` functionality that allows building
    appropriate design matrices easily.

    Parameters
    ----------
    formula_like : str (default: None)
        Formula to build design matrix, which is passed to `patsy.dmatrix`.
        Hence, only the right-hand needs to be specified.
        If None, ``formula_like`` only contains main effects,
        as well as adds patsy's built-in standardization transformation
        function to all numeric variables (when ``fit`` method is called).
        To exclude the intercept term, specify it either in ``formula_like``
        or set the ``intercept`` argument to ``False``.

    *args, **kwargs are passed to class `proflogit.base.ProfLogitCCP`.
    For more information, see help(proflogit.base.ProfLogitCCP).

    Attributes
    ----------
    rga : `proflogit.rga.RGA`
        RGA instance. The optimization result is stored under its ``res``
        attribute, which is represented as a `scipy.optimize.OptimizeResult`
        object with attributes:
          * ``x`` (ndarray) The solution of the optimization.
          * ``fun`` (float) Objective function value of the solution array.
          * ``success`` (bool) Whether the optimizer exited successfully.
          * ``message`` (str) Description of the cause of the termination.
          * ``nit`` (int) Number of iterations performed by the optimizer.
          * ``nit_diff`` (int) Number of consecutive non-improvements.
          * ``nfev`` (int) Number of evaluations of the objective functions.
        See `OptimizeResult` for a description of other attributes.

    References
    ----------
    [1] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2017). Profit Maximizing Logistic Model for
        Customer Churn Prediction Using Genetic Algorithms.
        Swarm and Evolutionary Computation.
    [2] Stripling, E., vanden Broucke, S., Antonio, K., Baesens, B. and
        Snoeck, M. (2015). Profit Maximizing Logistic Regression Modeling for
        Customer Churn Prediction. IEEE International Conference on
        Data Science and Advanced Analytics (DSAA) (pp. 1–10). Paris, France.

    """

    def __init__(self, formula_like=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert formula_like is None or isinstance(formula_like, str)
        self.formula = formula_like
        self.design_info = None

    def fit(self, X, y):
        """
        Train ProfLogit.

        Parameters
        ----------
        X : dict-like object
            Object to look up variables referenced in ``formula_like``.
            See `patsy.dmatrix`.

        y : 1D array-like or label indicator array, shape=(n_samples,)
            Binary target values. Churners have a ``case_label`` value.

        Returns
        -------
        self : object
            Returns self.

        """
        if self.formula is None:
            self.formula = default_formula(X)
            if self.intercept is False:
                self.formula += " - 1"
        dmat = patsy.dmatrix(self.formula, X)
        self.design_info = dmat.design_info
        if "Intercept" not in self.design_info.column_names:
            self.intercept = False
        return super().fit(dmat, y)

    def predict_proba(self, X):
        """
        Compute predicted probabilities.

        Parameters
        ----------
        X : dict-like object
            Object to look up variables referenced in ``formula_like``.
            See `patsy.dmatrix`.

        Returns
        -------
        y_score : 1D array-like, shape=(n_samples,)
            Predicted probabilities for churners.

        """
        dmat_test = patsy.build_design_matrices([self.design_info], X)[0]
        y_score = super().predict_proba(dmat_test)
        return y_score

    def score(self, X, y_true):
        """
        Compute EMPC performance score.

        Parameters
        ----------
        X : dict-like object
            Object to look up variables referenced in ``formula_like``.
            See `patsy.dmatrix`.

        y_true : 1D array-like or label indicator array, shape=(n_samples,)
            Binary target values. Churners have a ``case_label`` value.

        Returns
        -------
        score : float
            EMPC score.

        """
        y_score = self.predict_proba(X)
        empc = EMPChurn(y_true, y_score, **self.empc_kws).empc()
        return empc
