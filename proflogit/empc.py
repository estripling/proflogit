import warnings
import numpy as np
from sklearn.metrics import roc_curve
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from scipy.special import beta as beta_func
from scipy.special import betainc
from .utils import NUM_TYPE


class EMPChurn(object):
    """
    Expected Maximum Profit Measure for Customer Churn (EMPC)
    =========================================================

    Parameters
    ----------
    y_true : 1D array-like or label indicator array, shape=(n_samples,)
        Binary target values. Churners have a ``cases_label`` value.

    y_score : 1D array-like, shape=(n_samples,)
        Target scores, can either be probability estimates or
        non-thresholded decision values.

    alpha : float (default: 6)
        Alpha parameter of uni-modal beta distribution (alpha > 1).

    beta : float (default: 14)
        Beta parameter of uni-modal beta distribution (beta > 1).

    clv : float (default: 200)
        Constant customer lifetime value per retained customer (clv > d).

    d : float (default: 10)
        Constant cost of retention offer (d > 0).

    f : float (default: 1)
        Constant cost of contact (f > 0).

    case_label : str or int or float (default: 1)
        Label for the identification of cases (i.e., churners).

    Note
    ----
    The EMPC requires that the churn class is encoded as 0, and it is NOT
    interchangeable (see [3, p. 37]). However, this implementation assumes
    the standard notation ('churn': 1, 'no churn': 0) and makes the necessary
    changes internally. An equivalent R implementation is available in [2].

    References
    ----------
    [1] Verbraken, T., Verbeke, W. and Baesens, B. (2013).
        A Novel Profit Maximizing Metric for Measuring Classification
        Performance of Customer Churn Prediction Models. IEEE Transactions on
        Knowledge and Data Engineering, 25(5), 961-973. Available Online:
        http://ieeexplore.ieee.org/iel5/69/6486492/06165289.pdf?arnumber=6165289
    [2] Bravo, C. and Vanden Broucke, S. and Verbraken, T. (2015).
        EMP: Expected Maximum Profit Classification Performance Measure.
        R package version 2.0.1. Available Online:
        http://cran.r-project.org/web/packages/EMP/index.html
    [3] Verbraken, T. (2013). Business-Oriented Data Analytics:
        Theory and Case Studies. Ph.D. dissertation, Dept. LIRIS, KU Leuven,
        Leuven, Belgium, 2013.

    """

    def __init__(
        self,
        y_true,
        y_score=None,
        alpha=6,
        beta=14,
        clv=200,
        d=10,
        f=1,
        case_label=1,
    ):
        """Constructor for EMPChurn"""
        assert isinstance(alpha, NUM_TYPE) and alpha > 1
        assert isinstance(beta, NUM_TYPE) and beta > 1
        assert isinstance(clv, NUM_TYPE) and clv > 0
        assert isinstance(d, NUM_TYPE) and d > 0
        assert isinstance(f, NUM_TYPE) and f > 0
        assert clv > d
        assert isinstance(case_label, (str, NUM_TYPE))

        self._yt = None
        self._ys = None
        self._gamma_needs_update = None
        y = np.asarray(y_true)
        self.y_true = y == case_label  # Make it binary
        self.n_samples = len(self.y_true)
        if y_score is not None:
            self.y_score = np.asarray(y_score)
            assert len(self.y_true) == len(self.y_score)
        self.alpha = alpha
        self.beta = beta
        self.clv = clv
        self.d = d
        self.f = f

        # Recall: cases are labeled as 0s in the EMP framework
        self.pi0 = np.sum(self.y_true) / self.n_samples
        self.pi1 = 1 - self.pi0

        self._delta = self.d / self.clv
        self._phi = self.f / self.clv

        self._egamma = alpha / (alpha + beta)
        self._gamma = None

        self._F0 = None
        self._F1 = None

        self._empc = None

    @property
    def y_true(self):
        """Getter: y_true"""
        return self._yt

    @y_true.setter
    def y_true(self, value):
        """Setter: y_true"""
        y_true = np.asarray(value)
        assert y_true.ndim == 1
        assert len(np.unique(y_true)) == 2
        self._yt = y_true

    @property
    def y_score(self):
        """Getter: y_score"""
        return self._ys

    @y_score.setter
    def y_score(self, value):
        """Setter: y_score"""
        if value is not None:
            y_score = np.asarray(value)
            assert y_score.ndim == 1
            self._gamma_needs_update = True
            self._ys = y_score

    def empc(self, y_score=None):
        """
        Compute EMPC.

        Parameters
        ----------
        y_score : 1D array-like, shape=(n_samples,)
            Target scores, can either be probability estimates or
            non-thresholded decision values. Must have the same length
            as y_true.

        Returns
        -------
        empc : float
            Empirical EMP estimate for customer churn prediction.

        """
        if y_score is not None:
            self.y_score = y_score
        assert self._gamma_needs_update is not None and self._ys is not None
        assert len(self._yt) == len(self._ys)

        # Check _empc
        if not self._gamma_needs_update and self._empc:
            # Gamma hasn't changed and empc has already been computed
            return self._empc
        else:
            # Gamma has been changed, empc needs to be recomputed
            assert self._gamma_needs_update
            self._empc = self._compute_empc()
            return self._empc

    def _compute_empc(self):
        self._compute_gamma()
        gammaii = self._gamma[:-1]
        gammaie = self._gamma[1:]
        F0 = self._F0[range(len(gammaii))]
        F1 = self._F1[range(len(gammaii))]

        def betafn(x, a, b):
            return betainc(a, b, x) * beta_func(a, b)

        contr0 = (
            (self.clv * (1 - self._delta) * self.pi0 * F0)
            * (
                betafn(gammaie, self.alpha + 1, self.beta)
                - betafn(gammaii, self.alpha + 1, self.beta)
            )
            / betafn(1, self.alpha, self.beta)
        )

        tmp = (
            betafn(gammaie, self.alpha, self.beta)
            - betafn(gammaii, self.alpha, self.beta)
        ) / betafn(1, self.alpha, self.beta)
        contr1 = (
            -self.clv
            * (
                self._phi * self.pi0 * F0
                + (self._delta + self._phi) * self.pi1 * F1
            )
        ) * tmp
        empc = np.sum(np.add(contr0, contr1))
        return empc

    def _compute_gamma(self):
        self._emp_roc_info()
        numerator = self.pi1 * (self._delta + self._phi) * np.diff(
            self._F1
        ) + self.pi0 * self._phi * np.diff(self._F0)
        denominator = self.pi0 * (1 - self._delta) * np.diff(self._F0)
        with warnings.catch_warnings():
            # Ignore RuntimeWarning: division by zero
            # It is taken care of in the next gamma line
            warnings.simplefilter("ignore")
            gamma = np.append([0], numerator / denominator)
        self._gamma = np.append(gamma[gamma < 1], [1])
        self._gamma_needs_update = False

    def _emp_roc_info(self):
        # Compute ROC Convex Hull
        def compute_roc_convex_hull(y_true, y_score):
            fpr, tpr, _ = roc_curve(
                y_true, y_score, pos_label=True, drop_intermediate=False,
            )
            if fpr[0] != 0 or tpr[0] != 0:
                fpr = np.concatenate([[0], fpr])
                tpr = np.concatenate([[0], tpr])

            # For testing
            fpr = (
                np.concatenate([fpr, [1]])
                if fpr[-1] != 1 or tpr[-1] != 1
                else fpr
            )
            tpr = (
                np.concatenate([tpr, [1]])
                if fpr[-1] != 1 or tpr[-1] != 1
                else tpr
            )
            is_finite = np.isfinite(fpr) & np.isfinite(tpr)
            fpr = fpr[is_finite]
            tpr = tpr[is_finite]
            assert fpr.shape[0] >= 2, "Too few distinct predictions for ROCCH"
            try:
                points = np.c_[fpr, tpr]
                ind = ConvexHull(points).vertices
                ch_fpr = fpr[ind]
                ch_tpr = tpr[ind]
                ind_upper_triangle = ch_fpr < ch_tpr
                ch_fpr = np.concatenate([[0], ch_fpr[ind_upper_triangle], [1]])
                ch_tpr = np.concatenate([[0], ch_tpr[ind_upper_triangle], [1]])
                ind = np.argsort(ch_fpr)
                ch_fpr = ch_fpr[ind]
                ch_tpr = ch_tpr[ind]
            except QhullError:
                ch_fpr = np.array([0, 1])
                ch_tpr = np.array([0, 1])
            return ch_fpr, ch_tpr

        G0, G1 = compute_roc_convex_hull(self._yt, self._ys)
        # Recall: cases are labeled as 0s in the EMP framework
        self._F0 = G1
        self._F1 = G0
