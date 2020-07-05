import datetime
import numpy as np
from scipy.optimize import OptimizeResult
from .utils import check_random_state
from .utils import INTEGER_TYPES, FLOAT_TYPES


class RGA(object):
    """
    Real-coded Genetic Algorithm
    ============================

    Parameters
    ----------
    func : callable ``f(x, *args)``
        The objective function to be maximized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a tuple of any additional fixed parameters needed to
        completely specify the function.

    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
        For example, for a 2D problem with -10 <= x_i <= 10, i=1,2, specify:
        ``bounds=[(-10, 10)] * 2``.

    args : tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.

    popsize : None or int (default: None)
        If None, ``popsize`` is 10 * number of number of parameters.
        If int, ``popsize`` must be a positive integer >= 10.

    crossover_rate : float (default: 0.8)
        Perform local arithmetic crossover with probability ``crossover_rate``.

    mutation_rate : float (default: 0.1)
        Perform uniform random mutation with probability ``mutation_rate``.

    elitism : int or float (default: 0.05)
        Number of the fittest chromosomes to survive to the next generation.
        If float, ``elitism`` is ``int(max(1, round(popsize * elitism)))``.
        If int and larger than ``popsize``, an exception is raised.

    niter : int (default: np.inf)
        The maximum number of generations over which the entire population is
        evolved.
        If np.inf, ``nfev`` or ``niter_diff`` must be specified.
        If int, it must be larger than zero. In this case, the algorithm stops
        when ``niter`` is reached, or possibly earlier when ``niter_diff``
        or ``nfev`` are specified as well.

    niter_diff : int (default: np.inf)
        Stop the algorithm if the fitness (with ``ftol`` tolerance)
        between consecutive best-so-far solutions remains the same for
        ``niter_diff`` number of iterations.
        If np.inf, ``niter`` or ``nfev`` must be specified.
        If int, it must be larger than zero. In this case, the algorithm stops
        when ``niter_diff`` is reached, or possibly earlier when ``niter``
        or ``nfev`` are specified as well.

    nfev : int (default: np.inf)
        The maximum number of function evaluations over which the population is
        evolved.
        If np.inf, ``niter`` or ``niter_diff`` must be specified.
        If int, it must be larger than zero. In this case, the algorithm stops
        when ``nfev`` is reached, or possibly earlier when ``niter_diff`` or
        ``niter`` are specified as well.

    disp : bool (default: False)
        Set to True to print status messages.

    ftol : float (default: 1e-4)
        Absolute tolerance for convergence. See ``niter_diff``.

    random_state : None or int or `np.random.RandomState` (default: None)
        If None, a new `np.random.RandomState` is used;
        If int, a new `np.random.RandomState` with ``random_state`` as
        seed is used;
        If ``random_state`` is already a `np.random.RandomState` instance,
        that instance is used.

    Attributes
    ----------
    res : OptimizeResult
        The optimization result represented as a
        `scipy.optimize.OptimizeResult` object.
        Important attributes are:
          * ``x`` (ndarray) The solution of the optimization.
          * ``fun`` (float) Objective function value of the solution array.
          * ``success`` (bool) Whether the optimizer exited successfully.
          * ``message`` (str) Description of the cause of the termination.
          * ``nit`` (int) Number of iterations performed by the optimizer.
          * ``nit_diff`` (int) Number of consecutive non-improvements.
          * ``nfev`` (int) Number of evaluations of the objective functions.
        See `OptimizeResult` for a description of other attributes.

    fx_best : list
        Fitness values of the best solution per generation,
        including the zero generation (initialization).

    """

    def __init__(
        self,
        func,
        bounds,
        args=(),
        popsize=None,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism=0.05,
        niter=np.inf,
        niter_diff=np.inf,
        nfev=np.inf,
        disp=False,
        ftol=1e-4,
        random_state=None,
    ):
        self.name = "RGA"

        # Objective function to maximize
        self.func = func
        self.args = args

        # Check bounds
        bnd = list(bounds)
        assert all(
            isinstance(t, tuple)
            and len(t) == 2
            and isinstance(t[0], (INTEGER_TYPES, FLOAT_TYPES))
            and isinstance(t[1], (INTEGER_TYPES, FLOAT_TYPES))
            for t in bnd
        )
        ary_bnd = np.asarray(bnd, dtype=np.float64).T
        self.min_b = ary_bnd[0]
        self.max_b = ary_bnd[1]
        self.diff_b = np.fabs(self.max_b - self.min_b)
        self.n_dim = len(bnd)

        # Check population size
        if popsize is None:
            self.popsize = self.n_dim * 10
        else:
            assert isinstance(popsize, INTEGER_TYPES) and popsize >= 10
            self.popsize = popsize

        # Check crossover rate
        assert 0.0 <= crossover_rate <= 1.0
        self.crossover_rate = crossover_rate

        # Check mutation rate
        assert 0.0 <= mutation_rate <= 1.0
        self.mutation_rate = mutation_rate

        # Check elitism parameter
        assert isinstance(elitism, (INTEGER_TYPES, FLOAT_TYPES))
        if isinstance(elitism, INTEGER_TYPES):
            assert 0 <= elitism <= self.popsize
            self.elitism = int(elitism)
        else:
            assert 0.0 <= elitism <= 1.0
            self.elitism = int(max(1, round(self.popsize * elitism)))

        # Check niter, niter_diff, and nfev
        assert (
            np.isfinite(niter) or np.isfinite(niter_diff) or np.isfinite(nfev)
        )
        if np.isfinite(niter):
            assert isinstance(niter, INTEGER_TYPES) and niter > 0
        self.niter = niter

        if np.isfinite(niter_diff):
            assert isinstance(niter_diff, INTEGER_TYPES) and niter_diff > 0
        self.niter_diff = niter_diff

        if np.isfinite(nfev):
            assert isinstance(nfev, INTEGER_TYPES) and nfev > 0
        self.nfev = nfev

        # Check disp
        assert isinstance(disp, bool)
        self.disp = disp

        # Check ftol
        assert isinstance(ftol, FLOAT_TYPES) and ftol >= 0.0
        self.ftol = ftol

        # Get random state object
        self.rng = check_random_state(random_state)

        # Attributes
        self._nit_diff = 0
        self._nit = 0
        self._nfev = 0
        self._n_mating_pairs = int(self.popsize / 2)  # Constant for crossover
        self.pop = None
        self.elite_pool = None
        self.fitness = np.empty(self.popsize) * np.nan
        self.fx_best = []
        self.res = OptimizeResult(success=False)

    def init(self):
        rnd_pop = self.rng.rand(self.popsize, self.n_dim)
        return self.min_b + rnd_pop * self.diff_b

    def evaluate(self):
        for ix in range(self.popsize):
            fval = self.fitness[ix]
            if np.isnan(fval):
                x = self.pop[ix]
                fval_new = self.func(x, *self.args)
                self.fitness[ix] = fval_new
                self._nfev += 1
                if self._nfev >= self.nfev:
                    return True
        return False

    def select(self):
        """Perform linear scaling selection"""
        fvals = np.copy(self.fitness)
        fmin = np.min(fvals)
        favg = np.mean(fvals)
        fmax = np.max(fvals)
        if fmin < 0:
            fvals -= fmin
            fmin = np.min(fvals)
        if fmin > (2 * favg - fmax) / (2 - 1):  # c_factor = 2
            denominator = fmax - favg
            a = (2 - 1) * favg / denominator
            b = favg * (fmax - 2 * favg) / denominator
        else:
            denominator = favg - fmin
            a = favg / denominator
            b = -fmin * favg / denominator
        fscaled = np.abs(a * fvals + b)
        rel_fitval = fscaled / fscaled.sum()
        select_ix = self.rng.choice(
            self.popsize, size=self.popsize, replace=True, p=rel_fitval,
        )
        self.pop = self.pop[select_ix]
        self.fitness = self.fitness[select_ix]

    def crossover(self):
        """Perform local arithmetic crossover"""
        # Make iterator for pairs
        match_parents = (
            rnd_pair
            for rnd_pair in self.rng.choice(
                self.popsize, (self._n_mating_pairs, 2), replace=False,
            )
        )

        # Crossover parents
        for ix1, ix2 in match_parents:
            if self.rng.uniform() < self.crossover_rate:
                parent1 = self.pop[ix1]  # Pass-by-ref
                parent2 = self.pop[ix2]
                w = self.rng.uniform(size=self.n_dim)
                child1 = w * parent1 + (1 - w) * parent2
                child2 = w * parent2 + (1 - w) * parent1
                self.pop[ix1] = child1
                self.pop[ix2] = child2
                self.fitness[ix1] = np.nan
                self.fitness[ix2] = np.nan

    def mutate(self):
        """Perform uniform random mutation"""
        for ix in range(self.popsize):
            if self.rng.uniform() < self.mutation_rate:
                mutant = self.pop[ix]  # inplace
                rnd_gene = self.rng.choice(self.n_dim)
                rnd_val = self.rng.uniform(
                    low=self.min_b[rnd_gene], high=self.max_b[rnd_gene],
                )
                mutant[rnd_gene] = rnd_val
                self.fitness[ix] = np.nan

    def _get_sorted_non_nan_ix(self):
        """Get indices sorted according to non-nan fitness values"""
        non_nan_fx = (
            (ix, fx) for ix, fx in enumerate(self.fitness) if ~np.isnan(fx)
        )
        sorted_list = sorted(non_nan_fx, key=lambda t: t[1])
        return sorted_list

    def update(self):
        """
        Update population by replacing the worst solutions of the current
        with the ones from the elite pool.
        Then, update the elite pool.
        Also, check if there has been an improvement in
        the best-so-far solution.
        """
        if any(np.isnan(fx) for fx in self.fitness):
            sorted_fx = self._get_sorted_non_nan_ix()
            worst_ix = [t[0] for t in sorted_fx][: self.elitism]
        else:
            worst_ix = np.argsort(self.fitness)[: self.elitism]
        for i, ix in enumerate(worst_ix):
            elite, elite_fval = self.elite_pool[i]
            self.pop[ix] = elite
            self.fitness[ix] = elite_fval
        self.update_elite_pool()
        is_fdiff = self.fx_best[-1] > (self.fx_best[-2] + self.ftol)
        if is_fdiff:
            self._nit_diff = 0
        else:
            self._nit_diff += 1

    def update_elite_pool(self):
        if any(np.isnan(fx) for fx in self.fitness):
            sorted_fx = self._get_sorted_non_nan_ix()
            elite_ix = [t[0] for t in sorted_fx][-self.elitism :]
        else:
            elite_ix = np.argsort(self.fitness)[-self.elitism :]
        self.elite_pool = [
            (self.pop[ix].copy(), self.fitness[ix]) for ix in elite_ix
        ]
        # Append best solution
        self.fx_best.append(self.fitness[elite_ix[-1]])

    def _print_status_message(self):
        status_msg = "Iter = {:5d}; nfev = {:6d}; fx = {:.4f}".format(
            self._nit, self._nfev, self.fx_best[-1],
        )
        print(status_msg)

    def solve(self):
        self.pop = self.init()
        init_break = self.evaluate()
        self.update_elite_pool()

        if init_break:
            run_main_loop = False
            self.res.message = (
                "Maximum number of function evaluations has been reached "
                "during initialization."
            )
        else:
            run_main_loop = True

        # Do the optimization
        if self.disp:
            print(
                "# ---  {} ({})  --- #".format(
                    self.name,
                    datetime.datetime.now().strftime("%a %b %d %H:%M:%S"),
                )
            )

        while run_main_loop:
            if self.disp:
                self._print_status_message()
            self.select()  # parent selection
            self.crossover()
            self.mutate()
            break_loop = self.evaluate()
            self.update()  # survivor selection: overlapping-generation model
            self._nit += 1
            if break_loop:
                self.res.message = (
                    "Maximum number of function evaluations has been reached."
                )
                break
            if self._nit >= self.niter:
                self.res.message = (
                    "Maximum number of iterations has been reached."
                )
                break
            if self._nit_diff >= self.niter_diff:
                self.res.message = (
                    "Maximum number of consecutive non-improvements has been "
                    "reached."
                )
                break

        stop_time = datetime.datetime.now().strftime("%a %b %d %H:%M:%S")

        if any(np.isnan(fx) for fx in self.fitness):
            sorted_fx = self._get_sorted_non_nan_ix()
            best_ix = [t[0] for t in sorted_fx][-1]
        else:
            best_ix = np.argmax(self.fitness)
        self.res.x = np.copy(self.pop[best_ix])
        self.res.fun = self.fitness[best_ix]
        self.res.success = True
        self.res.nit = self._nit
        self.res.nit_diff = self._nit_diff
        self.res.nfev = self._nfev
        if self.disp:
            self._print_status_message()
            print(self.res)
            print("# ---  {} ({})  --- #".format(self.name, stop_time))
