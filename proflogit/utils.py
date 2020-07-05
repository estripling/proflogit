import os
import numbers
import numpy as np

INTEGER_TYPES = (numbers.Integral, np.int, np.integer)
FLOAT_TYPES = (float, np.float)
NUM_TYPE = (INTEGER_TYPES, FLOAT_TYPES)


def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    # Code slightly adjusted from scikit-learn utils/validation.py

    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, INTEGER_TYPES):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        '"{}" cannot be used to seed a numpy.random.RandomState instance'.format(
            seed
        )
    )


def load_data(file_name, target_variable=None, has_header=True, **kwargs):
    """
    Load data for easy use with ProfLogit.

    Parameters
    ----------
    file_name : str
        Name of data file to load.

    target_variable : None or int or str (default: None)
        Column index or name (if file has header) of the target variable.
        If None, all columns are in ``X`` and ``y`` is None.
        If not None, return ``X``, ``y``, where ``y`` is an array-like object.
        *Note*: Zero-based indexing for int.

    has_header : bool (default: True)
        If True, first row is used as header.

    **kwargs are passed to `numpy.recfromtxt`.

    Returns
    -------
    X, y: tuple
        ``X`` is a dict containing the predictor variables.
        ``y`` is None or an array-like object containing the target variable.
    """
    assert isinstance(file_name, str)
    assert os.path.isfile(file_name), "'{}' not found".format(file_name)
    assert target_variable is None or isinstance(
        target_variable, (str, INTEGER_TYPES)
    )
    assert isinstance(has_header, bool)

    # Load data as numpy.recarray
    if has_header:
        data = np.recfromtxt(file_name, names=True, encoding=None, **kwargs)
    else:
        data = np.recfromtxt(file_name, encoding=None, **kwargs)

    if target_variable is None:
        X = {k: data[k] for k in data.dtype.names}
        y = None
    elif isinstance(target_variable, str):
        assert target_variable in data.dtype.names
        X = {k: data[k] for k in data.dtype.names if k != target_variable}
        y = data[target_variable]
    else:
        n_cols = len(data.dtype)
        assert 0 <= target_variable <= n_cols
        target_col = [
            nm for i, nm in enumerate(data.dtype.names) if i == target_variable
        ][0]
        X = {k: data[k] for k in data.dtype.names if k != target_col}
        y = data[target_col]

    # If data type is not numeric: byte_string --> unicode_string
    check_dtypes = [k for k, v in X.items() if v.dtype.kind not in "iufc"]
    if check_dtypes:
        for k in check_dtypes:
            X[k] = [bs for bs in X[k]]

    return X, y


def default_formula(data):
    """
    Build ProfLogit's default formula.

    Parameters
    ----------
    data : dict-like object
        Object to look up variables referenced in ``formula_like``.
        See `patsy.dmatrix`.

    """
    colnm = [k for k in data]
    is_numeric = [
        (hasattr(data[k], "dtype") and data[k].dtype.kind in "iufc")
        or (
            isinstance(data[k], (list, tuple, set))
            and isinstance(data[k][0], NUM_TYPE)
        )
        for k in colnm
    ]
    form = " + ".join(
        [
            "standardize({})".format(cnm) if is_num else "{}".format(cnm)
            for cnm, is_num in zip(colnm, is_numeric)
        ]
    )
    return form
