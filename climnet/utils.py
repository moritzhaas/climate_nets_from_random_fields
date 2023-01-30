"""Util functions."""

import contextlib
import numpy as np
import pandas as pd
import xarray as xr

SEED = 42


@contextlib.contextmanager
def temp_seed(seed=SEED):
    """Set seed locally.
    Usage: 
    ------
    with temp_seed(42):
        np.random.randn(3)

    Parameters:
    ----------
    seed: int
        Seed of function, Default: SEED
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def holm(pvals, alpha=0.05, corr_type="bonf"):
    """
    Returns indices of p-values using Holm's method for multiple testing.

    Args:
    ----
    pvals: list
        list of p-values
    alpha: float
        TODO
    corr_type: str
        TODO
    """
    n = len(pvals)
    sortidx = np.argsort(pvals)
    p_ = pvals[sortidx]
    j = np.arange(1, n + 1)
    if corr_type == "bonf":
        corr_factor = alpha / (n - j + 1)
    elif corr_type == "dunn":
        corr_factor = 1. - (1. - alpha) ** (1. / (n - j + 1))
    try:
        idx = np.where(p_ <= corr_factor)[0][-1]
        lst_idx = sortidx[:idx]
    except IndexError:
        lst_idx = []
    return lst_idx


def get_nino34(fname, time_range=None, time_roll=3):
    df = pd.read_csv(
        fname, skiprows=1, header=0, delim_whitespace=True
    )
    time = []
    for i, row in df.iterrows():
       time.append(np.datetime64('{}-{:02d}'.format(int(row['YR']), int(row['MON'])), 'D'))

    da = xr.DataArray(data=df['ANOM.3'], name='nino3.4', coords={"time": np.array(time)},
                      dims=["time"])
    ts_mean = da.rolling(time=time_roll, center=True).mean()
    if time_range is not None:
        ts_mean = ts_mean.sel(time=slice(np.datetime64(time_range[0], "M"),
                                         np.datetime64(time_range[1], "M")) )
    return ts_mean
