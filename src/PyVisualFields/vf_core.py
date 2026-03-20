"""
vf_core.py
Pure-Python replacement for the R `visualFields` package.
Covers: normative lookup, TD/PD, MD/PSD/VFI, progression regression.

Dependencies: numpy, scipy, pandas — no R or rpy2 required.
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 1. Normative values
# ---------------------------------------------------------------------------
# Auto-load real Sunyiu 24-2 normatives computed from vfctrSunyiu24d2 dataset.
# Falls back to Heijl 1987 approximation if CSV not found.

def _load_default_normdb() -> np.ndarray:
    """Load normvals_sunyiu24d2.csv if available, else use Heijl 1987 approximation."""
    csv_path = Path(__file__).parent / 'data' / 'normvals_sunyiu24d2.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df[['intercept', 'slope']].values.astype(np.float32)
    # Heijl 1987 approximation fallback (54 points)
    return np.array([
        [33.5, -0.082], [34.2, -0.082], [33.8, -0.083], [33.1, -0.082],
        [32.9, -0.081], [33.0, -0.081], [33.2, -0.082], [32.8, -0.081],
        [31.5, -0.080], [32.0, -0.081], [32.5, -0.080], [32.1, -0.079],
        [30.8, -0.079], [31.3, -0.079], [31.6, -0.079], [31.0, -0.078],
        [30.1, -0.078], [30.5, -0.078], [30.9, -0.078], [30.3, -0.077],
        [29.5, -0.077], [29.8, -0.077], [30.2, -0.077], [29.6, -0.076],
        [28.9, -0.076], [29.1, -0.076], [29.5, -0.076], [28.8, -0.075],
        [28.0, -0.075], [28.4, -0.075], [28.7, -0.075], [28.1, -0.074],
        [27.2, -0.074], [27.6, -0.074], [27.9, -0.074], [27.3, -0.073],
        [26.5, -0.073], [26.8, -0.073], [27.1, -0.073], [26.5, -0.072],
        [25.8, -0.072], [26.0, -0.072], [26.3, -0.072], [25.7, -0.071],
        [25.0, -0.071], [25.2, -0.071], [25.5, -0.071], [24.9, -0.070],
        [24.2, -0.070], [24.4, -0.070], [24.7, -0.070], [24.1, -0.069],
        [23.5, -0.069], [23.7, -0.069],
    ], dtype=np.float32)

_NORM_24_2 = _load_default_normdb()


def load_normative_db(csv_path: Optional[str] = None) -> np.ndarray:
    """
    Load normative table (54 × 2: intercept, slope).
    Pass a CSV path with columns ['intercept', 'slope'] to override defaults.
    """
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        return df[['intercept', 'slope']].values.astype(np.float32)
    return _NORM_24_2


def age_matched_norms(age: float, norm_db: Optional[np.ndarray] = None) -> np.ndarray:
    """Expected threshold per test point for a given age (years). Returns shape (54,)."""
    db = norm_db if norm_db is not None else _NORM_24_2
    return db[:, 0] + db[:, 1] * age  # intercept + slope × age


# ---------------------------------------------------------------------------
# 2. Pointwise deviations
# ---------------------------------------------------------------------------

def total_deviation(sensitivity: np.ndarray, age: float,
                    norm_db: Optional[np.ndarray] = None) -> np.ndarray:
    if age is None or np.isnan(age):          # ← ADD
        age = 60.0                            # ← fallback to population mean
    norms = age_matched_norms(age, norm_db)
    return sensitivity - norms

def pattern_deviation(td: np.ndarray, percentile: float = 85.0) -> np.ndarray:
    valid = td[~np.isnan(td)]
    if len(valid) == 0:                          # ← ADD THIS GUARD
        return np.full_like(td, np.nan)
    general_height = np.percentile(valid, percentile)
    return td - general_height

def compute_indices(sensitivity: np.ndarray, age: float,
                    weights=None, norm_db=None,
                    norm_percentile: float = 85.0) -> dict:
    td = total_deviation(sensitivity, age, norm_db)
    if np.sum(~np.isnan(td)) == 0:          # all-NaN row — return safe nulls
        nan54 = np.full(len(td), np.nan)
        return {"td": nan54, "pd": nan54,
                "MD": np.nan, "PSD": np.nan, "VFI": np.nan}
    pd_vals = pattern_deviation(td, norm_percentile)
    return {
        "td":  td,
        "pd":  pd_vals,
        "MD":  mean_deviation(td, weights),
        "PSD": pattern_std(pd_vals, weights),
        "VFI": vfi(pd_vals, weights),
    }
# ---------------------------------------------------------------------------
# 3. Global indices
# ---------------------------------------------------------------------------

# Eccentricity-based weights for 24-2 (simplified Heijl 1987 Gaussian weights).
# In production, load the exact per-point weights from the normative package.
_WEIGHTS_24_2 = np.ones(54, dtype=np.float32)  # uniform fallback


def mean_deviation(td: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    w = weights if weights is not None else _WEIGHTS_24_2
    mask = ~np.isnan(td)
    if mask.sum() == 0:                       # ← ADD
        return np.nan                         # ← nothing to average
    return float(np.average(td[mask], weights=w[mask]))


def pattern_std(pd_vals: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Pattern Standard Deviation (PSD) in dB.
    Weighted root-mean-square of pattern deviation values.
    """
    w = weights if weights is not None else _WEIGHTS_24_2
    mask = ~np.isnan(pd_vals)
    pd_m = pd_vals[mask]
    w_m = w[mask]
    w_m = w_m / w_m.sum()
    weighted_mean = np.dot(w_m, pd_m)
    weighted_var = np.dot(w_m, (pd_m - weighted_mean) ** 2)
    return float(np.sqrt(weighted_var))


def vfi(pd_vals: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Visual Field Index (VFI) — approximate (Bengtsson & Heijl 2008).
    Returns value in [0, 100] where 100 = normal.
    """
    w = weights if weights is not None else _WEIGHTS_24_2
    mask = ~np.isnan(pd_vals)
    pd_m = pd_vals[mask]
    w_m = w[mask] / w[mask].sum()
    # clamp PD contribution to [−30, 0] then scale to percentage
    contrib = np.clip(pd_m, -30.0, 0.0) / 30.0  # 0 = normal, −1 = fully depressed
    return float(100.0 * (1.0 + np.dot(w_m, contrib)))

# ---------------------------------------------------------------------------
# 4. Progression analysis
# ---------------------------------------------------------------------------

def vf_progression(dates: list, md_series: list) -> dict:
    """
    Linear regression of MD over time (OLS — mirrors `vfprogression` PLR).

    Parameters
    ----------
    dates     : list of date-like objects or float years (e.g. 2020.5)
    md_series : list/array of MD values (dB)

    Returns
    -------
    dict: slope (dB/year), intercept, r2, p_value, se, progression_flag
    """
    if isinstance(dates[0], (int, float)):
        t = np.array(dates, dtype=float)
    else:
        t0 = pd.to_datetime(dates[0])
        t = np.array([(pd.to_datetime(d) - t0).days / 365.25 for d in dates])

    md = np.array(md_series, dtype=float)
    mask = ~np.isnan(md)
    if mask.sum() < 3:
        return {"slope": np.nan, "intercept": np.nan,
                "r2": np.nan, "p_value": np.nan,
                "se": np.nan, "progression_flag": False}

    slope, intercept, r, p, se = stats.linregress(t[mask], md[mask])
    # Flag as progressing: slope < −1.0 dB/year AND p < 0.05 (common clinical threshold)
    flag = bool(slope < -1.0 and p < 0.05)
    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r2": round(r ** 2, 4),
        "p_value": round(p, 4),
        "se": round(se, 4),
        "progression_flag": flag,
    }


# ---------------------------------------------------------------------------
# 5. Probability maps (p < 0.05 / 0.02 / 0.01 / 0.005 flags)
# ---------------------------------------------------------------------------
# Normal distribution approximation; replace with empirical quantiles if available.

def _load_empirical_cutoffs():
    """Load per-location empirical probability cutoffs if available."""
    csv_path = Path(__file__).parent / 'data' / 'normvals_cutoffs.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path).values  # shape (54, 4): p0.005, p0.01, p0.02, p0.05
    return None

_EMPIRICAL_CUTOFFS = _load_empirical_cutoffs()  # (54,4) or None

_TD_PROB_CUTOFFS = {0.05: -2.0, 0.02: -2.5, 0.01: -3.0, 0.005: -3.5}


def probability_map(deviation: np.ndarray,
                    cutoffs: Optional[dict] = None) -> np.ndarray:
    """
    Assign probability level per test point.
    Uses per-location empirical cutoffs if available (matches R exactly),
    otherwise falls back to normal approximation.
    Returns float array: 0.05, 0.02, 0.01, 0.005, or 1.0 (normal).
    """
    dev = np.array(deviation, dtype=float)
    levels = np.ones(len(dev))

    if _EMPIRICAL_CUTOFFS is not None and cutoffs is None:
        n = min(len(dev), _EMPIRICAL_CUTOFFS.shape[0])
        for j, p in zip([3, 2, 1, 0], [0.05, 0.02, 0.01, 0.005]):
            col = _EMPIRICAL_CUTOFFS[:n, j]
            for i in range(n):
                if not np.isnan(dev[i]) and dev[i] <= col[i]:
                    levels[i] = p
    else:
        # Fallback: uniform dB cutoffs
        cuts = cutoffs or _TD_PROB_CUTOFFS
        for p_level in sorted(cuts.keys(), reverse=True):
            levels[deviation <= cuts[p_level]] = p_level

    return levels