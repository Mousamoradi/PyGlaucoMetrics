
"""
vfprogression.py — pure Python replacement.
AGIS and CIGTS scoring ported directly from the original R source.
All rpy2/R calls removed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyGlaucoMetrics import vf_core

# ---------------------------------------------------------------------------
# Sample datasets — load from bundled CSVs (same pattern as visualFields.py)
# ---------------------------------------------------------------------------

import os
import requests, tarfile, io

def _load_bundled(name):
    """Try local CSV first, then fall back to downloading from CRAN tarballs."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # Try dot and underscore variants
    for fname in [name, name.replace('.', '_')]:
        path = os.path.join(pkg_dir, 'data', f'{fname}.csv')
        if os.path.exists(path):
            return pd.read_csv(path)

    # Fall back: download from CRAN and cache locally
    return _download_rda(name)


def _download_rda(name):
    """Download .rda from CRAN tarball, convert to DataFrame, cache as CSV."""
    try:
        import rdata
    except ImportError:
        raise ImportError("pip install rdata  — needed to download vfprogression datasets")

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir  = os.path.join(pkg_dir, 'data')
    os.makedirs(out_dir, exist_ok=True)

    # Try current version then archived versions
    urls = [
        "https://cran.r-project.org/src/contrib/vfprogression_0.0.2.tar.gz",
        "https://cran.r-project.org/src/contrib/Archive/vfprogression/vfprogression_0.0.1.tar.gz",
        "https://cran.r-project.org/src/contrib/Archive/vfprogression/vfprogression_0.0.2.tar.gz",
    ]

    tar = None
    for url in urls:
        try:
            r = requests.get(url, timeout=30)
            content = r.content
            if content[:2] == b'\x1f\x8b':  # valid gzip magic bytes
                tar = tarfile.open(fileobj=io.BytesIO(content), mode="r:gz")
                break
        except Exception:
            continue

    if tar is None:
        raise RuntimeError(
            "Could not download vfprogression from CRAN. "
            "Please manually place the CSV files in src/PyVisualFields/data/.\n"
            "Alternatively run in R: library(vfprogression); "
            "write.csv(vfseries, 'vfseries.csv')"
        )

    # Extract and cache all datasets while we have the tarball open
    dataset_names = ["vfseries", "vf.vfi", "vf.cigts", "vf.plr.nouri.2012", "vf.schell2014"]
    result = None
    for ds in dataset_names:
        for member_path in [f"vfprogression/data/{ds}.rda",
                            f"vfprogression/data/{ds.replace('.','_')}.rda"]:
            try:
                f = tar.extractfile(member_path)
                if f is None:
                    continue
                raw = f.read()
                parsed = rdata.parser.parse_data(raw)
                converted = rdata.conversion.convert(parsed)
                key = ds if ds in converted else list(converted.keys())[0]
                df = converted[key]
                # Cache with both naming variants
                df.to_csv(os.path.join(out_dir, f"{ds}.csv"), index=False)
                df.to_csv(os.path.join(out_dir, f"{ds.replace('.','_')}.csv"), index=False)
                if ds == name or ds.replace('.','_') == name:
                    result = df
                break
            except Exception:
                continue

    if result is None:
        raise FileNotFoundError(f"Dataset '{name}' not found in vfprogression tarball.")
    return result


def data_vfi():           return _load_bundled('vf.vfi')
def data_vfseries():      return _load_bundled('vfseries')
def data_cigts():         return _load_bundled('vf.cigts')
def data_plrnouri2012():  return _load_bundled('vf.plr.nouri.2012')
def data_schell2014():    return _load_bundled('vf.schell2014')


# ---------------------------------------------------------------------------
# AGIS scoring — ported from Gaasterland et al. 1994 / Katz 1999
# ---------------------------------------------------------------------------

# Sector assignment for 54 test points (1-indexed → 0-indexed below)
_AGIS_SECTORS = (
    ['upper']*10 +
    ['nasal'] + ['upper']*7 +
    ['nasal', 'nasal'] + ['upper']*5 + [None] + ['upper'] +
    ['nasal', 'nasal'] + ['lower']*5 + [None] + ['lower'] +
    ['nasal'] + ['lower']*7 +
    ['lower']*10
)  # len=54, 0-indexed

# Neighbors (1-indexed in original → convert to 0-indexed)
_AGIS_NEIGHBORS_1 = [
    [2,5,6,7],[1,3,6,7,8],[2,4,7,8,9],[3,8,9,10],
    [1,6,12,13],[1,2,5,7,12,13,14],[1,2,3,6,8,13,14,15],[2,3,4,7,9,14,15,16],
    [3,4,8,10,15,16,17],[4,9,16,17,18],
    [19,20],[5,6,13,21,22],[5,6,7,12,14,21,22,23],[6,7,8,13,15,22,23,24],
    [7,8,9,14,16,23,24,25],[8,9,10,15,17,24,25],[9,10,16,18,25,27],[10,17,27],
    [11,20,28,29],[11,19,28,29],[12,13,22],[12,13,14,21,23],[13,14,15,22,24],
    [14,15,16,23,25],[15,16,17,24],[None],[17,18],
    [19,20,29,37],[19,20,28,37],[31,38,39],[30,32,38,39,40],[31,33,39,40,41],
    [32,34,40,41,42],[33,41,42,43],[None],[43,44],
    [28,29],[30,31,39,45,46],[30,31,32,38,40,45,46,47],[31,32,33,39,41,46,47,48],
    [32,33,34,40,42,47,48,49],[33,34,41,43,48,49,50],[34,36,42,44,49,50],[36,43,50],
    [38,39,46,51],[38,39,40,45,47,51,52],[39,40,41,46,48,51,52,53],
    [40,41,42,47,49,52,53,54],[41,42,43,48,50,53,54],[42,43,44,49,54],
    [45,46,47,52],[46,47,48,51,53],[47,48,49,52,54],[48,49,50,53],
]
# Convert to 0-indexed, remove None entries
_AGIS_NEIGHBORS = [
    [x-1 for x in nb if x is not None] if nb is not None else []
    for nb in _AGIS_NEIGHBORS_1
]

_AGIS_CRITERIA = -(np.array(
    [9,9,9,9,
     8,8,8,8,8,8,8,8, 6,6,6,6, 8,8,
     9,8, 6,6,6,6,6, np.nan, 8,
     9,7, 5,5,5,5,5, np.nan, 7,
     7,7, 5,5,5,5, 7,7,7,7,7,7,7,7,
     7,7,7,7,7,7], dtype=float))  # shape (54,)


def _agis_is_abnormal(vf54):
    """vf54: array of 54 TD values (NaN at blind spots). Returns bool array."""
    abnormal = np.zeros(54, dtype=bool)
    for i in range(54):
        if not np.isnan(vf54[i]) and not np.isnan(_AGIS_CRITERIA[i]):
            abnormal[i] = vf54[i] <= _AGIS_CRITERIA[i]
    return abnormal


def _agis_clusterize(abnormal_indices, neighbors):
    """Group abnormal indices into connected clusters via neighbor list."""
    clusters = []
    remaining = list(abnormal_indices)
    while remaining:
        if not clusters:
            clusters.append([remaining.pop(0)])
        else:
            last_cluster = clusters[-1]
            nb_set = set()
            for idx in last_cluster:
                nb_set.update(neighbors[idx])
            new_members = [x for x in remaining if x in nb_set]
            if not new_members:
                clusters.append([remaining.pop(0)])
            else:
                last_cluster.extend(new_members)
                last_cluster.sort()
                remaining = [x for x in remaining if x not in new_members]
    return clusters


def _agis_clusters(vf54):
    abnormal = _agis_is_abnormal(vf54)
    upper_idx = [i for i, s in enumerate(_AGIS_SECTORS) if s == 'upper' and abnormal[i]]
    lower_idx = [i for i, s in enumerate(_AGIS_SECTORS) if s == 'lower' and abnormal[i]]
    nasal_idx = [i for i, s in enumerate(_AGIS_SECTORS) if s == 'nasal' and abnormal[i]]
    return {
        'upper': _agis_clusterize(upper_idx, _AGIS_NEIGHBORS),
        'lower': _agis_clusterize(lower_idx, _AGIS_NEIGHBORS),
        'nasal': _agis_clusterize(nasal_idx, _AGIS_NEIGHBORS),
    }


def get_score_AGIS(df_VF_py):
    """
    Compute AGIS score for one VF exam.
    Input: DataFrame row with td1..td52 (or td1..td54) columns.
    Returns: int score in [0, 20].
    """
    if isinstance(df_VF_py, pd.Series):
        row = df_VF_py
    else:
        row = df_VF_py.iloc[0]

    # Extract TD columns
    td_cols = sorted([c for c in row.index if c.startswith('td') and c[2:].isdigit()],
                     key=lambda x: int(x[2:]))
    tds = np.array([row[c] for c in td_cols], dtype=float)
    n = len(tds)
    if n < 52:
        raise ValueError(f"AGIS requires ≥52 TD values, got {n}")

    # Expand 52 → 54 by inserting NaN at blind spot positions (indices 25 and 34, 1-indexed)
    if n == 52:
        vf = np.concatenate([tds[:25], [np.nan], tds[25:33], [np.nan], tds[33:]])
    else:
        vf = tds[:54].copy()

    cl = _agis_clusters(vf)
    score = 0

    # Nasal sector scoring
    if cl['nasal']:
        nasal_indices_all = [i for i, s in enumerate(_AGIS_SECTORS) if s == 'nasal']
        if len(cl['nasal']) == 1 and len(cl['nasal'][0]) < 3:
            # Nasal step: restricted to one hemifield
            upper_nasal = {10, 18, 19}   # 0-indexed: 11,19,20
            lower_nasal = {27, 28, 36}   # 0-indexed: 28,29,37
            pts = set(cl['nasal'][0])
            if pts.issubset(upper_nasal) or pts.issubset(lower_nasal):
                score += 1
        else:
            if any(len(c) > 2 for c in cl['nasal']):
                score += 1
        nasal_vals = vf[[i for i in nasal_indices_all if not np.isnan(vf[i])]]
        if np.sum(nasal_vals <= -12) >= 4:
            score += 1

    # Hemifield scoring
    def _score_hemifield(clusters):
        s = 0
        large = [c for c in clusters if len(c) >= 3]
        if not large:
            return 0
        total = sum(len(c) for c in large)
        if total >= 3:  s += 1
        if total >= 6:  s += 1
        if total >= 13: s += 1
        if total >= 20: s += 1
        locs = vf[np.concatenate(large)]
        locs = locs[~np.isnan(locs)]
        half = len(locs) / 2
        for thresh in [12, 16, 20, 24, 28]:
            if np.sum(locs <= -thresh) >= half:
                s += 1
        return s

    score += _score_hemifield(cl['upper'])
    score += _score_hemifield(cl['lower'])
    return score


# ---------------------------------------------------------------------------
# CIGTS scoring — ported from Gillespie et al. 2003
# ---------------------------------------------------------------------------

# Neighbors for 52-point CIGTS (1-indexed → 0-indexed below)
_CIGTS_NEIGHBORS_1 = [
    [2,5,6,7],[1,3,6,7,8],[2,4,7,8,9],[3,8,9,10],
    [1,6,11,12,13],[1,2,5,7,12,13,14],[1,2,3,6,8,13,14,15],[2,3,4,7,9,14,15,16],
    [3,4,8,10,15,16,17],[4,9,16,17,18],
    [5,12,19,20,21],[5,6,11,13,20,21,22],[5,6,7,12,14,21,22,23],
    [6,7,8,13,15,22,23,24],[7,8,9,14,16,23,24,25],[8,9,10,15,17,24,25],
    [9,10,16,18,25,26],[10,17,26],
    [11,20],[11,12,19,21],[11,12,13,20,22],[12,13,14,21,23],
    [13,14,15,22,24],[14,15,16,23,25],[15,16,17,24],[17,18],
    [28,35],[27,29,35,36],[28,30,35,36,37],[29,31,36,37,38],
    [30,32,37,38,39],[31,33,38,39,40],[32,39,40,41],[41,42],
    [27,28,29,36,43],[28,29,30,35,37,43,44],[29,30,31,36,38,43,44,45],
    [30,31,32,37,39,44,45,46],[31,32,33,38,40,45,46,47],
    [32,33,34,39,41,46,47,48],[33,34,40,42,47,48],[34,41,48],
    [35,36,37,44,49],[36,37,38,43,45,49,50],[37,38,39,44,46,49,50,51],
    [38,39,40,45,47,50,51,52],[39,40,41,46,48,51,52],[40,41,42,47,52],
    [43,44,45,50],[44,45,46,49,51],[45,46,47,50,52],[46,47,48,51],
]
_CIGTS_NEIGHBORS = [[x-1 for x in nb] for nb in _CIGTS_NEIGHBORS_1]  # 0-indexed


def get_score_CIGTS(df_VF_py):
    """
    Compute CIGTS score for one VF exam.
    Input: DataFrame row with tdp1..tdp52 (or tdp1..tdp54) columns.
    Returns: float score (0 = normal, higher = worse).
    """
    if isinstance(df_VF_py, pd.Series):
        row = df_VF_py
    else:
        row = df_VF_py.iloc[0]

    tdp_cols = sorted([c for c in row.index if c.startswith('tdp') and c[3:].isdigit()],
                      key=lambda x: int(x[3:]))
    tdprobs = np.array([row[c] for c in tdp_cols], dtype=float)
    n = len(tdprobs)
    if n < 52:
        raise ValueError(f"CIGTS requires ≥52 TD probability values, got {n}")
    if n == 54:
        tdprobs = np.delete(tdprobs, [25, 34])  # remove blind spots (0-indexed)
    tdprobs = tdprobs[:52]

    # Weights: p=0.005→4, p=0.01→3, p=0.02→2, p=0.05→1, else→0
    def _w(p):
        if p <= 0.005: return 4
        if p <= 0.01:  return 3
        if p <= 0.02:  return 2
        if p <= 0.05:  return 1
        return 0

    pw = np.array([_w(p) for p in tdprobs])

    def _effective_weight(i):
        nb_weights = sorted([pw[j] for j in _CIGTS_NEIGHBORS[i]], reverse=True)
        second = nb_weights[1] if len(nb_weights) >= 2 else 0
        return min(second, pw[i])

    eff_weights = np.array([_effective_weight(i) for i in range(52)])
    return float(np.sum(eff_weights) / 10.4)


# ---------------------------------------------------------------------------
# Progression methods
# ---------------------------------------------------------------------------

def _get_years(df):
    """
    Return years array. Uses 'yearsfollowed' column if present (all R datasets
    provide this), otherwise falls back to date parsing or integer index.
    """
    if 'yearsfollowed' in df.columns:
        return df['yearsfollowed'].values.astype(float)
    if 'date' in df.columns:
        raw = df['date'].reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(raw):
            dates = raw
        else:
            dates = pd.to_datetime(raw, errors='coerce')
            if dates.isna().all():
                try:
                    dates = pd.to_datetime(
                        raw.astype(float).astype(int),
                        unit='D', origin='1970-01-01')
                except Exception:
                    return np.arange(len(df), dtype=float)
        t0 = dates.iloc[0]
        return np.array([(d - t0).days / 365.25 for d in dates])
    return np.arange(len(df), dtype=float)


def _extract_md_series(df):
    """Get years and MD series from a VF DataFrame."""
    years = _get_years(df)
    md_col = next((c for c in ['md', 'tmd', 'MD', 'tMD'] if c in df.columns), None)
    md = df[md_col].values.astype(float) if md_col else np.zeros(len(df))
    return years, md


def progression_cigts(df_VFs_py):
    """CIGTS progression: worsening if CIGTS score increases ≥ 3 on 3 consecutive exams."""
    by_eye = df_VFs_py.groupby('eyeid') if 'eyeid' in df_VFs_py.columns else [(1, df_VFs_py)]
    results = []
    for eid, grp in by_eye:
        grp = grp.sort_values('date') if 'date' in grp.columns else grp
        scores = np.array([get_score_CIGTS(grp.iloc[[i]]) for i in range(len(grp))])
        baseline = scores[0]
        diffs = scores[1:] - baseline
        # Worsening: ≥3 on 3 consecutive post-baseline exams
        progressed = False
        for i in range(len(diffs) - 2):
            if all(diffs[i:i+3] >= 3):
                progressed = True
                break
        results.append('worsening' if progressed else 'stable')
    return tuple(results)


def progression_vfi(df_VFs_py):
    """VFI linear regression progression (slope < 0, p < 0.05)."""
    from scipy.stats import linregress

    by_eye = df_VFs_py.groupby('eyeid') if 'eyeid' in df_VFs_py.columns else [(1, df_VFs_py)]
    results = []
    for eid, grp in by_eye:
        grp = grp.sort_values('date') if 'date' in grp.columns else grp

        # support vfi, VFI, tvfi column names
        vfi_col = next((c for c in ['vfi', 'VFI', 'tVFI', 'tvfi']
                        if c in grp.columns), None)
        if vfi_col is None:
            results.append('stable')
            continue

        vfi_vals = grp[vfi_col].values.astype(float)

        # parse years
        if 'date' in grp.columns:
            dates = pd.to_datetime(grp['date'], errors='coerce')
            if dates.isna().all():
                try:
                    dates = pd.to_datetime(
                        grp['date'].astype(float).astype(int),
                        unit='D', origin='1970-01-01')
                except Exception:
                    dates = None
        else:
            dates = None

        if dates is not None and not dates.isna().all():
            t0 = dates.iloc[0]
            years = np.array([(d - t0).days / 365.25 for d in dates])
        else:
            years = np.arange(len(grp), dtype=float)

        mask = ~np.isnan(vfi_vals) & ~np.isnan(years)
        if mask.sum() < 3 or np.all(years[mask] == years[mask][0]):
            results.append('stable')
            continue

        slope, _, _, p, _ = linregress(years[mask], vfi_vals[mask])
        # Progression: VFI declining (slope < 0) with statistical significance
        flag = bool(slope < 0 and p < 0.05)
        results.append('worsening' if flag else 'stable')

    return tuple(results)


def _ols_progression(years, md, slope_thresh, p_thresh=0.05):
    """OLS linear regression, returns 'worsening' or 'stable'."""
    from scipy.stats import linregress
    mask = ~np.isnan(md) & ~np.isnan(years)
    if mask.sum() < 3 or np.all(years[mask] == years[mask][0]):
        return 'stable'
    slope, _, _, p, _ = linregress(years[mask], md[mask])
    return 'worsening' if (slope < slope_thresh and p < p_thresh) else 'stable'


def progression_plrnouri2012(df_VFs_py):
    """PLR Nouri 2012: MD slope < -1 dB/yr, p < 0.05."""
    by_eye = df_VFs_py.groupby('eyeid') if 'eyeid' in df_VFs_py.columns else [(1, df_VFs_py)]
    results = []
    for eid, grp in by_eye:
        grp = grp.sort_values('date') if 'date' in grp.columns else grp
        years, md = _extract_md_series(grp)
        results.append(_ols_progression(years, md, slope_thresh=-1.0))
    return tuple(results)


def progression_schell2014(df_VFs_py):
    """Schell 2014: MD-based progression using OLS.
    Threshold is -2 dB/year when real dates exist, or -0.5 dB/visit
    when only visit indices are available (no date column).
    """
    from scipy.stats import linregress
    by_eye = df_VFs_py.groupby('eyeid') if 'eyeid' in df_VFs_py.columns else [(1, df_VFs_py)]
    has_dates = 'date' in df_VFs_py.columns
    results = []
    for eid, grp in by_eye:
        grp = grp.sort_values('date') if has_dates else grp
        years, md = _extract_md_series(grp)
        mask = ~np.isnan(md) & ~np.isnan(years)
        if mask.sum() < 3 or np.all(years[mask] == years[mask][0]):
            results.append('stable')
            continue
        slope, _, _, p, _ = linregress(years[mask], md[mask])
        # Use dB/visit threshold when no real dates available
        thresh = -2.0 if has_dates else -0.5
        flag = bool(slope < thresh and p < 0.05)
        results.append('worsening' if flag else 'stable')
    return tuple(results)
    return tuple(results)


def progression_agis(df_VFs_py):
    """
    AGIS progression: worsening if AGIS score increases ≥4 on 3 of last 4 exams
    vs baseline (AGIS 2, Gaasterland 1994).
    """
    by_eye = df_VFs_py.groupby('eyeid') if 'eyeid' in df_VFs_py.columns else [(1, df_VFs_py)]
    results = []
    for eid, grp in by_eye:
        grp = grp.sort_values('date') if 'date' in grp.columns else grp
        if len(grp) < 5:
            raise ValueError("AGIS progression requires ≥5 VFs")
        scores = np.array([get_score_AGIS(grp.iloc[[i]]) for i in range(len(grp))])
        baseline = scores[0]
        diffs = list(reversed(scores[1:] - baseline))

        def _classify(d):
            if d >= 4:   return 'worsening'
            if d <= -3:  return 'improving'
            return 'stable'

        labels = [_classify(d) for d in diffs]
        final_set = set(labels[:3])
        if len(final_set) == 1:
            final = labels[0]
            interim = labels[3:]
            if any(x != 'stable' and x != final for x in interim):
                final = 'stable'
        else:
            final = 'stable'
        results.append(final)
    return tuple(results)


# ---------------------------------------------------------------------------
# Plotting helpers from vfprogression
# ---------------------------------------------------------------------------

def plotValues(values, title='Deviation', save=False, filename='tmp', fmt='pdf'):
    """Plain-text VF grid — matches R's plotTDvalues (vfprogression convention: nasal LEFT)."""
    from PyVisualFields.visualFields import _GRID_24D2_STD
    vals = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, pos in enumerate(_GRID_24D2_STD):
        if pos is None or i >= len(vals):
            continue
        c, r = pos
        v = vals[i]
        if not np.isnan(v):
            ax.text(c, 7 - r, f'{int(round(v))}',
                    ha='center', va='center', fontsize=9, color='black')
    ax.axvline(x=4.5, color='black', linewidth=1.0)
    ax.axhline(y=3.5, color='black', linewidth=1.0)
    ax.set_xlim(-0.6, 8.6); ax.set_ylim(-0.6, 7.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    if save:
        fig.savefig(f'{filename}.{fmt}', bbox_inches='tight', dpi=150)
    plt.show()


def plotProbabilities(values, title='Probability', save=False, filename='tmp', fmt='pdf'):
    """Symbol probability plot — matches R's plotTdProbabilities (nasal LEFT)."""
    from PyVisualFields.visualFields import _GRID_24D2_STD
    _SYM = [
        (0.005, 's', 18, 'black'),
        (0.01,  's', 11, 'black'),
        (0.02,  's',  7, 'none'),
        (0.05,  'o',  5, 'black'),
        (1.0,   'o',  2, 'black'),
    ]
    vals = np.array(values, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, pos in enumerate(_GRID_24D2_STD):
        if pos is None or i >= len(vals):
            continue
        c, r = pos
        p = vals[i]
        y = 7 - r
        for thresh, marker, sz, fc in _SYM:
            if p <= thresh:
                mfc = fc if fc != 'none' else 'white'
                ax.plot(c, y, marker=marker, markersize=sz,
                        markerfacecolor=mfc, markeredgecolor='black',
                        markeredgewidth=0.8)
                break
    ax.axvline(x=4.5, color='black', linewidth=1.0)
    ax.axhline(y=3.5, color='black', linewidth=1.0)
    ax.set_xlim(-0.6, 8.6); ax.set_ylim(-0.6, 7.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold')
    legend_items = [
        plt.Line2D([0],[0], marker='o', markersize=2,  color='w', markerfacecolor='black', markeredgecolor='black', label='p≥0.05'),
        plt.Line2D([0],[0], marker='o', markersize=5,  color='w', markerfacecolor='black', markeredgecolor='black', label='p<0.05'),
        plt.Line2D([0],[0], marker='s', markersize=7,  color='w', markerfacecolor='white', markeredgecolor='black', label='p<0.02'),
        plt.Line2D([0],[0], marker='s', markersize=11, color='w', markerfacecolor='black', markeredgecolor='black', label='p<0.01'),
        plt.Line2D([0],[0], marker='s', markersize=18, color='w', markerfacecolor='black', markeredgecolor='black', label='p<0.005'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=7, frameon=True)
    plt.tight_layout()
    if save:
        fig.savefig(f'{filename}.{fmt}', bbox_inches='tight', dpi=150)
    plt.show()