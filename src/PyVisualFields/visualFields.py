
"""
visualFields.py — pure Python replacement.
All R/rpy2 calls replaced with vf_core.py + matplotlib.
API surface preserved so existing notebooks keep working.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection

from PyVisualFields import vf_core

# Exact HFA 24-2 grid from R visualFields locmaps$p24d2$coord
# X-axis is FLIPPED to match R's OD display convention:
# temporal (positive x) on LEFT, nasal (negative x) on RIGHT
# col = 8 - (x_degree + 27) / 6
# Blind spots: 1-indexed points 26 (x=15,y=3) and 35 (x=15,y=-3) → None
# HFA 24-2 grid — nasal LEFT, temporal RIGHT (standard R visualFields convention)
# Both visualFields and vfprogression R packages use this same orientation
_GRID_24D2 = [
    (3,0),(4,0),(5,0),(6,0),
    (2,1),(3,1),(4,1),(5,1),(6,1),(7,1),
    (1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),(8,2),
    (0,3),(1,3),(2,3),(3,3),(4,3),(5,3),(6,3),None,(8,3),
    (0,4),(1,4),(2,4),(3,4),(4,4),(5,4),(6,4),None,(8,4),
    (1,5),(2,5),(3,5),(4,5),(5,5),(6,5),(7,5),(8,5),
    (2,6),(3,6),(4,6),(5,6),(6,6),(7,6),
    (3,7),(4,7),(5,7),(6,7),
]  # 54 entries — standard orientation (nasal LEFT)

# Alias for vfprogression — same grid
_GRID_24D2_STD = _GRID_24D2
_BLIND_SPOT_IDX = frozenset(i for i, pos in enumerate(_GRID_24D2) if pos is None)  # {25, 34}

# Probability colormap matching R visualFields (background fill + border color)
# Below normal (p < threshold): colored fill, black text
# Above normal (p > 0.95): white fill with colored border
_VF_PROB_SCHEME = [
    # (upper_prob, fill_color, border_color, text_color)
    (0.005, '#000000', '#000000', 'white'),   # black
    (0.010, '#8B0000', '#8B0000', 'white'),   # dark red
    (0.020, '#CC0000', '#CC0000', 'white'),   # red
    (0.050, '#FF8C00', '#FF8C00', 'black'),   # orange
    (0.950, '#F5F0E8', '#CCCCCC', 'black'),   # normal (beige)
    (0.980, '#F5F0E8', '#228B22', 'black'),   # light green border
    (0.990, '#F5F0E8', '#228B22', 'black'),   # green border
    (0.995, '#F5F0E8', '#006400', 'black'),   # dark green border
    (1.001, '#F5F0E8', '#004400', 'black'),   # darkest green border
]

def _get_prob_colors(p_val):
    """Return (fill, border, text) colors for a given p-value."""
    for thresh, fill, border, text in _VF_PROB_SCHEME:
        if p_val <= thresh:
            return fill, border, text
    return '#F5F0E8', '#CCCCCC', 'black'


# ── REPLACE _vf_grid_plot ────────────────────────────────────────────────────
def _vf_grid_plot(values, title='', colormap=None, cbar_label='dB',
                  vmin=None, vmax=None, fmt='.1f', ax=None, figsize=(7, 6)):
    """
    Generic 24-2 grid plot.
    Layout: white grid axes (top, ratio 11) + transparent colorbar axes (bottom, ratio 1).
    Figure background is transparent so the colorbar floats on Qt's grey window.
    """
    from matplotlib.gridspec import GridSpec

    if ax is None:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor('none')            # transparent figure bg
        gs  = GridSpec(2, 1, figure=fig,
                       height_ratios=[11, 1],
                       hspace=0.15)
        ax  = fig.add_subplot(gs[0])
        ax.set_facecolor('white')                  # white grid box
        cax = fig.add_subplot(gs[1])
        cax.set_facecolor('none')                  # transparent colorbar bg
    else:
        fig = ax.get_figure()
        cax = None

    vals = np.array(values, dtype=float)
    if vmin is None: vmin = np.nanmin(vals)
    if vmax is None: vmax = np.nanmax(vals)
    cmap = colormap or plt.cm.RdYlGn

    for i, pos in enumerate(_GRID_24D2):
        if pos is None:
            continue
        c, r = pos
        v = vals[i] if i < len(vals) else np.nan
        color = ('lightgrey' if np.isnan(v)
                 else cmap((v - vmin) / max(vmax - vmin, 1e-6)))
        rect = mpatches.FancyBboxPatch(
            (c - 0.45, 7 - r - 0.45), 0.9, 0.9,
            boxstyle='round,pad=0.05', color=color, ec='grey', lw=0.5)
        ax.add_patch(rect)
        if not np.isnan(v):
            brightness = (v - vmin) / max(vmax - vmin, 1)
            ax.text(c, 7 - r, format(v, fmt),
                    ha='center', va='center', fontsize=7,
                    color='black' if brightness > 0.4 else 'white')

    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-0.6, 7.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)
    ax.axvline(x=4.5, color='grey', linewidth=0.5)
    ax.axhline(y=3.5, color='grey', linewidth=0.5)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    if cax is not None:
        # Draw gradient strip entirely inside cax — no text outside axes bounds.
        # This ensures bbox_inches='tight' produces the same height as the
        # probability strip in TD/PD so all three panels align perfectly.
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        cax.imshow(gradient, aspect='auto', cmap=cmap,
                   extent=[0, 1, 0, 1], transform=cax.transAxes,
                   vmin=0, vmax=1, origin='lower')
        # "0 dB" on the left, "35 dB" on the right — inside the strip
        cax.text(0.01, 0.5, f'{int(vmin)} {cbar_label}',
                 ha='left', va='center', fontsize=9,
                 color='white', transform=cax.transAxes)
        cax.text(0.99, 0.5, f'{int(vmax)} {cbar_label}',
                 ha='right', va='center', fontsize=9,
                 color='black', transform=cax.transAxes)
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.axis('off')
    else:
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04,
                     label=cbar_label, orientation='horizontal',
                     location='bottom')

    return fig, ax

# ── REPLACE _vf_prob_plot ────────────────────────────────────────────────────
def _vf_prob_plot(values, probs, title='', fmt='.0f', ax=None):
    """
    VF grid plot with probability-based coloring.
    Same layout as _vf_grid_plot: white grid axes (top) +
    transparent probability-strip axes (bottom), transparent figure bg.
    """
    from matplotlib.gridspec import GridSpec

    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        fig.patch.set_facecolor('none')            # transparent figure bg
        gs  = GridSpec(2, 1, figure=fig,
                       height_ratios=[11, 1],
                       hspace=0.15)
        ax  = fig.add_subplot(gs[0])
        ax.set_facecolor('white')                  # white grid box
        cax = fig.add_subplot(gs[1])
        cax.set_facecolor('none')                  # transparent colorbar bg
    else:
        fig = ax.get_figure()
        cax = None

    vals = np.array(values, dtype=float)
    prbs = np.array(probs,  dtype=float)

    for i, pos in enumerate(_GRID_24D2):
        if pos is None:
            continue
        c, r = pos
        y = 7 - r
        v = vals[i] if i < len(vals) else np.nan
        p = prbs[i] if i < len(prbs) else 1.0

        if i == 25 or i == 34:
            ellipse = mpatches.Ellipse((c, y), 0.8, 0.7,
                                       color='#BBBBBB', ec='#999999', lw=1)
            ax.add_patch(ellipse)
            continue

        fill, border, text_col = _get_prob_colors(p)
        rect = mpatches.FancyBboxPatch(
            (c - 0.44, y - 0.44), 0.88, 0.88,
            boxstyle='round,pad=0.04',
            facecolor=fill, edgecolor='black', linewidth=1.0)
        ax.add_patch(rect)
        if not np.isnan(v):
            display_v = 0.0 if abs(v) < 0.5 else v
            ax.text(c, y, format(display_v, fmt),
                    ha='center', va='center', fontsize=11,
                    color=text_col,
                    fontweight='bold' if p < 0.05 else 'normal')

    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-0.6, 7.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10)

    # ── Probability colour strip in cax ──────────────────────────────────
    if cax is not None:
        _strip_labels  = ['0',    '0.005',   '0.01',    '0.02',
                          '0.05', '0.95',    '0.98',    '0.99',
                          '0.995','1']
        _strip_fills   = ['#000000','#8B0000','#CC0000','#FF8C00',
                          '#F5F0E8','#F5F0E8','#F5F0E8','#F5F0E8',
                          '#F5F0E8','#F5F0E8']
        _strip_borders = ['#000000','#8B0000','#CC0000','#FF8C00',
                          '#CCCCCC','#228B22','#228B22','#006400',
                          '#004400','#004400']
        n = len(_strip_labels)
        for k, (lbl, fc, ec) in enumerate(
                zip(_strip_labels, _strip_fills, _strip_borders)):
            rect = mpatches.FancyBboxPatch(
                (k / n, 0.05), 0.9 / n, 0.9,
                boxstyle='round,pad=0.01',
                transform=cax.transAxes,
                facecolor=fc, edgecolor=ec, linewidth=1.5,
                clip_on=False)
            cax.add_patch(rect)
            tc = 'white' if fc in ('#000000', '#8B0000', '#CC0000') else 'black'
            cax.text((k + 0.45) / n, 0.5, lbl,
                     ha='center', va='center',
                     fontsize=8, color=tc,
                     transform=cax.transAxes)
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.axis('off')

    return fig, ax


# ── REPLACE vfplot ───────────────────────────────────────────────────────────
def vfplot(df_vf_py, type='s', save=False, filename='tmp', fmt='pdf'):
    """Plot a single VF exam. type: 's'=sensitivity, 'td'=TD, 'pd'=PD,
    'tds'=TD probability, 'pds'=PD probability."""
    row   = df_vf_py.iloc[0]
    scols = _get_sensitivity_cols(df_vf_py)
    sens  = _row_to_array(row, scols)
    age   = float(row['age']) if 'age' in row else 60.0

    if type == 's':
        fig, _ = _vf_grid_plot(sens, title='', vmin=0, vmax=35,
                               colormap=plt.cm.gray, figsize=(7, 6))
    elif type == 'td':
        res  = vf_core.compute_indices(sens, age, norm_db=_active_norm_db)
        prbs = vf_core.probability_map(res['td'])
        fig, _ = _vf_prob_plot(res['td'], prbs, title='')
    elif type == 'pd':
        res  = vf_core.compute_indices(sens, age, norm_db=_active_norm_db)
        prbs = vf_core.probability_map(res['pd'])
        fig, _ = _vf_prob_plot(res['pd'], prbs, title='')
    elif type == 'tds':
        res  = vf_core.compute_indices(sens, age, norm_db=_active_norm_db)
        prbs = vf_core.probability_map(res['td'])
        fig, _ = _vf_prob_plot(res['td'], prbs, title='', fmt='.0f')
    elif type == 'pds':
        res  = vf_core.compute_indices(sens, age, norm_db=_active_norm_db)
        prbs = vf_core.probability_map(res['pd'])
        fig, _ = _vf_prob_plot(res['pd'], prbs, title='', fmt='.0f')
    else:
        raise ValueError(f"Unknown plot type: {type}")

    if save:
        # Always save with transparency so the colorbar strip floats on
        # Qt's grey window background, separated from the white grid box.
        fig.savefig(f'{filename}.{fmt}', bbox_inches='tight', dpi=150,
                    transparent=True)
    plt.show()

def _parse_years(df):
    """Extract years-from-baseline robustly from a date column."""
    if 'date' not in df.columns:
        return np.arange(len(df), dtype=float)

    raw = df['date'].reset_index(drop=True)

    def _to_years(dates):
        t0 = dates.iloc[0]
        return np.array([(d - t0).days / 365.25 for d in dates])

    def _is_valid(years):
        return len(years) <= 1 or not np.all(years == years[0])

    # Case 1: already datetime64
    if pd.api.types.is_datetime64_any_dtype(raw):
        years = _to_years(raw)
        if _is_valid(years):
            return years

    # Case 2: R integer days since 1970-01-01 (stored as int or float)
    try:
        int_vals = pd.to_numeric(raw, errors='coerce').dropna().astype(int)
        if len(int_vals) == len(raw) and int_vals.between(0, 40000).all():
            dates = pd.to_datetime(int_vals, unit='D', origin='1970-01-01')
            years = _to_years(dates)
            if _is_valid(years):
                return years
    except Exception:
        pass

    # Case 3: string date
    try:
        dates = pd.to_datetime(raw, errors='coerce')
        if not dates.isna().all():
            years = _to_years(dates)
            if _is_valid(years):
                return years
    except Exception:
        pass

    # Last resort
    return np.arange(len(df), dtype=float)

def _prob_grid_plot(prob_values, title='Probability', ax=None):
    """Plot probability map with standard 5-level color coding."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    probs = np.array(prob_values, dtype=float)
    for i, pos in enumerate(_GRID_24D2):
        if pos is None:
            continue
        c, r = pos
        p = probs[i] if i < len(probs) else 1.0
        # snap to nearest level
        level = min(_PROB_COLORS.keys(), key=lambda x: abs(x - p))
        fc = _PROB_COLORS[level]
        tc = _PROB_TEXT_COLORS[level]
        rect = mpatches.FancyBboxPatch((c - 0.45, 7 - r - 0.45), 0.9, 0.9,
                                        boxstyle='round,pad=0.05', color=fc, ec='grey', lw=0.5)
        ax.add_patch(rect)
        ax.text(c, 7 - r, f'{p:.3f}', ha='center', va='center', fontsize=6, color=tc)

    # legend
    handles = [mpatches.Patch(color=c, label=f'p={p}') for p, c in _PROB_COLORS.items()]
    ax.legend(handles=handles, loc='lower right', fontsize=7)
    ax.set_xlim(-0.6, 8.6); ax.set_ylim(-0.6, 7.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10)
    return fig, ax


# ---------------------------------------------------------------------------
# Helper: extract sensitivity columns from a VF dataframe row
# ---------------------------------------------------------------------------

def _get_sensitivity_cols(df):
    """
    Return ordered list of sensitivity column names.
    Handles naming conventions: p1..p54, s1..s54, sens1..sens54, l1..l54.
    """
    for prefix in ('s', 'l', 'p', 'sens', 'vf'):
        cols = sorted(
            [c for c in df.columns if c.startswith(prefix) and c[len(prefix):].isdigit()],
            key=lambda x: int(x[len(prefix):])
        )
        if len(cols) >= 52:
            return cols
    # fallback: any numeric-suffix columns that look like point data
    cols = sorted(
        [c for c in df.columns if c[-1].isdigit() and c.rstrip('0123456789') not in
         ('age', 'date', 'id', 'eye', 'time', 'fpr', 'fnr', 'fl', 'duration', 'eyeid', 'patientid')],
        key=lambda x: int(x.lstrip('abcdefghijklmnopqrstuvwxyz'))
    )
    return cols

def _row_to_array(row, cols):
    arr = row[cols].values.astype(float)
    for i in _BLIND_SPOT_IDX:
        if i < len(arr):
            arr[i] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Part I: Sample datasets (embedded minimal CSVs from R package)
# ---------------------------------------------------------------------------

def _load_bundled(name):
    """Load a bundled CSV from the package data directory, if available."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(pkg_dir, 'data', f'{name}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'date' in df.columns:
            raw = df['date']
            # Detect R integer days since 1970-01-01 (values typically 10000-25000)
            try:
                int_vals = pd.to_numeric(raw, errors='coerce')
                if int_vals.notna().all() and int_vals.between(1000, 40000).all():
                    df['date'] = pd.to_datetime(int_vals.astype(int),
                                                unit='D', origin='1970-01-01')
                else:
                    df['date'] = pd.to_datetime(raw, errors='coerce')
            except Exception:
                df['date'] = pd.to_datetime(raw, errors='coerce')
        return df
    raise FileNotFoundError(
        f"Bundled dataset '{name}' not found at {path}. "
        "Export it once from R with: write.csv(vfctrSunyiu24d2, 'vfctrSunyiu24d2.csv')"
    )

def data_vfctrSunyiu24d2():   return _load_bundled('vfctrSunyiu24d2')
def data_vfctrSunyiu10d2():   return _load_bundled('vfctrSunyiu10d2')
def data_vfpwgRetest24d2():   return _load_bundled('vfpwgRetest24d2')
def data_vfpwgSunyiu24d2():   return _load_bundled('vfpwgSunyiu24d2')
def data_vfctrIowaPC26():     return _load_bundled('vfctrIowaPC26')
def data_vfctrIowaPeri():     return _load_bundled('vfctrIowaPeri')

def vfplot_s(df, **kw):   vfplot(df, type='s', **kw)
def vfplot_td(df, **kw):  vfplot(df, type='td', **kw)
def vfplot_pd(df, **kw):  vfplot(df, type='pd', **kw)
def vfplot_tds(df, **kw): vfplot(df, type='tds', **kw)
def vfplot_pds(df, **kw): vfplot(df, type='pds', **kw)


def vfplotsparklines(df_vf_py, type='s', save=False, filename='tmp', fmt='pdf'):
    """
    Sparkline plot — one grid with a mini time-series line per test location.
    Matches R visualFields vfsparklines: each cell shows value over visits.
    Red line = declining trend, black = stable/improving.
    """
    scols = _get_sensitivity_cols(df_vf_py)
    age_arr = df_vf_py['age'].values.astype(float) if 'age' in df_vf_py.columns \
              else np.full(len(df_vf_py), 60.)
    raw = df_vf_py[scols].values.astype(float)  # shape (nvisits, npts)
    nvisits = raw.shape[0]
    x = np.arange(nvisits)

    # Build data matrix per location
    if type == 's':
        data = raw
    elif type == 'td':
        data = np.vstack([vf_core.total_deviation(raw[i], age_arr[i])
                          for i in range(nvisits)])
    elif type == 'pd':
        data = np.vstack([vf_core.pattern_deviation(
                              vf_core.total_deviation(raw[i], age_arr[i]))
                          for i in range(nvisits)])
    else:
        raise ValueError(f"Unknown sparkline type: {type}")

    cell_w, cell_h = 0.85, 0.85
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(-0.6, 8.6)
    ax.set_ylim(-0.6, 7.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Sparklines ({type})', fontsize=10)

    # Draw grid cell backgrounds
    for i, pos in enumerate(_GRID_24D2):
        if pos is None:
            continue
        c, r = pos
        y = 7 - r
        if i in (25, 34):  # blind spot
            ellipse = mpatches.Ellipse((c, y), cell_w * 0.8, cell_h * 0.6,
                                        color='#BBBBBB', ec='#999999', lw=0.8)
            ax.add_patch(ellipse)
            continue
        rect = mpatches.FancyBboxPatch((c - cell_w/2, y - cell_h/2),
                                        cell_w, cell_h,
                                        boxstyle='round,pad=0.03',
                                        facecolor='#F8F8F8',
                                        edgecolor='#CCCCCC', linewidth=0.6)
        ax.add_patch(rect)

        if i >= data.shape[1]:
            continue
        series = data[:, i]
        if np.all(np.isnan(series)):
            continue

        # Normalise series to fit within cell
        vmin_s = np.nanmin(data)
        vmax_s = np.nanmax(data)
        rng = max(vmax_s - vmin_s, 1e-6)
        sy = (series - vmin_s) / rng  # 0-1
        sx = np.linspace(0, 1, nvisits)

        # Map to cell coordinates
        px = c - cell_w/2 + 0.05 + sx * (cell_w - 0.10)
        py = y - cell_h/2 + 0.05 + sy * (cell_h - 0.10)

        # Red if trend is declining (slope < 0), black otherwise
        if nvisits >= 2:
            from scipy.stats import linregress
            slope, *_ = linregress(x[~np.isnan(series)],
                                   series[~np.isnan(series)])
            lcolor = 'red' if slope < 0 else 'black'
        else:
            lcolor = 'black'

        ax.plot(px, py, color=lcolor, linewidth=0.7, solid_capstyle='round')

    # Hemifield dividers
    ax.axvline(x=4.5, color='#AAAAAA', linewidth=0.8)
    ax.axhline(y=3.5, color='#AAAAAA', linewidth=0.8)

    plt.tight_layout()
    if save:
        fig.savefig(f'{filename}.{fmt}', bbox_inches='tight', dpi=150)
    plt.show()

def vfplotsparklines_s(df, **kw):  vfplotsparklines(df, type='s', **kw)
def vfplotsparklines_td(df, **kw): vfplotsparklines(df, type='td', **kw)
def vfplotsparklines_pd(df, **kw): vfplotsparklines(df, type='pd', **kw)


def vfplotplr(df_vf_py, type='s', save=False, filename='tmp', fmt='pdf'):
    """
    PLR slope map — matches R vfplotplr.
    Shows slope values (dB/yr) with probability-based background coloring.
    Color reflects p-value of the slope test (left-tailed).
    """
    res = plr(df_vf_py, type=type)
    slopes = res['sl']       # shape (n_pts,)
    pvals  = res['pval']     # shape (n_pts,)

    # Convert left-tailed p-values to probability scale for colormap:
    # p-value here is for H0: slope=0 vs slope<0
    # Use 1 - pval so that significant negative slopes → low probability → red
    probs = 1.0 - pvals

    fig, _ = _vf_prob_plot(np.round(slopes, 1), probs,
                            title=f'PLR slopes ({type})', fmt='.1f')
    if save:
        fig.savefig(f'{filename}.{fmt}', bbox_inches='tight', dpi=150)
    plt.show()

def vfplotplr_s(df, **kw):  vfplotplr(df, type='s', **kw)
def vfplotplr_td(df, **kw): vfplotplr(df, type='td', **kw)
def vfplotplr_pd(df, **kw): vfplotplr(df, type='pd', **kw)


def vflegoplot(df_vf_py, type='s', thr=2, save=False, filename='tmp', fmt='pdf'):
    """
    Lego plot — matches R vflegoplot exactly.
    Shows mean(last thr visits) - mean(first thr visits) per location.
    Grayscale: darker circle = more negative change (worsening).
    Circle size proportional to absolute change magnitude.
    """
    scols   = _get_sensitivity_cols(df_vf_py)
    age_arr = df_vf_py['age'].values.astype(float) if 'age' in df_vf_py.columns \
              else np.full(len(df_vf_py), 60.)
    raw     = df_vf_py[scols].values.astype(float)  # (nvisits, n_pts)
    nvisits = len(raw)
    thr     = max(1, min(thr, nvisits // 2))

    if type == 's':
        data = raw
    elif type == 'td':
        data = np.vstack([vf_core.total_deviation(raw[i], age_arr[i],
                          norm_db=_active_norm_db) for i in range(nvisits)])
    elif type == 'pd':
        data = np.vstack([vf_core.pattern_deviation(
                              vf_core.total_deviation(raw[i], age_arr[i],
                              norm_db=_active_norm_db)) for i in range(nvisits)])
    else:
        raise ValueError(f"Unknown lego type: {type}")

    baseline = np.nanmean(data[:thr],  axis=0)   # avg of first thr visits
    followup = np.nanmean(data[-thr:], axis=0)   # avg of last  thr visits
    diff     = followup - baseline                # change per location

    abs_max = max(np.nanmax(np.abs(diff)), 1e-6)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(-0.6, 8.6); ax.set_ylim(-0.6, 7.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'Lego plot ({type})', fontsize=10)

    for i, pos in enumerate(_GRID_24D2):
        if pos is None:
            continue
        c, r = pos
        y = 7 - r

        if i in (25, 34):   # blind spot
            ellipse = mpatches.Ellipse((c, y), 0.7, 0.55,
                                        color='#BBBBBB', ec='#999999', lw=0.8)
            ax.add_patch(ellipse)
            continue

        if i >= len(diff): continue
        v = diff[i]
        if np.isnan(v): continue

        # Light grey background cell
        rect = mpatches.FancyBboxPatch((c - 0.44, y - 0.44), 0.88, 0.88,
                                        boxstyle='round,pad=0.04',
                                        facecolor='#EFEFEF',
                                        edgecolor='#BBBBBB', linewidth=0.5)
        ax.add_patch(rect)

        # Circle: size = |change|, gray level = magnitude
        # Negative (worsening) → dark; positive (improving) → light gray
        norm_mag = np.clip(abs(v) / abs_max, 0, 1)
        radius   = 0.08 + 0.34 * norm_mag
        if v < 0:
            gray = 1.0 - norm_mag          # 0=black (worst), 1=white
        else:
            gray = 0.7 + 0.3 * norm_mag   # light gray for improvements
        color = (gray, gray, gray)
        tc    = 'white' if gray < 0.45 else 'black'

        circle = mpatches.Circle((c, y), radius=radius,
                                   facecolor=color, edgecolor='#888888', lw=0.5)
        ax.add_patch(circle)
        ax.text(c, y, f'{v:.1f}', ha='center', va='center',
                fontsize=6.5, color=tc)

    ax.axvline(x=4.5, color='#AAAAAA', lw=0.8)
    ax.axhline(y=3.5, color='#AAAAAA', lw=0.8)

    # Grayscale colorbar: white=0 change, black=max worsening
    cmap_cb = plt.cm.gray_r
    sm = plt.cm.ScalarMappable(cmap=cmap_cb,
                                norm=mcolors.Normalize(vmin=0, vmax=abs_max))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label='|dB change|')
    plt.tight_layout()
    if save:
        fig.savefig(f'{filename}.{fmt}', bbox_inches='tight', dpi=150)
    plt.show()


def vflegoplot_s(df, **kw):  vflegoplot(df, type='s', **kw)
def vflegoplot_td(df, **kw): vflegoplot(df, type='td', **kw)
def vflegoplot_pd(df, **kw): vflegoplot(df, type='pd', **kw)


def vfsfa(df_vf_py, filename='report.pdf'):
    """Single field analysis summary — saves a matplotlib figure as PDF."""
    row   = df_vf_py.iloc[0]
    scols = _get_sensitivity_cols(df_vf_py)
    sens  = _row_to_array(row, scols)
    age   = float(row['age']) if 'age' in row else 60.0
    res   = vf_core.compute_indices(sens, age, norm_db=_active_norm_db)
    prbs_td = vf_core.probability_map(res['td'])
    prbs_pd = vf_core.probability_map(res['pd'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Sensitivity: grayscale
    _vf_grid_plot(sens, title='Sensitivity (dB)', vmin=0, vmax=35,
                  colormap=plt.cm.gray, ax=axes[0])

    # TD: probability-based colormap
    _vf_prob_plot(res['td'], prbs_td, title='Total Deviation (dB)',
                  fmt='.0f', ax=axes[1])

    # PD: probability-based colormap
    _vf_prob_plot(res['pd'], prbs_pd, title='Pattern Deviation (dB)',
                  fmt='.0f', ax=axes[2])

    fig.suptitle(
        f"MD={res['MD']:.2f} dB   PSD={res['PSD']:.2f} dB   VFI={res['VFI']:.1f}%",
        fontsize=11)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")


def plotProbColormap(save=False, filename='tmp', fmt='pdf'):
    """Display the standard probability colormap matching R visualFields."""
    labels = ['0', '0.005', '0.01', '0.02', '0.05', '0.95', '0.98', '0.99', '0.995', '1']
    colors = ['#000000','#8B0000','#CC0000','#FF8C00','#F5F0E8',
              '#F5F0E8','#F5F0E8','#F5F0E8','#F5F0E8','#F5F0E8']
    borders= ['#000000','#8B0000','#CC0000','#FF8C00','#CCCCCC',
              '#228B22','#228B22','#006400','#004400','#004400']
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8, 0.8))
    for i, (lbl, fc, ec) in enumerate(zip(labels, colors, borders)):
        rect = mpatches.FancyBboxPatch((i, 0.1), 0.9, 0.8,
                                        boxstyle='round,pad=0.02',
                                        facecolor=fc, edgecolor=ec, linewidth=2)
        ax.add_patch(rect)
        tc = 'white' if fc in ('#000000','#8B0000','#CC0000') else 'black'
        ax.text(i + 0.45, 0.5, lbl, ha='center', va='center', fontsize=7, color=tc)
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()
    if save:
        fig.savefig(f'{filename}.{fmt}', bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Part III: Computations
# ---------------------------------------------------------------------------

def _build_output_df(df_in, new_cols_prefix, values_2d, col_names):
    """Attach computed per-point columns to a copy of the metadata columns."""
    sens_cols = set(_get_sensitivity_cols(df_in))
    meta_cols = [c for c in df_in.columns if c not in sens_cols]
    out = df_in[meta_cols].copy().reset_index(drop=True)
    for i, cn in enumerate(col_names):
        out[cn] = values_2d[:, i]
    return out


def getallvalues(df_vf_py):
    scols = _get_sensitivity_cols(df_vf_py)
    col_n = len(scols)
    td_arr = np.zeros((len(df_vf_py), col_n))
    pd_arr = np.zeros_like(td_arr)
    tdp_arr = np.zeros_like(td_arr)
    pdp_arr = np.zeros_like(td_arr)
    md_arr, psd_arr, vfi_arr, gh_arr = [], [], [], []
    for i, (_, row) in enumerate(df_vf_py.iterrows()):
        sens = _row_to_array(row, scols)
        age  = float(row['age']) if 'age' in row else 60.0
        res  = vf_core.compute_indices(sens, age, norm_db=_active_norm_db)
        td_arr[i]  = res['td'][:col_n]
        pd_arr[i]  = res['pd'][:col_n]
        tdp_arr[i] = vf_core.probability_map(res['td'])[:col_n]
        pdp_arr[i] = vf_core.probability_map(res['pd'])[:col_n]
        md_arr.append(res['MD']); psd_arr.append(res['PSD'])
        vfi_arr.append(res['VFI'])
        td_valid = res['td'][~np.isnan(res['td'])]
        gh_arr.append(float(np.percentile(td_valid, 85)))

    td_cols  = [f'l{c[1:]}'  for c in scols]
    pd_cols  = [f'l{c[1:]}'  for c in scols]
    tdp_cols = [f'l{c[1:]}' for c in scols]
    pdp_cols = [f'l{c[1:]}' for c in scols]

    TotalDev     = _build_output_df(df_vf_py, 'td',  td_arr,  td_cols)
    PatternDev   = _build_output_df(df_vf_py, 'pd',  pd_arr,  pd_cols)
    TotalDevProbs= _build_output_df(df_vf_py, 'tdp', tdp_arr, tdp_cols)
    PatternDevProbs=_build_output_df(df_vf_py,'pdp', pdp_arr, pdp_cols)

    meta = df_vf_py[[c for c in df_vf_py.columns
                     if c not in set(scols)]].copy().reset_index(drop=True)
    GlobalIndices = meta.copy()
    GlobalIndices['md']  = md_arr
    GlobalIndices['tmd'] = md_arr   # alias used by original R package
    GlobalIndices['psd'] = psd_arr
    GlobalIndices['vfi'] = vfi_arr
    GlobalIndicesProbs = GlobalIndices.copy()  # p-values not yet implemented

    GeneralHeight = pd.DataFrame({'gh': gh_arr})

    return (TotalDev, TotalDevProbs, GlobalIndices,
            GlobalIndicesProbs, PatternDev, PatternDevProbs, GeneralHeight)


def gettd(df_vf_py):
    td, _, _, _, _, _, _ = getallvalues(df_vf_py)
    return td

def gettdp(df_td_py):
    scols = sorted([c for c in df_td_py.columns
                    if (c.startswith('td') and c[2:].isdigit()) or
                       (c.startswith('l')  and c[1:].isdigit())],
                   key=lambda x: int(x.lstrip('tdl')))
    col_n = len(scols)
    probs = np.zeros((len(df_td_py), col_n))
    for i, (_, row) in enumerate(df_td_py.iterrows()):
        td = row[scols].values.astype(float)
        probs[i] = vf_core.probability_map(td)
    out_cols = [f'l{c.lstrip("tdl")}' for c in scols]
    return _build_output_df(df_td_py, 'l', probs, out_cols)

def getpd(df_td_py):
    scols = sorted([c for c in df_td_py.columns
                    if (c.startswith('td') and c[2:].isdigit()) or
                       (c.startswith('l')  and c[1:].isdigit())],
                   key=lambda x: int(x.lstrip('tdl')))
    col_n = len(scols)
    pd_arr = np.zeros((len(df_td_py), col_n))
    for i, (_, row) in enumerate(df_td_py.iterrows()):
        td = row[scols].values.astype(float)
        pd_arr[i] = vf_core.pattern_deviation(td)
    out_cols = [f'l{c.lstrip("tdl")}' for c in scols]
    return _build_output_df(df_td_py, 'l', pd_arr, out_cols)

def getpdp(df_pd_py):
    scols = sorted([c for c in df_pd_py.columns
                    if (c.startswith('pd') and c[2:].isdigit()) or
                       (c.startswith('l')  and c[1:].isdigit())],
                   key=lambda x: int(x.lstrip('pdl')))
    col_n = len(scols)
    probs = np.zeros((len(df_pd_py), col_n))
    for i, (_, row) in enumerate(df_pd_py.iterrows()):
        pd_v = row[scols].values.astype(float)
        probs[i] = vf_core.probability_map(pd_v)
    out_cols = [f'l{c.lstrip("pdl")}' for c in scols]
    return _build_output_df(df_pd_py, 'l', probs, out_cols)

def getgh(df_td_py):
    scols = [c for c in df_td_py.columns if c.startswith('td') and c[2:].isdigit()]
    gh = []
    for _, row in df_td_py.iterrows():
        td = row[scols].values.astype(float)
        valid = td[~np.isnan(td)]
        gh.append(float(np.percentile(valid, 85)))
    return pd.DataFrame({'gh': gh})

def getgl(df_vf_py):
    _, _, gi, _, _, _, _ = getallvalues(df_vf_py)
    return gi

def getglp(df_gi_py):
    # p-values for global indices — placeholder (not trivial without normative SD)
    out = df_gi_py.copy()
    for col in ['md', 'psd', 'vfi']:
        if col in out.columns:
            out[f'{col}p'] = np.nan
    return out

def vfdesc(df):
    print(df.describe())


# ---------------------------------------------------------------------------
# Part IV: Normative value management (stubs — no R)
# ---------------------------------------------------------------------------

def normvals():
    return vf_core.load_normative_db()

def get_info_normvals():
    print("Pure-Python mode: using vf_core normative DB.")
    print("Keys: default_24d2 (Heijl 1987 approximation)")

# Module-level active normative DB (None = use vf_core default)
_active_norm_db = None

def setnv(inp):
    global _active_norm_db
    if isinstance(inp, np.ndarray):
        _active_norm_db = inp
        print(f"setnv: custom normative DB set ({inp.shape[0]} points).")
    else:
        print(f"setnv: '{inp}' — named normative sets not implemented in pure-Python mode.")

def setdefaults():
    global _active_norm_db
    _active_norm_db = None
    print("Defaults set: using vf_core built-in normative DB (Heijl 1987 24-2).")

def getnv():
    if _active_norm_db is not None:
        print(f"Current normative: custom DB ({_active_norm_db.shape[0]} points).")
    else:
        print("Current normative: vf_core default (Heijl 1987 24-2).")
    return _active_norm_db

def nvgenerate(df_py, method='pointwise', name='custom', **kw):
    scols = _get_sensitivity_cols(df_py)
    all_sens = df_py[scols].values.astype(float)
    ages = df_py['age'].values.astype(float) if 'age' in df_py.columns else np.full(len(df_py), 60.)
    # Pointwise: fit intercept + slope × age per location
    from scipy.stats import linregress
    n_pts = all_sens.shape[1]
    nv = np.zeros((n_pts, 2))
    for j in range(n_pts):
        col = all_sens[:, j]
        mask = ~np.isnan(col)
        if mask.sum() >= 3:
            sl, ic, *_ = linregress(ages[mask], col[mask])
            nv[j] = [ic, sl]
        else:
            nv[j] = [30.0, -0.08]
    print(f"nvgenerate: fitted {n_pts}-point normative DB (method={method}).")
    return nv, nv  # (newNV_r, newNV_py) — same object in pure-Python mode


# ---------------------------------------------------------------------------
# Part V: Analysis — GLR / PLR / PoPLR
# ---------------------------------------------------------------------------

def _parse_years(df):
    """Extract years-from-baseline robustly from a date column."""
    if 'date' not in df.columns:
        return np.arange(len(df), dtype=float)

    raw = df['date'].reset_index(drop=True)

    # Case 1: already datetime (e.g. loaded by _load_bundled)
    if pd.api.types.is_datetime64_any_dtype(raw):
        t0 = raw.iloc[0]
        return np.array([(d - t0).days / 365.25 for d in raw])

    # Case 2: string date (e.g. "2007-01-15")
    dates = pd.to_datetime(raw, errors='coerce')
    if not dates.isna().all():
        t0 = dates.iloc[0]
        return np.array([(d - t0).days / 365.25 for d in dates])

    # Case 3: R integer days since 1970-01-01
    try:
        int_vals = raw.astype(float).astype(int)
        if int_vals.between(0, 30000).all():  # sanity: 1970–2052
            dates = pd.to_datetime(int_vals, unit='D', origin='1970-01-01')
            t0 = dates.iloc[0]
            return np.array([(d - t0).days / 365.25 for d in dates])
    except Exception:
        pass

    return np.arange(len(df), dtype=float)


def glr(df_gi_py, type='md', testSlope=0):
    """Global linear regression on a global index series."""
    from scipy.stats import linregress

    years = _parse_years(df_gi_py)

    # support both 'md' and 'tmd' column names (R package uses 'tmd')
    col_map = {'ms': 'ms', 'md': 'md', 'tmd': 'tmd',
               'pmd': 'pmd', 'psd': 'psd', 'vfi': 'vfi', 'gh': 'gh'}
    col = col_map.get(type, type)
    # fallback: tmd → md, psd → psd
    if col not in df_gi_py.columns:
        col = {'tmd': 'md', 'pmd': 'pmd'}.get(col, col)
    data = df_gi_py[col].values.astype(float)

    mask = ~np.isnan(data) & ~np.isnan(years)
    sl, intercept, r, p, se = linregress(years[mask], data[mask])
    tval = (sl - testSlope) / (se + 1e-12)
    pred = sl * years + intercept

    return {
        'type': type, 'testSlope': testSlope,
        'nvisits': len(df_gi_py),
        'years': years,
        'data': data,
        'pred': pred,
        'sl': sl,
        'int': intercept,
        'r2': r ** 2,
        'tval': tval,
        'pval': p,
        'se': se,
    }


def plr(df_vf_py, type='td', testSlope=0):
    """Pointwise linear regression — one slope per test location."""
    scols = _get_sensitivity_cols(df_vf_py)
    years = _parse_years(df_vf_py)

    # Build per-row data array
    raw = df_vf_py[scols].values.astype(float)
    age_arr = df_vf_py['age'].values.astype(float) if 'age' in df_vf_py.columns else np.full(len(df_vf_py), 60.)

    if type in ('td', 'tds'):
        data = np.vstack([vf_core.total_deviation(raw[i], age_arr[i])
                          for i in range(len(raw))])
    elif type in ('pd', 'pds'):
        data = np.vstack([vf_core.pattern_deviation(vf_core.total_deviation(raw[i], age_arr[i]))
                          for i in range(len(raw))])
    else:
        data = raw  # 's': raw sensitivity

    n_pts = data.shape[1]
    from scipy.stats import linregress
    sl, intercept, tval, pval, se_arr = (np.zeros(n_pts) for _ in range(5))
    for j in range(n_pts):
        col = data[:, j]
        mask = ~np.isnan(col)
        if mask.sum() >= 3:
            s, ic, r, p, se = linregress(years[mask], col[mask])
            sl[j] = s; intercept[j] = ic
            tval[j] = (s - testSlope) / (se + 1e-12)
            # Left-tailed p-value (testing slope < testSlope)
            from scipy.stats import t as t_dist
            pval[j] = t_dist.cdf(tval[j], df=mask.sum() - 2)
            se_arr[j] = se
    pred = np.outer(years, sl) + intercept

    return {
        'type': type, 'testSlope': testSlope,
        'nvisits': len(df_vf_py), 'years': years,
        'data': data, 'pred': pred,
        'sl': sl, 'int': intercept,
        'tval': tval, 'pval': pval,
        'se': se_arr,
    }


def poplr(df_vf_py, type='td', testSlope=0, nperm='default', trunc=1):
    """
    PoPLR: permutation-based combined significance of PLR slopes.
    O'Leary et al., IOVS 2012.
    """
    base = plr(df_vf_py, type=type, testSlope=testSlope)
    pvals = base['pval']
    years = base['years']
    data  = base['data']
    n_vis, n_pts = data.shape

    if nperm == 'default':
        import math
        nperm = min(math.factorial(n_vis), 5000)

    from scipy.stats import linregress
    rng = np.random.default_rng(42)

    def _combined_s(pv, trunc):
        """Modified Fisher statistic with truncation."""
        clipped = np.clip(pv, 1e-9, trunc)
        return -2 * np.sum(np.log(clipped))

    obs_pvals = np.zeros(n_pts)
    for j in range(n_pts):
        col = data[:, j]; mask = ~np.isnan(col)
        if mask.sum() >= 3:
            _, _, _, p, _ = linregress(years[mask], col[mask])
            obs_pvals[j] = p

    csl_obs = _combined_s(obs_pvals, trunc)

    perm_csl = np.zeros(nperm)
    for k in range(nperm):
        idx = rng.permutation(n_vis)
        yp = years[idx]
        pv_perm = np.zeros(n_pts)
        for j in range(n_pts):
            col = data[:, j]; mask = ~np.isnan(col)
            if mask.sum() >= 3:
                _, _, _, p, _ = linregress(yp[mask], col[mask])
                pv_perm[j] = p
        perm_csl[k] = _combined_s(pv_perm, trunc)

    cslp = float(np.mean(perm_csl >= csl_obs))

    return {**base,
            'csl': csl_obs, 'cslp': cslp,
            'csr': np.nan,  'csrp': np.nan,  # right-tailed not implemented
            'nperm': nperm}