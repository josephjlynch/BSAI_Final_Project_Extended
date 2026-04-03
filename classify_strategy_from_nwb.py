"""
SDK-free Behavioral Strategy Classification (Piet et al. 2023)
===============================================================

Classifies each mouse as visual-strategy or timing-strategy using the
dynamic logistic regression of Piet et al. (2023) / Roy et al. (2018),
implemented via the ``psytrack`` package.

**Path A (ACTIVE):** PsyTrack dynamic logistic regression at flash level.
  Each active image presentation is an observation.  y = 1 if a new licking
  bout starts on that flash, 0 otherwise.  Strategy index is the difference
  in model-evidence reduction when the visual vs. timing regressor is
  removed.

**Path B (COMMENTED):** Flash-level static logistic regression using
  scikit-learn.  Same design matrix as Path A but with a single static fit.

Key methodological differences vs. the original trial-level script:
  1. Each image presentation (flash) is an observation, not each trial.
  2. y = lick-bout onset (Piet), not trial response (old script).
  3. Timing = sigmoid(images since last lick bout), not block_position.
  4. Strategy index = model evidence comparison (Piet), not |beta| ratio.
  5. Licking bouts segmented by 700 ms ILI threshold (Piet default).

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python classify_strategy_from_nwb.py
"""

import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    RANDOM_SEED, N_MICE_EXPECTED, STRATEGY_INDEX_THRESHOLD,
    STRATEGY_MIN_TRIALS, ANTICIPATORY_WINDOW_S,
    ENGAGEMENT_REWARD_RATE_MIN,
    LICK_BOUT_ILI_S, TIMING_SIGMOID_MIDPOINT, TIMING_SIGMOID_SLOPE,
    RESPONSE_WINDOW_MS,
)

try:
    from psytrack.hyperOpt import hyperOpt as _psytrack_hyperOpt
    _PSYTRACK_AVAILABLE = True
except ImportError:
    _PSYTRACK_AVAILABLE = False

OUTPUT_CSV = 'results/tables/strategy_classification.csv'
OUTPUT_FIG = 'results/figures/strategy_classification.png'
CACHE_BASE = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0'
    '/behavior_ecephys_sessions'
)
META_DIR = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0/project_metadata'
)
SESSION_IDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'session_ids.txt'
)

FLASH_RESPONSE_WINDOW_S = RESPONSE_WINDOW_MS[1] / 1000.0
ENGAGE_BOUT_WINDOW_S = 10.0
ENGAGE_REWARD_WINDOW_S = 120.0


# ─────────────────────────────────────────────────────────────────────────────
# NWB LOADING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _nwb_path(session_id: int) -> str:
    return os.path.join(CACHE_BASE, str(session_id),
                        f'ecephys_session_{session_id}.nwb')


def _read_dataset(group, key):
    """Read a dataset from h5py Group, handling both Group/data and Dataset."""
    obj = group[key]
    if isinstance(obj, h5py.Dataset):
        return obj[()]
    elif isinstance(obj, h5py.Group) and 'data' in obj:
        return obj['data'][()]
    raise ValueError(f"Cannot read {key}: type={type(obj)}")


def load_trials_from_nwb(nwb_path: str) -> pd.DataFrame:
    """Read trials table directly from NWB with h5py."""
    with h5py.File(nwb_path, 'r') as f:
        trials_grp = f['intervals/trials']

        df = pd.DataFrame({
            'start_time': trials_grp['start_time'][()],
            'stop_time': trials_grp['stop_time'][()],
            'is_change': trials_grp['is_change'][()],
            'hit': trials_grp['hit'][()],
            'miss': trials_grp['miss'][()],
            'false_alarm': trials_grp['false_alarm'][()],
            'correct_reject': trials_grp['correct_reject'][()],
            'aborted': trials_grp['aborted'][()],
            'auto_rewarded': trials_grp['auto_rewarded'][()],
            'change_time_no_display_delay': trials_grp['change_time_no_display_delay'][()],
            'response_time': trials_grp['response_time'][()],
            'reward_volume': trials_grp['reward_volume'][()],
            'go': trials_grp['go'][()],
            'catch': trials_grp['catch'][()],
        })

    df['sdt_category'] = 'auto_reward_or_other'
    for cat in ['correct_reject', 'false_alarm', 'miss', 'hit']:
        df.loc[df[cat] == True, 'sdt_category'] = cat

    return df


def load_stim_presentations_from_nwb(nwb_path: str) -> pd.DataFrame:
    """Read stimulus presentations (Natural_Images) from NWB."""
    with h5py.File(nwb_path, 'r') as f:
        intervals = list(f['intervals'].keys())
        img_keys = [k for k in intervals if 'Natural_Images' in k]
        if not img_keys:
            return pd.DataFrame()
        stim_grp = f[f'intervals/{img_keys[0]}']

        cols = {}
        for key in ['start_time', 'stop_time', 'is_change', 'active',
                     'flashes_since_change']:
            if key in stim_grp:
                cols[key] = _read_dataset(stim_grp, key)

        if 'omitted' in stim_grp:
            cols['omitted'] = _read_dataset(stim_grp, key='omitted')
        else:
            cols['omitted'] = np.zeros(len(cols['start_time']), dtype=bool)

    return pd.DataFrame(cols)


def load_lick_times_from_nwb(nwb_path: str) -> np.ndarray:
    """Read lick timestamps from NWB."""
    with h5py.File(nwb_path, 'r') as f:
        if 'processing/licking/licks/timestamps' not in f:
            return np.array([])
        ts = f['processing/licking/licks/timestamps']
        if isinstance(ts, h5py.Dataset):
            return ts[()]
        elif isinstance(ts, h5py.Group) and 'data' in ts:
            return ts['data'][()]
    return np.array([])


def get_mouse_session_pairs() -> pd.DataFrame:
    """Build familiar/novel session pairs from Bennett's QC-passed sessions.

    Filters the full 153-session manifest to the 103 sessions in
    ``session_ids.txt`` (no abnormal_histology / abnormal_activity flags).
    Bennett's 54 mice are those with both a Familiar and a Novel session
    in this curated set.
    """
    sessions = pd.read_csv(os.path.join(META_DIR, 'ecephys_sessions.csv'))
    sessions = sessions[['ecephys_session_id', 'mouse_id',
                         'experience_level']].copy()

    if os.path.exists(SESSION_IDS_FILE):
        with open(SESSION_IDS_FILE) as f:
            valid_sids = {
                int(x.strip()) for x in f
                if x.strip() and not x.strip().startswith('#')
            }
        sessions = sessions[sessions['ecephys_session_id'].isin(valid_sids)]

    familiar = sessions[
        sessions['experience_level'].str.contains('Familiar', case=False,
                                                  na=False)
    ].groupby('mouse_id')['ecephys_session_id'].first().rename(
        'familiar_session_id')

    novel = sessions[
        sessions['experience_level'].str.contains('Novel', case=False,
                                                  na=False)
    ].groupby('mouse_id')['ecephys_session_id'].first().rename(
        'novel_session_id')

    pairs = familiar.to_frame().join(novel.to_frame(),
                                     how='inner').reset_index()
    if len(pairs) < 40:
        warnings.warn(
            f"get_mouse_session_pairs() returned only {len(pairs)} mice. "
            "Expected ~54 (Bennett). Check ecephys_sessions.csv filtering."
        )
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def segment_licking_bouts(lick_times):
    """Segment lick timestamps into bouts using the 700 ms ILI threshold.

    Parameters
    ----------
    lick_times : array-like

    Returns
    -------
    bout_onsets : np.ndarray  – first lick of each bout
    bout_offsets : np.ndarray – last lick of each bout
    """
    if len(lick_times) == 0:
        return np.array([]), np.array([])
    lt = np.sort(np.asarray(lick_times, dtype=float))
    ilis = np.diff(lt)
    new_bout = np.concatenate([[True], ilis > LICK_BOUT_ILI_S])
    starts = np.where(new_bout)[0]
    bout_onsets = lt[starts]
    ends = np.append(starts[1:] - 1, len(lt) - 1)
    bout_offsets = lt[ends]
    return bout_onsets, bout_offsets


def build_flash_design_matrix(stim_df, lick_times, bout_onsets, bout_offsets,
                               reward_times):
    """Construct the flash-level design matrix for Piet's logistic regression.

    Each active image presentation is an observation.  Returns ``None`` if
    fewer than ``STRATEGY_MIN_TRIALS`` flashes survive engagement and
    mid-bout filters, or if only one response class is present.
    """
    if 'active' in stim_df.columns:
        df = stim_df[stim_df['active'] == True].sort_values(
            'start_time').copy()
    else:
        df = stim_df.sort_values('start_time').copy()

    flash_starts = df['start_time'].values
    n = len(df)
    if n == 0:
        return None

    sorted_licks = (np.sort(np.asarray(lick_times, dtype=float))
                    if len(lick_times) > 0 else np.array([]))

    # ── y: did a NEW lick-bout start on this flash? ──────────────────────
    y = np.zeros(n, dtype=int)
    if len(bout_onsets) > 0:
        idx_arr = np.searchsorted(flash_starts, bout_onsets,
                                  side='right') - 1
        for i, idx in enumerate(idx_arr):
            if 0 <= idx < n:
                if (bout_onsets[i] - flash_starts[idx]) < FLASH_RESPONSE_WINDOW_S:
                    y[idx] = 1

    # ── x_visual: is_change ──────────────────────────────────────────────
    x_visual = (df['is_change'].astype(float).values
                if 'is_change' in df.columns else np.zeros(n))

    # ── x_timing: sigmoid(images since last bout ended) ──────────────────
    reset_set = set()
    if len(bout_offsets) > 0:
        ri = np.searchsorted(flash_starts, bout_offsets, side='right')
        for r in ri:
            if 0 <= r < n:
                reset_set.add(int(r))

    images_since_bout = np.zeros(n)
    counter = 0
    for i in range(n):
        if i in reset_set:
            counter = 0
        images_since_bout[i] = counter
        counter += 1

    x_timing = 1.0 / (1.0 + np.exp(
        -(images_since_bout - TIMING_SIGMOID_MIDPOINT)
        / TIMING_SIGMOID_SLOPE
    ))

    # ── x_omission ──────────────────────────────────────────────────────
    x_omission = (df['omitted'].astype(float).values
                  if 'omitted' in df.columns else np.zeros(n))

    # ── x_post_omission ─────────────────────────────────────────────────
    x_post_omission = np.zeros(n)
    if 'omitted' in df.columns:
        x_post_omission[1:] = df['omitted'].values[:-1].astype(float)

    # ── Mid-bout detection ───────────────────────────────────────────────
    #   A flash is mid-bout if the most recent lick was within ILI AND the
    #   lick sequence continues (next lick also within ILI of that lick).
    mid_bout = np.zeros(n, dtype=bool)
    if len(sorted_licks) > 0:
        prev_pos = np.searchsorted(sorted_licks, flash_starts,
                                   side='left') - 1
        valid = prev_pos >= 0
        clipped = np.clip(prev_pos, 0, len(sorted_licks) - 1)
        time_since = flash_starts - sorted_licks[clipped]
        within_ili = valid & (time_since >= 0) & (
            time_since < LICK_BOUT_ILI_S)

        has_next = (clipped + 1) < len(sorted_licks)
        next_clp = np.clip(clipped + 1, 0, len(sorted_licks) - 1)
        continues = has_next & (
            (sorted_licks[next_clp] - sorted_licks[clipped])
            <= LICK_BOUT_ILI_S
        )
        mid_bout = within_ili & continues

    # ── Engagement filter (Piet: bout in 10 s OR reward in 120 s) ────────
    if len(bout_onsets) > 0:
        sorted_bo = np.sort(bout_onsets)
        r_bo = np.searchsorted(sorted_bo, flash_starts, side='right')
        l_bo = np.searchsorted(sorted_bo,
                               flash_starts - ENGAGE_BOUT_WINDOW_S,
                               side='left')
        recent_bout_n = r_bo - l_bo
    else:
        recent_bout_n = np.zeros(n, dtype=int)

    if len(reward_times) > 0:
        sorted_rw = np.sort(reward_times)
        r_rw = np.searchsorted(sorted_rw, flash_starts, side='right')
        l_rw = np.searchsorted(sorted_rw,
                               flash_starts - ENGAGE_REWARD_WINDOW_S,
                               side='left')
        recent_reward_n = r_rw - l_rw
    else:
        recent_reward_n = np.zeros(n, dtype=int)

    engaged = (recent_bout_n >= 1) | (recent_reward_n >= 1)
    if n > 0:
        grace = flash_starts < (flash_starts[0] + ENGAGE_BOUT_WINDOW_S)
        engaged = engaged | grace

    # ── Apply filters ────────────────────────────────────────────────────
    keep = engaged & ~mid_bout
    if np.sum(keep) < STRATEGY_MIN_TRIALS:
        return None

    y_k = y[keep]
    if len(np.unique(y_k)) < 2:
        return None

    vis_k = x_visual[keep]
    tim_k = x_timing[keep]
    omi_k = x_omission[keep]
    pom_k = x_post_omission[keep]

    vis_k = vis_k - vis_k.mean()
    tim_k = tim_k - tim_k.mean()
    omi_k = omi_k - omi_k.mean()
    pom_k = pom_k - pom_k.mean()

    return {
        'y': y_k,
        'inputs': {
            'visual': vis_k.reshape(-1, 1),
            'timing': tim_k.reshape(-1, 1),
            'omission': omi_k.reshape(-1, 1),
            'post_omission': pom_k.reshape(-1, 1),
        },
        'n_flashes': int(np.sum(keep)),
        'n_bouts': int(y_k.sum()),
        'n_engaged': int(np.sum(engaged)),
        'n_mid_bout_excluded': int(np.sum(mid_bout & engaged)),
        'images_since_bout': images_since_bout[keep],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PATH A — PsyTrack Dynamic Logistic Regression (ACTIVE)
# ─────────────────────────────────────────────────────────────────────────────

def _fit_psytrack(design_matrix, weight_names, hess_calc='weights'):
    """Run a single PsyTrack optimisation.

    Parameters
    ----------
    design_matrix : dict  – must contain 'y' (1-d int array) and 'inputs'
        (dict of (N,1) arrays keyed by regressor name).
    weight_names : list[str]  – which regressors from *inputs* to include
        ('bias' is added automatically by psytrack).
    hess_calc : str or None  – forwarded to ``psytrack.hyperOpt``.

    Returns
    -------
    dict with keys 'hyper', 'logEvd', 'wMode', 'hess_info', 'weight_order',
    'name_to_idx', or ``None`` on failure.
    """
    if not _PSYTRACK_AVAILABLE:
        raise RuntimeError("psytrack is not installed – run "
                           "'pip install psytrack' in your venv")

    weights = {'bias': 1}
    for name in weight_names:
        weights[name] = 1
    K = sum(weights.values())

    dat = {
        'y': design_matrix['y'].copy(),
        'inputs': {k: design_matrix['inputs'][k].copy()
                   for k in weight_names},
    }

    hyper_guess = {
        'sigma': [2**-4] * K,
        'sigInit': [2**4] * K,
    }

    try:
        best_hyper, best_logEvd, best_wMode, hess_info = (
            _psytrack_hyperOpt(
                dat, hyper_guess, weights, ['sigma'],
                hess_calc=hess_calc,
            )
        )
    except Exception as exc:
        warnings.warn(f"PsyTrack fit failed: {exc}")
        return None

    sorted_names = sorted(weights.keys())
    name_to_idx = {n: i for i, n in enumerate(sorted_names)}

    return {
        'hyper': best_hyper,
        'logEvd': float(best_logEvd),
        'wMode': best_wMode,
        'hess_info': hess_info,
        'weight_order': sorted_names,
        'name_to_idx': name_to_idx,
    }


def classify_session_psytrack(stim_df, lick_times, trials):
    """Run Piet-style dynamic strategy classification for one session.

    Returns a dict of per-session results, or ``None`` on failure.
    """
    bout_onsets, bout_offsets = segment_licking_bouts(lick_times)

    reward_mask = (trials['reward_volume'] > 0
                   if 'reward_volume' in trials.columns
                   else pd.Series(False, index=trials.index))
    reward_times = trials.loc[reward_mask, 'start_time'].values

    dm = build_flash_design_matrix(
        stim_df, lick_times, bout_onsets, bout_offsets, reward_times,
    )
    if dm is None:
        return None

    all_names = ['visual', 'timing', 'omission', 'post_omission']

    full_res = _fit_psytrack(dm, all_names, hess_calc='weights')
    if full_res is None:
        return None

    no_vis_res = _fit_psytrack(
        dm, [n for n in all_names if n != 'visual'], hess_calc=None)
    no_tim_res = _fit_psytrack(
        dm, [n for n in all_names if n != 'timing'], hess_calc=None)

    evd_full = full_res['logEvd']
    evd_no_vis = no_vis_res['logEvd'] if no_vis_res else evd_full
    evd_no_tim = no_tim_res['logEvd'] if no_tim_res else evd_full

    reduction_vis = evd_full - evd_no_vis
    reduction_tim = evd_full - evd_no_tim
    strategy_index = reduction_vis - reduction_tim

    nidx = full_res['name_to_idx']
    sigma_arr = np.asarray(full_res['hyper']['sigma'], dtype=float)

    return {
        'strategy_index': float(strategy_index),
        'model_evidence': evd_full,
        'evidence_visual': float(reduction_vis),
        'evidence_timing': float(reduction_tim),
        'sigma_visual': float(sigma_arr[nidx['visual']]),
        'sigma_timing': float(sigma_arr[nidx['timing']]),
        'n_flashes': dm['n_flashes'],
        'n_bouts': dm['n_bouts'],
        'wMode': full_res['wMode'],
        'weight_order': full_res['weight_order'],
        'name_to_idx': nidx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# =======================================================================
# PATH B: Flash-level static logistic regression (pragmatic alternative)
# Uncomment this block and comment Path A if PsyTrack is unavailable
# or if Disheng / Dr. Jia prefer the simpler static approximation.
# This fixes the block_position=0 collinearity bug from the original
# trial-level implementation but does not match Piet's dynamic method.
# =======================================================================
#
# def classify_session_static(stim_df, lick_times, trials):
#     """Flash-level static logistic regression (Path B).
#
#     Same design matrix as Path A but fit with a single static
#     ``LogisticRegression`` instead of PsyTrack's dynamic model.
#
#     strategy_index = |beta_vis| / (|beta_vis| + |beta_tim|)
#     """
#     bout_onsets, bout_offsets = segment_licking_bouts(lick_times)
#
#     reward_mask = (trials['reward_volume'] > 0
#                    if 'reward_volume' in trials.columns
#                    else pd.Series(False, index=trials.index))
#     reward_times = trials.loc[reward_mask, 'start_time'].values
#
#     dm = build_flash_design_matrix(
#         stim_df, lick_times, bout_onsets, bout_offsets, reward_times,
#     )
#     if dm is None:
#         return None
#
#     X = np.column_stack([
#         dm['inputs']['visual'],
#         dm['inputs']['timing'],
#         dm['inputs']['omission'],
#         dm['inputs']['post_omission'],
#     ])
#     y = dm['y']
#
#     scaler = StandardScaler()
#     X_s = scaler.fit_transform(X)
#
#     try:
#         clf = LogisticRegression(
#             penalty=None, max_iter=1000,
#             random_state=RANDOM_SEED, solver='lbfgs',
#         )
#         clf.fit(X_s, y)
#     except Exception:
#         return None
#
#     beta_vis = abs(clf.coef_[0][0])
#     beta_tim = abs(clf.coef_[0][1])
#     denom = beta_vis + beta_tim
#     si = beta_vis / denom if denom > 0 else 0.5
#
#     return {
#         'strategy_index': float(si),
#         'model_evidence': np.nan,
#         'evidence_visual': np.nan,
#         'evidence_timing': np.nan,
#         'sigma_visual': np.nan,
#         'sigma_timing': np.nan,
#         'n_flashes': dm['n_flashes'],
#         'n_bouts': dm['n_bouts'],
#         'wMode': None,
#         'weight_order': None,
#         'name_to_idx': None,
#         'beta_visual_static': float(clf.coef_[0][0]),
#         'beta_timing_static': float(clf.coef_[0][1]),
#     }
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOURAL METRICS (trial-level, independent of strategy method)
# ─────────────────────────────────────────────────────────────────────────────

def compute_reward_rate(trials: pd.DataFrame,
                        window: float = 120.0) -> pd.Series:
    """Approximate engagement via rolling reward rate (rewards / min)."""
    rewarded = trials[trials['reward_volume'] > 0]['start_time'].values
    rates = []
    for _, row in trials.iterrows():
        t = row['start_time']
        n_rew = np.sum((rewarded >= t - window / 2)
                       & (rewarded <= t + window / 2))
        rates.append(n_rew * (60.0 / window))
    return pd.Series(rates, index=trials.index)


def _compute_behavioral_metrics(trials_list, all_lick_times,
                                all_catch_starts):
    """Compute trial-level behavioural metrics across sessions."""
    out = {k: np.nan for k in [
        'n_trials_pooled', 'hit_rate', 'fa_rate', 'dprime',
        'rt_median_s', 'rt_cv', 'anticipatory_lick_frac',
    ]}
    if not trials_list:
        return out

    pooled = pd.concat(trials_list, ignore_index=True)
    out['n_trials_pooled'] = len(pooled)

    n_change = pooled['is_change_flag'].sum()
    n_catch = len(pooled) - n_change
    n_hit = (pooled['sdt_category'] == 'hit').sum()
    n_fa = (pooled['sdt_category'] == 'false_alarm').sum()

    hit_rate = n_hit / max(n_change, 1)
    fa_rate = n_fa / max(n_catch, 1)
    out['hit_rate'] = hit_rate
    out['fa_rate'] = fa_rate

    hr_adj = np.clip(hit_rate, 0.01, 0.99)
    far_adj = np.clip(fa_rate, 0.01, 0.99)
    out['dprime'] = float(stats.norm.ppf(hr_adj) - stats.norm.ppf(far_adj))

    hit_rts = pooled.loc[pooled['sdt_category'] == 'hit', 'rt'].dropna()
    out['rt_median_s'] = (float(hit_rts.median())
                          if len(hit_rts) > 0 else np.nan)
    out['rt_cv'] = (float(hit_rts.std() / hit_rts.mean())
                    if len(hit_rts) > 1 and hit_rts.mean() > 0
                    else np.nan)

    if all_lick_times and all_catch_starts:
        lick_arr = np.asarray(all_lick_times)
        n_ant, n_tot = 0, 0
        for ss in all_catch_starts:
            if np.isnan(ss):
                continue
            n_tot += 1
            w_start = ss - ANTICIPATORY_WINDOW_S
            if np.any((lick_arr >= w_start) & (lick_arr < ss)):
                n_ant += 1
        if n_tot > 0:
            out['anticipatory_lick_frac'] = n_ant / n_tot

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-MOUSE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_single_mouse(mouse_id, familiar_sid, novel_sid):
    """Classify one mouse using PsyTrack (Path A) on each session."""

    row = {
        'mouse_id': mouse_id,
        'familiar_session_id': (int(familiar_sid)
                                if not pd.isna(familiar_sid) else np.nan),
        'novel_session_id': (int(novel_sid)
                             if not pd.isna(novel_sid) else np.nan),
        'classification_method': 'psytrack_dynamic',
    }

    sids_to_load = []
    if not pd.isna(familiar_sid):
        p = _nwb_path(int(familiar_sid))
        if os.path.exists(p):
            sids_to_load.append(('familiar', int(familiar_sid), p))
    if not pd.isna(novel_sid):
        p = _nwb_path(int(novel_sid))
        if os.path.exists(p):
            sids_to_load.append(('novel', int(novel_sid), p))

    row['n_sessions_used'] = len(sids_to_load)
    if not sids_to_load:
        row['strategy_label'] = 'uncached'
        return row

    # ── per-session classification + behavioural metrics collection ───
    session_results = {}
    trials_for_metrics = []
    all_lick_ts = []
    all_catch_starts = []
    example_wMode = None

    for label, sid, nwb_path in sids_to_load:
        try:
            trials = load_trials_from_nwb(nwb_path)
            stim = load_stim_presentations_from_nwb(nwb_path)
            lick_ts = load_lick_times_from_nwb(nwb_path)
        except Exception as e:
            print(f"  WARNING: could not load session {sid}: {e}")
            continue

        # --- trial-level data for behavioural metrics ---
        engaged_trials = trials[trials['sdt_category'].isin(
            ['hit', 'miss', 'false_alarm', 'correct_reject']
        )].copy()
        rr = compute_reward_rate(trials)
        eng_mask = rr.loc[engaged_trials.index] >= ENGAGEMENT_REWARD_RATE_MIN
        engaged_trials = engaged_trials[eng_mask].copy()
        if len(engaged_trials) > 0:
            engaged_trials['responded'] = engaged_trials[
                'sdt_category'].isin(['hit', 'false_alarm']).astype(int)
            engaged_trials['is_change_flag'] = engaged_trials[
                'sdt_category'].isin(['hit', 'miss']).astype(int)
            ct = engaged_trials['change_time_no_display_delay'].values
            engaged_trials['rt'] = (
                engaged_trials['response_time'] - ct
                if not np.all(np.isnan(ct)) else np.nan
            )
            trials_for_metrics.append(engaged_trials)
            if len(lick_ts) > 0:
                all_lick_ts.extend(lick_ts.tolist())
                catch = engaged_trials[engaged_trials['sdt_category'].isin(
                    ['false_alarm', 'correct_reject'])]
                all_catch_starts.extend(
                    catch['start_time'].values.tolist())

        # --- flash-level strategy classification (Path A) ---
        if len(stim) == 0 or len(lick_ts) == 0:
            continue
        sess_res = classify_session_psytrack(stim, lick_ts, trials)
        if sess_res is not None:
            session_results[label] = sess_res
            row[f'n_flashes_{label}'] = sess_res['n_flashes']
            row[f'model_evidence_{label}'] = sess_res['model_evidence']
            row[f'evidence_visual_{label}'] = sess_res['evidence_visual']
            row[f'evidence_timing_{label}'] = sess_res['evidence_timing']
            row[f'sigma_visual_{label}'] = sess_res['sigma_visual']
            row[f'sigma_timing_{label}'] = sess_res['sigma_timing']
            row[f'strategy_index_{label}'] = sess_res['strategy_index']
            if example_wMode is None and sess_res.get('wMode') is not None:
                example_wMode = sess_res['wMode']
                row['_weight_order'] = sess_res['weight_order']
                row['_name_to_idx'] = sess_res['name_to_idx']
                row['_n_flashes_example'] = sess_res['n_flashes']

    # ── behavioural metrics ──────────────────────────────────────────────
    metrics = _compute_behavioral_metrics(
        trials_for_metrics, all_lick_ts, all_catch_starts)
    row.update(metrics)

    # ── mouse-level strategy index and label ─────────────────────────────
    si_vals = [session_results[k]['strategy_index']
               for k in session_results]
    if si_vals:
        avg_si = float(np.mean(si_vals))
        row['strategy_index'] = avg_si
        row['strategy_label'] = 'visual' if avg_si > 0 else 'timing'
    else:
        row['strategy_index'] = np.nan
        row['strategy_label'] = 'unconverged'

    row['strategy_stability_r'] = np.nan
    row['_wMode_example'] = example_wMode
    return row


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(df, output_path, example_row=None):
    """Generate a 4-panel strategy classification figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── Panel 1: strategy index histogram ────────────────────────────────
    ax = axes[0, 0]
    classified = df[df['strategy_label'].isin(['visual', 'timing'])]
    if len(classified) > 0 and 'strategy_index' in classified.columns:
        vals = classified['strategy_index'].dropna()
        ax.hist(vals, bins=20, color='steelblue', edgecolor='white',
                alpha=0.8)
        ax.axvline(0, color='red', ls='--', lw=1.5,
                   label='threshold (0)')
        ax.set_xlabel('Strategy Index (evidence-based)')
        ax.set_ylabel('Count')
        ax.set_title(f'Strategy Index (n={len(classified)} mice)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No classified mice', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Strategy Index')

    # ── Panel 2: label counts ────────────────────────────────────────────
    ax = axes[0, 1]
    labels = df['strategy_label'].value_counts()
    colours = {'visual': '#4a90d9', 'timing': '#d94a4a',
               'uncached': '#999', 'unconverged': '#ccc'}
    bar_c = [colours.get(l, '#888') for l in labels.index]
    ax.bar(labels.index, labels.values, color=bar_c, edgecolor='white')
    ax.set_xlabel('Strategy Label')
    ax.set_ylabel('Count')
    ax.set_title(f'Strategy Labels (N={len(df)} mice)')

    # ── Panel 3: split-session stability ─────────────────────────────────
    ax = axes[1, 0]
    both = df.dropna(subset=['strategy_index_familiar',
                             'strategy_index_novel'])
    if len(both) >= 3:
        ax.scatter(both['strategy_index_familiar'],
                   both['strategy_index_novel'],
                   c='steelblue', s=40, alpha=0.7, edgecolors='white',
                   linewidths=0.5)
        r, p = stats.pearsonr(both['strategy_index_familiar'],
                              both['strategy_index_novel'])
        ax.set_xlabel('Strategy Index (Familiar)')
        ax.set_ylabel('Strategy Index (Novel)')
        ax.set_title(f'Split-session stability  r={r:.3f}, p={p:.4f}')
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', alpha=0.2)
    else:
        ax.text(0.5, 0.5, f'Only {len(both)} mice with both sessions',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Split-session stability')

    # ── Panel 4: example dynamic weight traces ───────────────────────────
    ax = axes[1, 1]
    if (example_row is not None
            and example_row.get('_wMode_example') is not None):
        wMode = example_row['_wMode_example']
        nidx = example_row['_name_to_idx']
        n_flash = wMode.shape[1]
        x_ax = np.arange(n_flash)
        ax.plot(x_ax, wMode[nidx['visual'], :], label='visual',
                color='#4a90d9', lw=1.2)
        ax.plot(x_ax, wMode[nidx['timing'], :], label='timing',
                color='#d94a4a', lw=1.2)
        ax.plot(x_ax, wMode[nidx['bias'], :], label='bias',
                color='#666', lw=0.8, alpha=0.5)
        ax.set_xlabel('Flash number')
        ax.set_ylabel('Weight')
        ax.set_title('Example dynamic weights (1 session)')
        ax.legend(fontsize=8)
        ax.axhline(0, color='k', lw=0.5, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No dynamic weights available', ha='center',
                va='center', transform=ax.transAxes)
        ax.set_title('Example dynamic weights')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.perf_counter()

    print("=" * 70)
    print("BEHAVIORAL STRATEGY CLASSIFICATION")
    print("Method: PsyTrack dynamic logistic regression (Piet et al. 2023)")
    print("=" * 70)

    if not _PSYTRACK_AVAILABLE:
        print("\nERROR: psytrack is not installed.  Run:\n"
              "  .venv/bin/pip install psytrack\n")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)

    pairs = get_mouse_session_pairs()
    print(f"Found {len(pairs)} mice\n")

    CSV_COLUMNS = [
        'mouse_id', 'familiar_session_id', 'novel_session_id',
        'classification_method', 'n_sessions_used',
        'n_flashes_familiar', 'n_flashes_novel',
        'model_evidence_familiar', 'model_evidence_novel',
        'evidence_visual_familiar', 'evidence_visual_novel',
        'evidence_timing_familiar', 'evidence_timing_novel',
        'sigma_visual_familiar', 'sigma_visual_novel',
        'sigma_timing_familiar', 'sigma_timing_novel',
        'strategy_index_familiar', 'strategy_index_novel',
        'strategy_index', 'strategy_label',
        'n_trials_pooled', 'hit_rate', 'fa_rate', 'dprime',
        'rt_median_s', 'rt_cv', 'anticipatory_lick_frac',
        'strategy_stability_r',
    ]

    all_rows = []
    example_row = None

    for i_mouse, (_, pair_row) in enumerate(pairs.iterrows()):
        mouse_id = pair_row['mouse_id']
        familiar_sid = pair_row['familiar_session_id']
        novel_sid = pair_row['novel_session_id']

        print(
            f"[{i_mouse + 1}/{len(pairs)}] Mouse {mouse_id}  "
            f"fam={int(familiar_sid) if not pd.isna(familiar_sid) else 'NA'}  "
            f"nov={int(novel_sid) if not pd.isna(novel_sid) else 'NA'}",
            flush=True,
        )

        result = classify_single_mouse(mouse_id, familiar_sid, novel_sid)

        if (example_row is None
                and result.get('_wMode_example') is not None):
            example_row = result

        all_rows.append(result)

        lbl = result.get('strategy_label', '?')
        si = result.get('strategy_index', np.nan)
        si_str = f"{si:+.3f}" if not np.isnan(si) else "N/A"
        print(f"          -> {lbl}  SI={si_str}")

    df = pd.DataFrame(all_rows)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # ── split-session stability ──────────────────────────────────────────
    both = df.dropna(subset=['strategy_index_familiar',
                             'strategy_index_novel'])
    if len(both) >= 3:
        r, p = stats.pearsonr(both['strategy_index_familiar'],
                              both['strategy_index_novel'])
        df['strategy_stability_r'] = r
        print(f"\nSplit-session stability: r={r:.3f}, p={p:.4f} "
              f"(n={len(both)} mice)")
    else:
        print(f"\nSplit-session stability: insufficient mice "
              f"(n={len(both)})")

    # ── write outputs ────────────────────────────────────────────────────
    df_out = df[CSV_COLUMNS].copy()
    df_out.to_csv(OUTPUT_CSV, index=False)
    make_figure(df_out, OUTPUT_FIG, example_row=example_row)

    elapsed = time.perf_counter() - t_start
    n_vis = (df['strategy_label'] == 'visual').sum()
    n_tim = (df['strategy_label'] == 'timing').sum()
    n_unc = (df['strategy_label'] == 'uncached').sum()
    n_unv = (df['strategy_label'] == 'unconverged').sum()
    n_classified = n_vis + n_tim

    print(f"\n{'=' * 70}")
    print("STRATEGY CLASSIFICATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nTotal mice:       {len(df)}")
    print(f"  Visual:         {n_vis}")
    print(f"  Timing:         {n_tim}")
    print(f"  Uncached:       {n_unc}")
    print(f"  Unconverged:    {n_unv}")
    pct_vis = n_vis / max(n_classified, 1) * 100
    print(f"\n[CHECK] Mouse count: {len(df)} "
          f"(Bennett expected ~{N_MICE_EXPECTED})")
    print(f"[CHECK] Visual / classified: "
          f"{n_vis}/{n_classified} = {pct_vis:.1f}%")
    print(f"\nElapsed time: {elapsed:.1f}s")
    print(f"Output CSV:   {OUTPUT_CSV}")
    print(f"Output Fig:   {OUTPUT_FIG}")


if __name__ == '__main__':
    main()
