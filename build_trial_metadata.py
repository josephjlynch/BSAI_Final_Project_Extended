"""
Trial-Type Alignment with Full Granularity (Week 4)
=====================================================

Builds per-session trial metadata tables with:
  - Trial-type labels (omission, passive, change, pre_change_repeat)
  - Repeat position (1-4, 5plus) using SDK flashes_since_change or manual counter
  - Transition type (familiar_to_familiar, familiar_to_novel, novel_to_familiar,
    novel_to_novel, unknown_transition)
  - SDT outcome and reaction time for change trials
  - Engagement flag (boolean, NOT a row filter)
  - Behavioral history tags: previous_sdt, after_hit, after_miss,
    n_consecutive_hits, outcome_lag_1 through outcome_lag_10

Outputs:
  results/derivatives/trial_tables/{session_id}.parquet  (one per session)
  results/derivatives/spike_times/{session_id}_stim_fr.npz  (companion FR files)
  results/tables/trial_metadata_summary.csv

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python build_trial_metadata.py
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    ENGAGEMENT_REWARD_RATE_MIN,
    MAX_LABELED_REPEAT_POSITION,
    GLM_HISTORY_LAGS,
    RESET_INDEX_MIN_TRIALS,
    CCG_MIN_FIRING_RATE_HZ,
    STIMULUS_DURATION_MS,
)
from src.data_loading import load_cache, label_sdt_category

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SESSION_IDS_FILE = 'session_ids.txt'
CACHE_BASE = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0'
    '/behavior_ecephys_sessions'
)
TRIAL_TABLES_DIR = 'results/derivatives/trial_tables'
SPIKE_TIMES_DIR = 'results/derivatives/spike_times'
SUMMARY_CSV = 'results/tables/trial_metadata_summary.csv'


def is_session_cached(sid: int) -> bool:
    nwb = os.path.join(CACHE_BASE, str(sid), f'ecephys_session_{sid}.nwb')
    return os.path.exists(nwb)


# ===================================================================
# Step 2A: Load stimulus_presentations + trials, compute engagement
# ===================================================================

def load_and_tag_engagement(session) -> tuple:
    """Load stimulus_presentations with engagement flag and merged trial info.

    Returns (stim_df, trials_df, join_method) where stim_df has the
    ``engaged`` boolean column and trial-level columns joined in.
    """
    stim = session.stimulus_presentations.copy()
    trials = label_sdt_category(session.trials.copy())

    # -- Compute engagement flag (replicate filter_engaged_trials logic
    #    but retain ALL rows) --
    try:
        reward_rate = session.get_reward_rate()
    except Exception:
        reward_rate = None

    if reward_rate is not None:
        if isinstance(reward_rate, pd.Series):
            if 'trials_id' in stim.columns:
                stim['reward_rate'] = stim['trials_id'].map(reward_rate)
            else:
                stim['reward_rate'] = reward_rate.reindex(
                    stim.index, method='ffill'
                )
        else:
            rr = np.asarray(reward_rate)
            if len(rr) == len(stim):
                stim['reward_rate'] = rr
            elif 'trials_id' in stim.columns and len(rr) == len(trials):
                rr_series = pd.Series(rr, index=trials.index)
                stim['reward_rate'] = stim['trials_id'].map(rr_series)
            else:
                stim['reward_rate'] = np.nan
        stim['engaged'] = stim['reward_rate'] >= ENGAGEMENT_REWARD_RATE_MIN
    else:
        stim['reward_rate'] = np.nan
        stim['engaged'] = False

    # -- Join trials onto stimulus_presentations --
    join_method = 'none'
    trial_cols = ['sdt_category']
    for col in ['reaction_time', 'response_time', 'change_time_no_display_delay']:
        if col in trials.columns:
            trial_cols.append(col)

    if 'trials_id' in stim.columns:
        join_method = 'trials_id'
        trial_subset = trials[trial_cols].copy()
        trial_subset.index.name = 'trials_id_idx'
        stim = stim.merge(
            trial_subset, left_on='trials_id', right_index=True, how='left'
        )
    elif 'change_time_no_display_delay' in trials.columns:
        join_method = 'temporal'
        stim = _temporal_join(stim, trials, trial_cols)
    else:
        for col in trial_cols:
            stim[col] = np.nan

    return stim, trials, join_method


def _temporal_join(stim, trials, trial_cols):
    """Fallback: nearest-time join when trials_id is absent."""
    trial_times = trials['change_time_no_display_delay'].dropna().sort_values()
    trial_subset = trials.loc[trial_times.index, trial_cols]

    stim_sorted = stim.sort_values('start_time')
    stim_times = stim_sorted['start_time'].values

    trial_t = trial_times.values
    indices = np.searchsorted(trial_t, stim_times, side='right') - 1
    indices = np.clip(indices, 0, len(trial_t) - 1)

    for col in trial_cols:
        vals = trial_subset[col].values
        matched = vals[indices]
        time_diff = np.abs(stim_times - trial_t[indices])
        matched[time_diff > 2.0] = np.nan
        stim_sorted[col] = matched

    stim = stim_sorted.reindex(stim.index)
    return stim


# ===================================================================
# Step 2B: Assign trial_type
# ===================================================================

def assign_trial_type(stim: pd.DataFrame) -> pd.DataFrame:
    """Assign mutually exclusive trial_type with priority ordering."""
    stim['trial_type'] = 'pre_change_repeat'

    is_omitted = stim.get('omitted', pd.Series(False, index=stim.index))
    if is_omitted.dtype == object:
        is_omitted = is_omitted.fillna(False).astype(bool)
    else:
        is_omitted = is_omitted.fillna(False)

    is_active = stim.get('active', pd.Series(True, index=stim.index))
    if is_active.dtype == object:
        is_active = is_active.fillna(True).astype(bool)
    else:
        is_active = is_active.fillna(True)

    is_change = stim.get('is_change', pd.Series(False, index=stim.index))
    if is_change.dtype == object:
        is_change = is_change.fillna(False).astype(bool)
    else:
        is_change = is_change.fillna(False)

    stim.loc[is_omitted, 'trial_type'] = 'omission'
    stim.loc[~is_active & ~is_omitted, 'trial_type'] = 'passive'
    stim.loc[is_change & is_active & ~is_omitted, 'trial_type'] = 'change'

    return stim


# ===================================================================
# Step 2C: Assign repeat_position
# ===================================================================

def assign_repeat_position(stim: pd.DataFrame) -> pd.DataFrame:
    """Assign repeat_position for pre_change_repeat rows."""
    stim['repeat_position'] = np.nan

    if 'flashes_since_change' in stim.columns:
        fsc = stim['flashes_since_change']
        mask = stim['trial_type'] == 'pre_change_repeat'
        pos = fsc.copy()
        pos[pos > MAX_LABELED_REPEAT_POSITION] = -1
        stim.loc[mask & (pos >= 1) & (pos <= MAX_LABELED_REPEAT_POSITION),
                 'repeat_position'] = pos
        stim.loc[mask & (pos == -1), 'repeat_position'] = -1
        stim['repeat_position'] = stim['repeat_position'].replace(
            {-1: '5plus'}
        )
        pos_nums = stim.loc[
            mask & stim['repeat_position'].notna()
            & (stim['repeat_position'] != '5plus'), 'repeat_position'
        ]
        if len(pos_nums) > 0:
            stim.loc[pos_nums.index, 'repeat_position'] = (
                pos_nums.astype(int).astype(str)
            )
    else:
        _assign_repeat_position_manual(stim)

    return stim


def _assign_repeat_position_manual(stim: pd.DataFrame) -> None:
    """Manual repeat position counter when SDK column is absent."""
    is_active = stim.get('active', pd.Series(True, index=stim.index)).fillna(True)
    is_omitted = stim.get('omitted', pd.Series(False, index=stim.index)).fillna(False)
    is_change = stim.get('is_change', pd.Series(False, index=stim.index)).fillna(False)

    counter = 0
    for idx in stim.index:
        if not is_active[idx] or is_omitted[idx]:
            continue
        if is_change[idx]:
            counter = 0
        else:
            counter += 1
            if stim.at[idx, 'trial_type'] == 'pre_change_repeat':
                if counter <= MAX_LABELED_REPEAT_POSITION:
                    stim.at[idx, 'repeat_position'] = str(counter)
                else:
                    stim.at[idx, 'repeat_position'] = '5plus'


# ===================================================================
# Step 2D: Assign transition_type
# ===================================================================

def assign_transition_type(
    stim: pd.DataFrame, experience_level: str, session
) -> pd.DataFrame:
    """Assign transition_type for change trials."""
    stim['transition_type'] = np.nan
    change_mask = stim['trial_type'] == 'change'

    if not change_mask.any():
        return stim

    is_familiar_session = 'familiar' in str(experience_level).lower()

    if is_familiar_session:
        stim.loc[change_mask, 'transition_type'] = 'familiar_to_familiar'
        return stim

    familiar_images = _get_familiar_images(session, stim)

    if familiar_images is None:
        stim.loc[change_mask, 'transition_type'] = 'unknown_transition'
        return stim

    image_names = stim['image_name']
    prev_image = image_names.shift(1)

    for idx in stim.index[change_mask]:
        post_img = image_names[idx]
        pre_img = prev_image[idx]

        if pd.isna(pre_img) or pd.isna(post_img):
            stim.at[idx, 'transition_type'] = 'unknown_transition'
            continue
        if str(pre_img) == 'omitted' or str(post_img) == 'omitted':
            stim.at[idx, 'transition_type'] = 'unknown_transition'
            continue

        pre_is_fam = str(pre_img) in familiar_images
        post_is_fam = str(post_img) in familiar_images

        if pre_is_fam and post_is_fam:
            stim.at[idx, 'transition_type'] = 'familiar_to_familiar'
        elif pre_is_fam and not post_is_fam:
            stim.at[idx, 'transition_type'] = 'familiar_to_novel'
        elif not pre_is_fam and post_is_fam:
            stim.at[idx, 'transition_type'] = 'novel_to_familiar'
        else:
            stim.at[idx, 'transition_type'] = 'novel_to_novel'

    return stim


def _get_familiar_images(session, stim: pd.DataFrame):
    """Determine which image_names are from the familiar set.

    Returns a set of image name strings, or None if unresolvable.
    """
    try:
        metadata = session.metadata
        if isinstance(metadata, dict):
            img_set = metadata.get('image_set', None)
        elif hasattr(metadata, 'image_set'):
            img_set = metadata.image_set
        else:
            img_set = None
    except Exception:
        img_set = None

    try:
        if hasattr(session, 'task_parameters'):
            tp = session.task_parameters
            if isinstance(tp, dict) and 'image_set' in tp:
                img_set = tp['image_set']
    except Exception:
        pass

    unique_images = set(
        str(x) for x in stim['image_name'].unique()
        if str(x) not in ('omitted', 'nan', '')
    )

    if len(unique_images) <= 8:
        return unique_images

    try:
        session_table = None
        if hasattr(session, '_cache'):
            session_table = session._cache.get_ecephys_session_table()
        if session_table is not None:
            mouse_id = None
            if hasattr(session, 'metadata') and isinstance(session.metadata, dict):
                mouse_id = session.metadata.get('mouse_id', None)
            if mouse_id is not None:
                fam_sessions = session_table[
                    (session_table['mouse_id'] == mouse_id)
                    & session_table['experience_level'].str.contains(
                        'Familiar', case=False, na=False
                    )
                ]
                if len(fam_sessions) > 0:
                    pass
    except Exception:
        pass

    if len(unique_images) > 8:
        return None

    return unique_images


# ===================================================================
# Step 2E: Assign behavioral outcome
# ===================================================================

def assign_outcome(stim: pd.DataFrame) -> pd.DataFrame:
    """Map sdt_category to outcome for change trials; add reaction_time_s."""
    stim['outcome'] = np.nan
    change_mask = stim['trial_type'] == 'change'

    if 'sdt_category' in stim.columns:
        valid_cats = {'hit', 'miss', 'false_alarm', 'correct_reject'}
        cat_mask = change_mask & stim['sdt_category'].isin(valid_cats)
        stim.loc[cat_mask, 'outcome'] = stim.loc[cat_mask, 'sdt_category']

    stim['reaction_time_s'] = np.nan
    if 'reaction_time' in stim.columns:
        hit_mask = change_mask & (stim.get('outcome', '') == 'hit')
        stim.loc[hit_mask, 'reaction_time_s'] = stim.loc[
            hit_mask, 'reaction_time'
        ]
    elif 'response_time' in stim.columns and 'change_time_no_display_delay' in stim.columns:
        hit_mask = change_mask & (stim.get('outcome', '') == 'hit')
        stim.loc[hit_mask, 'reaction_time_s'] = (
            stim.loc[hit_mask, 'response_time']
            - stim.loc[hit_mask, 'change_time_no_display_delay']
        )

    return stim


# ===================================================================
# Step 2F: Behavioral history tags
# ===================================================================

def assign_history_tags(stim: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Tag each active, non-omitted presentation with behavioral history.

    Lags count trial-level events (rows in session.trials), NOT stimulus
    presentations.  Encoding: hit=+1, miss=-1, fa/cr/none=0.
    """
    n = len(stim)
    stim['previous_sdt'] = 'none'
    stim['after_hit'] = False
    stim['after_miss'] = False
    stim['n_consecutive_hits'] = 0
    for lag in range(1, GLM_HISTORY_LAGS + 1):
        stim[f'outcome_lag_{lag}'] = 0

    if 'sdt_category' not in trials.columns:
        return stim

    valid_mask = trials['sdt_category'].isin(
        {'hit', 'miss', 'false_alarm', 'correct_reject'}
    )
    valid_trials = trials[valid_mask].copy()
    if len(valid_trials) == 0:
        return stim

    if 'change_time_no_display_delay' in valid_trials.columns:
        trial_times = valid_trials['change_time_no_display_delay'].values
    elif 'start_time' in valid_trials.columns:
        trial_times = valid_trials['start_time'].values
    else:
        return stim

    trial_cats = valid_trials['sdt_category'].values
    trial_encoded = np.where(
        trial_cats == 'hit', 1.0,
        np.where(trial_cats == 'miss', -1.0, 0.0)
    )

    go_trial_mask = np.isin(trial_cats, ['hit', 'miss'])
    go_indices = np.where(go_trial_mask)[0]

    active_mask = (
        (stim.get('active', pd.Series(True, index=stim.index)).fillna(True))
        & (~stim.get('omitted', pd.Series(False, index=stim.index)).fillna(False))
    )

    stim_times = stim['start_time'].values
    trial_insert_idx = np.searchsorted(trial_times, stim_times, side='right')

    for i in range(n):
        if not active_mask.iloc[i]:
            stim.iat[i, stim.columns.get_loc('previous_sdt')] = np.nan
            stim.iat[i, stim.columns.get_loc('after_hit')] = np.nan
            stim.iat[i, stim.columns.get_loc('after_miss')] = np.nan
            stim.iat[i, stim.columns.get_loc('n_consecutive_hits')] = np.nan
            for lag in range(1, GLM_HISTORY_LAGS + 1):
                stim.iat[i, stim.columns.get_loc(f'outcome_lag_{lag}')] = np.nan
            continue

        t_idx = trial_insert_idx[i]

        if t_idx > 0:
            stim.iat[i, stim.columns.get_loc('previous_sdt')] = trial_cats[t_idx - 1]

        relevant_go = go_indices[go_indices < t_idx]
        if len(relevant_go) > 0:
            last_go = relevant_go[-1]
            last_go_cat = trial_cats[last_go]
            stim.iat[i, stim.columns.get_loc('after_hit')] = (last_go_cat == 'hit')
            stim.iat[i, stim.columns.get_loc('after_miss')] = (last_go_cat == 'miss')

            consec = 0
            for j in range(len(relevant_go) - 1, -1, -1):
                if trial_cats[relevant_go[j]] == 'hit':
                    consec += 1
                else:
                    break
            stim.iat[i, stim.columns.get_loc('n_consecutive_hits')] = consec

        for lag in range(1, GLM_HISTORY_LAGS + 1):
            tidx = t_idx - lag
            if 0 <= tidx < len(trial_encoded):
                stim.iat[i, stim.columns.get_loc(f'outcome_lag_{lag}')] = trial_encoded[tidx]

    return stim


# ===================================================================
# Build trial table for one session
# ===================================================================

def build_session_trial_table(
    session, session_id: int, session_meta_row
) -> tuple:
    """Build the full trial metadata DataFrame for one session.

    Returns (df, diagnostics_dict).
    """
    mouse_id = int(session_meta_row.get('mouse_id', -1)) \
        if 'mouse_id' in session_meta_row.index else -1
    experience_level = str(session_meta_row.get('experience_level', 'unknown')) \
        if 'experience_level' in session_meta_row.index else 'unknown'

    stim, trials, join_method = load_and_tag_engagement(session)
    stim = assign_trial_type(stim)
    stim = assign_repeat_position(stim)
    stim = assign_transition_type(stim, experience_level, session)
    stim = assign_outcome(stim)
    stim = assign_history_tags(stim, trials)

    has_fsc = 'flashes_since_change' in session.stimulus_presentations.columns

    out = pd.DataFrame({
        'session_id': session_id,
        'mouse_id': mouse_id,
        'experience_level': experience_level,
        'stimulus_idx': range(len(stim)),
        'start_time_s': stim['start_time'].values,
        'image_name': stim['image_name'].values if 'image_name' in stim.columns else '',
        'active': stim['active'].values if 'active' in stim.columns else True,
        'engaged': stim['engaged'].values,
        'trial_type': stim['trial_type'].values,
        'repeat_position': stim['repeat_position'].values,
        'transition_type': stim['transition_type'].values,
        'outcome': stim['outcome'].values,
        'reaction_time_s': stim['reaction_time_s'].values,
        'after_hit': stim['after_hit'].values,
        'after_miss': stim['after_miss'].values,
        'n_consecutive_hits': stim['n_consecutive_hits'].values,
    })
    for lag in range(1, GLM_HISTORY_LAGS + 1):
        out[f'outcome_lag_{lag}'] = stim[f'outcome_lag_{lag}'].values

    diag = {
        'session_id': session_id,
        'mouse_id': mouse_id,
        'experience_level': experience_level,
        'n_stimuli': len(out),
        'n_change': int((out['trial_type'] == 'change').sum()),
        'n_pre_change': int((out['trial_type'] == 'pre_change_repeat').sum()),
        'n_omission': int((out['trial_type'] == 'omission').sum()),
        'n_passive': int((out['trial_type'] == 'passive').sum()),
        'n_unknown_transition': int(
            (out['transition_type'] == 'unknown_transition').sum()
        ),
        'join_method': join_method,
        'has_flashes_since_change': has_fsc,
    }

    return out, diag


# ===================================================================
# Task 2.5: Stimulus-epoch firing rate refinement
# ===================================================================

def compute_stim_epoch_firing_rates(session_id: int, trial_df: pd.DataFrame):
    """Compute per-unit firing rates during active, non-omitted stimuli.

    Saves a companion {session_id}_stim_fr.npz alongside the original NPZ.
    Returns a dict with summary stats, or None if NPZ is missing.
    """
    npz_path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}.npz')
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)
    unit_ids = data['unit_ids']
    spike_times_arr = data['spike_times']
    n_units = len(unit_ids)

    active_stim = trial_df[
        (trial_df['trial_type'].isin(['change', 'pre_change_repeat']))
        & (trial_df['active'] == True)
    ]
    if len(active_stim) == 0:
        return None

    stim_starts = active_stim['start_time_s'].values
    stim_dur_s = STIMULUS_DURATION_MS / 1000.0
    stim_ends = stim_starts + stim_dur_s
    total_stim_time = len(stim_starts) * stim_dur_s

    rates = np.zeros(n_units, dtype=np.float64)
    for i in range(n_units):
        st = spike_times_arr[i]
        if len(st) == 0:
            continue
        count = 0
        for s, e in zip(stim_starts, stim_ends):
            count += int(np.sum((st >= s) & (st < e)))
        rates[i] = count / total_stim_time if total_stim_time > 0 else 0.0

    ccg_eligible_stim = (rates >= CCG_MIN_FIRING_RATE_HZ).astype(bool)

    old_eligible = data['ccg_eligible'].astype(bool)
    gained = int(np.sum(ccg_eligible_stim & ~old_eligible))
    lost = int(np.sum(~ccg_eligible_stim & old_eligible))

    out_path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}_stim_fr.npz')
    np.savez_compressed(
        out_path,
        unit_ids=unit_ids,
        firing_rates_stim_hz=rates,
        ccg_eligible_stim=ccg_eligible_stim,
    )

    return {
        'session_id': session_id,
        'n_units': n_units,
        'n_stim_presentations': len(stim_starts),
        'total_stim_time_s': round(total_stim_time, 2),
        'n_ccg_session_wide': int(old_eligible.sum()),
        'n_ccg_stim_epoch': int(ccg_eligible_stim.sum()),
        'n_gained': gained,
        'n_lost': lost,
    }


# ===================================================================
# Task 3: Summary table
# ===================================================================

def build_summary_table():
    """Read all per-session Parquet files and produce the summary CSV."""
    parquet_files = sorted([
        f for f in os.listdir(TRIAL_TABLES_DIR) if f.endswith('.parquet')
    ])
    if not parquet_files:
        print("WARNING: No parquet files found; skipping summary table.")
        return None

    rows = []
    for fname in parquet_files:
        df = pd.read_parquet(os.path.join(TRIAL_TABLES_DIR, fname))
        sid = df['session_id'].iloc[0]
        mid = df['mouse_id'].iloc[0]
        exp = df['experience_level'].iloc[0]

        change = df[df['trial_type'] == 'change']
        pre = df[df['trial_type'] == 'pre_change_repeat']

        n_hit = int((change['outcome'] == 'hit').sum())
        n_miss = int((change['outcome'] == 'miss').sum())
        n_fa = int((change['outcome'] == 'false_alarm').sum())
        n_cr = int((change['outcome'] == 'correct_reject').sum())

        pos_counts = {}
        for p in range(1, MAX_LABELED_REPEAT_POSITION + 1):
            pos_counts[f'n_pre_change_pos{p}'] = int(
                (pre['repeat_position'] == str(p)).sum()
            )
        pos_counts['n_pre_change_pos5plus'] = int(
            (pre['repeat_position'] == '5plus').sum()
        )

        n_after_hit = int(
            change[change['after_hit'] == True].shape[0]
        )
        n_after_miss = int(
            change[change['after_miss'] == True].shape[0]
        )
        ri_eligible = (
            n_after_hit >= RESET_INDEX_MIN_TRIALS
            and n_after_miss >= RESET_INDEX_MIN_TRIALS
        )

        row = {
            'session_id': sid,
            'mouse_id': mid,
            'experience_level': exp,
            'n_change_hit': n_hit,
            'n_change_miss': n_miss,
            'n_change_fa': n_fa,
            'n_change_cr': n_cr,
            **pos_counts,
            'n_omission': int((df['trial_type'] == 'omission').sum()),
            'n_passive': int((df['trial_type'] == 'passive').sum()),
            'n_after_hit_change': n_after_hit,
            'n_after_miss_change': n_after_miss,
            'ri_eligible': ri_eligible,
            'n_unknown_transition': int(
                (df['transition_type'] == 'unknown_transition').sum()
            ),
            'n_total_stimuli': len(df),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


# ===================================================================
# Main
# ===================================================================

def main():
    t_global = time.perf_counter()

    print('=' * 70)
    print('TRIAL-TYPE ALIGNMENT WITH FULL GRANULARITY (Week 4)')
    print('=' * 70)
    print(f'  MAX_LABELED_REPEAT_POSITION = {MAX_LABELED_REPEAT_POSITION}')
    print(f'  GLM_HISTORY_LAGS            = {GLM_HISTORY_LAGS}')
    print(f'  RESET_INDEX_MIN_TRIALS      = {RESET_INDEX_MIN_TRIALS}')
    print(f'  ENGAGEMENT_REWARD_RATE_MIN  = {ENGAGEMENT_REWARD_RATE_MIN}')
    print(f'  CCG_MIN_FIRING_RATE_HZ      = {CCG_MIN_FIRING_RATE_HZ}')
    print(f'  STIMULUS_DURATION_MS        = {STIMULUS_DURATION_MS}')
    print()

    os.makedirs(TRIAL_TABLES_DIR, exist_ok=True)
    os.makedirs(SPIKE_TIMES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)

    if os.path.exists(SESSION_IDS_FILE):
        with open(SESSION_IDS_FILE) as f:
            all_sids = [
                int(x.strip()) for x in f
                if x.strip() and not x.strip().startswith('#')
            ]
    else:
        print(f'ERROR: {SESSION_IDS_FILE} not found.')
        sys.exit(1)

    cached_sids = [sid for sid in all_sids if is_session_cached(sid)]
    print(f'Sessions in file:     {len(all_sids)}')
    print(f'Cached locally:       {len(cached_sids)}')

    already_done = set()
    for fname in os.listdir(TRIAL_TABLES_DIR):
        if fname.endswith('.parquet'):
            try:
                already_done.add(int(fname.replace('.parquet', '')))
            except ValueError:
                pass

    to_process = [sid for sid in cached_sids if sid not in already_done]
    print(f'Already complete:     {len(already_done)} (skipping)')
    print(f'To process:           {len(to_process)}')
    print()

    print('Loading cache (session_table loaded ONCE)...', flush=True)
    cache = load_cache()
    session_table = cache.get_ecephys_session_table()
    print(f'session_table rows: {len(session_table):,}')
    print()

    # Counters
    total_rows = 0
    type_counts = {'change': 0, 'pre_change_repeat': 0, 'omission': 0, 'passive': 0}
    transition_counts = {}
    pos_counts = {}
    n_processed = 0
    n_failed = 0
    n_with_trials_id = 0
    n_with_fsc = 0
    n_familiar = 0
    n_novel = 0
    mice_seen = set()
    fr_results = []

    for sid in tqdm(to_process, desc='Building trial tables'):
        if sid in session_table.index:
            meta_row = session_table.loc[sid]
        else:
            meta_row = pd.Series({'mouse_id': -1, 'experience_level': 'unknown'})

        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)
            trial_df, diag = build_session_trial_table(session, sid, meta_row)

            out_path = os.path.join(TRIAL_TABLES_DIR, f'{sid}.parquet')
            trial_df.to_parquet(out_path, index=False)

            total_rows += len(trial_df)
            for tt in type_counts:
                type_counts[tt] += int((trial_df['trial_type'] == tt).sum())

            for tt in trial_df['transition_type'].dropna().unique():
                transition_counts[tt] = transition_counts.get(tt, 0) + int(
                    (trial_df['transition_type'] == tt).sum()
                )
            for rp in trial_df['repeat_position'].dropna().unique():
                pos_counts[rp] = pos_counts.get(rp, 0) + int(
                    (trial_df['repeat_position'] == rp).sum()
                )

            if diag['join_method'] == 'trials_id':
                n_with_trials_id += 1
            if diag['has_flashes_since_change']:
                n_with_fsc += 1
            if 'familiar' in str(diag['experience_level']).lower():
                n_familiar += 1
            else:
                n_novel += 1
            mice_seen.add(diag['mouse_id'])

            fr_result = compute_stim_epoch_firing_rates(sid, trial_df)
            if fr_result is not None:
                fr_results.append(fr_result)

            n_processed += 1
            tqdm.write(
                f'  {sid}: {len(trial_df)} rows, '
                f'{diag["n_change"]} change, '
                f'{diag["n_unknown_transition"]} unknown_trans, '
                f'join={diag["join_method"]}'
            )

        except Exception as e:
            n_failed += 1
            tqdm.write(f'  FAILED {sid}: {e}')

    # -- Build summary table from ALL parquet files (including previously done) --
    print('\nBuilding summary table from all parquet files...', flush=True)
    summary_df = build_summary_table()

    elapsed = time.perf_counter() - t_global

    # ===================================================================
    # Task 4: Structured summary
    # ===================================================================
    print(f'\n{"=" * 70}')
    print('TRIAL-TYPE ALIGNMENT SUMMARY')
    print(f'{"=" * 70}')

    print(f'\nSessions processed this run: {n_processed}')
    print(f'Sessions skipped (done):     {len(already_done)}')
    print(f'Sessions failed:             {n_failed}')
    print(f'Total rows written:          {total_rows:,}')

    print(f'\nTrial-type counts (this run):')
    for tt, cnt in type_counts.items():
        print(f'  {tt:25s}: {cnt:,}')

    print(f'\nRepeat position distribution:')
    for p in sorted(pos_counts.keys(), key=lambda x: (x != '5plus', x)):
        print(f'  position {p:6s}: {pos_counts[p]:,}')

    print(f'\nTransition type distribution:')
    for tt in sorted(transition_counts.keys()):
        print(f'  {tt:25s}: {transition_counts[tt]:,}')

    print(f'\nSessions with trials_id:            {n_with_trials_id}/{n_processed}')
    print(f'Sessions with flashes_since_change: {n_with_fsc}/{n_processed}')
    print(f'Familiar / Novel sessions:          {n_familiar} / {n_novel}')
    print(f'Unique mice covered:                {len(mice_seen)}')

    if fr_results:
        total_gained = sum(r['n_gained'] for r in fr_results)
        total_lost = sum(r['n_lost'] for r in fr_results)
        total_units_fr = sum(r['n_units'] for r in fr_results)
        print(f'\nStimulus-epoch firing rate refinement ({len(fr_results)} sessions):')
        print(f'  Total units processed:     {total_units_fr:,}')
        print(f'  Gained CCG eligibility:    {total_gained}')
        print(f'  Lost CCG eligibility:      {total_lost}')
    else:
        print('\nNo stimulus-epoch firing rate files created (no NPZ files found).')

    if summary_df is not None:
        n_ri_eligible = int(summary_df['ri_eligible'].sum())
        print(f'\nSummary table: {SUMMARY_CSV}')
        print(f'  Total sessions in summary: {len(summary_df)}')
        print(f'  RI-eligible sessions:      {n_ri_eligible}')
        n_unknown = int(summary_df['n_unknown_transition'].sum())
        n_total_change = int(
            summary_df[['n_change_hit', 'n_change_miss',
                         'n_change_fa', 'n_change_cr']].sum().sum()
        )
        if n_total_change > 0:
            pct_resolved = (1 - n_unknown / n_total_change) * 100
            print(f'  Transition resolution rate: {pct_resolved:.1f}%')

    print(f'\nElapsed time: {elapsed:.1f}s')
    print(f'Output dir:   {TRIAL_TABLES_DIR}/')
    print(f'Summary CSV:  {SUMMARY_CSV}')


if __name__ == '__main__':
    main()
