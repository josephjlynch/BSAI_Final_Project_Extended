"""
Behavioral Strategy Classification (Week 3)
=============================================

Classifies each of the 54 mice as visual-strategy or timing-strategy
following Piet et al. (2023, bioRxiv):

  "Behavioral strategy shapes activation of the Vip-Sst disinhibitory
   circuit in visual cortex."

Primary method:
    Logistic regression per mouse on ENGAGED trials:
        P(lick) = sigmoid(beta_visual * is_change + beta_timing * block_position)
    Strategy index = |beta_visual| / (|beta_visual| + |beta_timing|)
    Index > 0.5 => "visual"; <= 0.5 => "timing"

Secondary method (commented):
    Composite behavioral metrics + k-means (k=2). Included for
    discussion with Disheng Tang.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python classify_behavioral_strategy.py
"""

import os
import sys
import time
import warnings

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
    ENGAGEMENT_REWARD_RATE_MIN
)
from src.data_loading import (
    load_cache, get_mouse_session_map, get_change_detection_trials
)

OUTPUT_CSV = 'results/tables/strategy_classification.csv'
OUTPUT_FIG = 'results/figures/strategy_classification.png'
SESSION_IDS_FILE = 'session_ids.txt'
CACHE_BASE = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0'
    '/behavior_ecephys_sessions'
)


def is_session_cached(sid):
    nwb = os.path.join(CACHE_BASE, str(sid), f'ecephys_session_{sid}.nwb')
    return os.path.exists(nwb)


def compute_block_positions(stim_df):
    """Walk active, engaged, non-omitted stimulus presentations
    chronologically and assign a block_position (count since last change).

    Returns a Series indexed like stim_df with block_position values.
    """
    active_mask = stim_df['active'] == True if 'active' in stim_df.columns else pd.Series(True, index=stim_df.index)
    omit_mask = stim_df['omitted'] == False if 'omitted' in stim_df.columns else pd.Series(True, index=stim_df.index)
    filtered = stim_df[active_mask & omit_mask].sort_values('start_time')

    positions = pd.Series(np.nan, index=filtered.index, dtype=float)
    counter = 0
    for idx in filtered.index:
        is_change = filtered.at[idx, 'is_change'] if 'is_change' in filtered.columns else False
        if is_change:
            counter = 0
        positions.at[idx] = counter
        counter += 1

    return positions


def classify_single_mouse(cache, mouse_id, familiar_sid, novel_sid):
    """Run strategy classification for a single mouse.

    Returns a dict of metrics, or None on failure.
    """
    row = {
        'mouse_id': mouse_id,
        'familiar_session_id': familiar_sid if not pd.isna(familiar_sid) else np.nan,
        'novel_session_id': novel_sid if not pd.isna(novel_sid) else np.nan,
    }

    sids_to_load = []
    if not pd.isna(familiar_sid) and is_session_cached(int(familiar_sid)):
        sids_to_load.append(('familiar', int(familiar_sid)))
    if not pd.isna(novel_sid) and is_session_cached(int(novel_sid)):
        sids_to_load.append(('novel', int(novel_sid)))

    row['n_sessions_used'] = len(sids_to_load)
    if len(sids_to_load) == 0:
        row['strategy_label'] = 'uncached'
        return row

    all_trials = []
    per_session_trials = {}
    all_lick_times = []
    all_catch_stim_starts = []

    for label, sid in sids_to_load:
        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)
        except Exception as e:
            print(f"  WARNING: could not load session {sid}: {e}")
            continue

        trial_data = get_change_detection_trials(session, engaged_only=True)
        engaged = trial_data['engaged_trials']
        stim = trial_data['stimulus_presentations']

        engaged = engaged[engaged['sdt_category'].isin(
            ['hit', 'miss', 'false_alarm', 'correct_reject']
        )].copy()

        if len(engaged) == 0:
            continue

        engaged['responded'] = engaged['sdt_category'].isin(['hit', 'false_alarm']).astype(int)
        engaged['is_change'] = engaged['sdt_category'].isin(['hit', 'miss']).astype(int)

        block_pos = compute_block_positions(stim)

        if 'change_time_no_display_delay' in engaged.columns:
            change_times = engaged['change_time_no_display_delay'].values
        elif 'change_time' in engaged.columns:
            change_times = engaged['change_time'].values
        else:
            change_times = np.full(len(engaged), np.nan)

        bp_series = pd.Series(np.nan, index=engaged.index)
        bp_stim_starts = block_pos.dropna()
        if len(bp_stim_starts) > 0:
            stim_starts_arr = stim.loc[bp_stim_starts.index, 'start_time'].values
            for t_idx, ct in zip(engaged.index, change_times):
                if np.isnan(ct):
                    continue
                diffs = np.abs(stim_starts_arr - ct)
                nearest = np.argmin(diffs)
                if diffs[nearest] < 1.0:
                    bp_series.at[t_idx] = bp_stim_starts.iloc[nearest]

        engaged['block_position'] = bp_series.values

        if 'response_time' in engaged.columns and not np.all(np.isnan(change_times)):
            engaged['rt'] = engaged['response_time'] - change_times
        else:
            engaged['rt'] = np.nan

        all_trials.append(engaged)
        per_session_trials[label] = engaged

        try:
            lick_df = session.licks
            if lick_df is not None and len(lick_df) > 0:
                lick_times = lick_df['timestamps'].values if 'timestamps' in lick_df.columns else lick_df.index.values
                all_lick_times.extend(lick_times)

                catch_trials = engaged[engaged['sdt_category'].isin(['false_alarm', 'correct_reject'])]
                if 'start_time' in catch_trials.columns:
                    all_catch_stim_starts.extend(catch_trials['start_time'].values)
                elif 'change_time_no_display_delay' in catch_trials.columns:
                    all_catch_stim_starts.extend(catch_trials['change_time_no_display_delay'].values)
        except Exception:
            pass

    if len(all_trials) == 0:
        row['strategy_label'] = 'uncached'
        return row

    pooled = pd.concat(all_trials, ignore_index=True)
    row['n_trials_pooled'] = len(pooled)

    n_change = pooled['is_change'].sum()
    n_catch = len(pooled) - n_change
    n_hit = (pooled['sdt_category'] == 'hit').sum()
    n_miss = (pooled['sdt_category'] == 'miss').sum()
    n_fa = (pooled['sdt_category'] == 'false_alarm').sum()
    n_cr = (pooled['sdt_category'] == 'correct_reject').sum()

    hit_rate = n_hit / max(n_change, 1)
    fa_rate = n_fa / max(n_catch, 1)
    row['hit_rate'] = hit_rate
    row['fa_rate'] = fa_rate

    hr_adj = np.clip(hit_rate, 0.01, 0.99)
    far_adj = np.clip(fa_rate, 0.01, 0.99)
    row['dprime'] = stats.norm.ppf(hr_adj) - stats.norm.ppf(far_adj)

    hit_rts = pooled.loc[pooled['sdt_category'] == 'hit', 'rt'].dropna()
    row['rt_median_s'] = hit_rts.median() if len(hit_rts) > 0 else np.nan
    row['rt_cv'] = (hit_rts.std() / hit_rts.mean()) if len(hit_rts) > 1 and hit_rts.mean() > 0 else np.nan

    if len(all_lick_times) > 0 and len(all_catch_stim_starts) > 0:
        lick_arr = np.array(all_lick_times)
        n_anticipatory = 0
        n_total_catch = len(all_catch_stim_starts)
        for stim_start in all_catch_stim_starts:
            if np.isnan(stim_start):
                n_total_catch -= 1
                continue
            window_start = stim_start - ANTICIPATORY_WINDOW_S
            window_end = stim_start
            if np.any((lick_arr >= window_start) & (lick_arr < window_end)):
                n_anticipatory += 1
        row['anticipatory_lick_frac'] = n_anticipatory / max(n_total_catch, 1)
    else:
        row['anticipatory_lick_frac'] = np.nan

    valid = pooled.dropna(subset=['is_change', 'block_position', 'responded'])
    if len(valid) < STRATEGY_MIN_TRIALS:
        print(f"  WARNING mouse {mouse_id}: only {len(valid)} valid trials "
              f"(min={STRATEGY_MIN_TRIALS}), flagging unconverged")
        row['strategy_label'] = 'unconverged'
        row['beta_visual'] = np.nan
        row['beta_timing'] = np.nan
        row['strategy_index'] = np.nan
        row['strategy_index_familiar'] = np.nan
        row['strategy_index_novel'] = np.nan
        row['strategy_stability_r'] = np.nan
        return row

    X = valid[['is_change', 'block_position']].values.astype(float)
    y = valid['responded'].values.astype(int)

    if len(np.unique(y)) < 2:
        print(f"  WARNING mouse {mouse_id}: only one response class, "
              f"flagging unconverged")
        row['strategy_label'] = 'unconverged'
        row['beta_visual'] = np.nan
        row['beta_timing'] = np.nan
        row['strategy_index'] = np.nan
        row['strategy_index_familiar'] = np.nan
        row['strategy_index_novel'] = np.nan
        row['strategy_stability_r'] = np.nan
        return row

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            clf = LogisticRegression(
                penalty=None, max_iter=1000,
                random_state=RANDOM_SEED, solver='lbfgs'
            )
            clf.fit(X_scaled, y)
        except Exception as e:
            print(f"  WARNING mouse {mouse_id}: LogisticRegression failed: {e}")
            row['strategy_label'] = 'unconverged'
            row['beta_visual'] = np.nan
            row['beta_timing'] = np.nan
            row['strategy_index'] = np.nan
            row['strategy_index_familiar'] = np.nan
            row['strategy_index_novel'] = np.nan
            row['strategy_stability_r'] = np.nan
            return row

        convergence_warn = any(
            issubclass(w.category, UserWarning) and 'converge' in str(w.message).lower()
            for w in caught
        )

    beta_vis = clf.coef_[0][0]
    beta_tim = clf.coef_[0][1]
    row['beta_visual'] = beta_vis
    row['beta_timing'] = beta_tim

    abs_v = abs(beta_vis)
    abs_t = abs(beta_tim)
    denom = abs_v + abs_t
    if denom > 0:
        strategy_index = abs_v / denom
    else:
        strategy_index = 0.5

    row['strategy_index'] = strategy_index

    if convergence_warn:
        row['strategy_label'] = 'unconverged'
    elif strategy_index > STRATEGY_INDEX_THRESHOLD:
        row['strategy_label'] = 'visual'
    else:
        row['strategy_label'] = 'timing'

    row['strategy_index_familiar'] = np.nan
    row['strategy_index_novel'] = np.nan

    for label_key in ['familiar', 'novel']:
        if label_key not in per_session_trials:
            continue
        sess_df = per_session_trials[label_key]
        sess_valid = sess_df.dropna(subset=['is_change', 'block_position', 'responded'])
        if len(sess_valid) < STRATEGY_MIN_TRIALS or len(np.unique(sess_valid['responded'])) < 2:
            continue
        Xs = sess_valid[['is_change', 'block_position']].values.astype(float)
        ys = sess_valid['responded'].values.astype(int)
        Xs_s = StandardScaler().fit_transform(Xs)
        try:
            clf_s = LogisticRegression(
                penalty=None, max_iter=1000,
                random_state=RANDOM_SEED, solver='lbfgs'
            )
            clf_s.fit(Xs_s, ys)
            bv = abs(clf_s.coef_[0][0])
            bt = abs(clf_s.coef_[0][1])
            si = bv / (bv + bt) if (bv + bt) > 0 else 0.5
            row[f'strategy_index_{label_key}'] = si
        except Exception:
            pass

    row['strategy_stability_r'] = np.nan

    return row


def make_figure(df, output_path):
    """2-panel figure: (A) beta scatter, (B) strategy index histogram."""
    classified = df[df['strategy_label'].isin(['visual', 'timing'])].copy()
    unconverged = df[df['strategy_label'] == 'unconverged']
    uncached = df[df['strategy_label'] == 'uncached']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'visual': '#2176AE', 'timing': '#D64933'}
    for label, color in colors.items():
        subset = classified[classified['strategy_label'] == label]
        ax1.scatter(subset['beta_visual'], subset['beta_timing'],
                    c=color, label=f'{label} (n={len(subset)})',
                    s=50, alpha=0.8, edgecolors='white', linewidths=0.5)

    if len(unconverged) > 0:
        ax1.scatter(unconverged['beta_visual'], unconverged['beta_timing'],
                    c='gray', marker='x', label=f'unconverged (n={len(unconverged)})',
                    s=40, alpha=0.6)

    bv_range = classified['beta_visual'].dropna()
    bt_range = classified['beta_timing'].dropna()
    if len(bv_range) > 0 and len(bt_range) > 0:
        lim = max(abs(bv_range).max(), abs(bt_range).max()) * 1.2
        line_pts = np.linspace(-lim, lim, 100)
        ax1.plot(line_pts, line_pts, 'k--', alpha=0.3, linewidth=1)
        ax1.plot(line_pts, -line_pts, 'k--', alpha=0.3, linewidth=1)

    ax1.set_xlabel('beta_visual (standardized)', fontsize=11)
    ax1.set_ylabel('beta_timing (standardized)', fontsize=11)
    ax1.set_title('(A) Logistic Regression Coefficients', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

    si_vals = classified['strategy_index'].dropna()
    if len(si_vals) > 0:
        bins = np.linspace(0, 1, 21)
        visual_si = classified.loc[classified['strategy_label'] == 'visual', 'strategy_index']
        timing_si = classified.loc[classified['strategy_label'] == 'timing', 'strategy_index']

        ax2.hist(timing_si, bins=bins, color=colors['timing'], alpha=0.7,
                 label=f'timing (n={len(timing_si)})', edgecolor='white')
        ax2.hist(visual_si, bins=bins, color=colors['visual'], alpha=0.7,
                 label=f'visual (n={len(visual_si)})', edgecolor='white')

    ax2.axvline(STRATEGY_INDEX_THRESHOLD, color='black', linestyle='--',
                linewidth=1.5, label=f'threshold = {STRATEGY_INDEX_THRESHOLD}')
    ax2.set_xlabel('Strategy Index', fontsize=11)
    ax2.set_ylabel('Number of Mice', fontsize=11)
    ax2.set_title('(B) Strategy Index Distribution', fontsize=12)

    n_vis = (df['strategy_label'] == 'visual').sum()
    n_tim = (df['strategy_label'] == 'timing').sum()
    n_unc = (df['strategy_label'] == 'uncached').sum()
    n_unconv = (df['strategy_label'] == 'unconverged').sum()
    text_lines = [f'N_visual={n_vis}', f'N_timing={n_tim}']
    if n_unc > 0:
        text_lines.append(f'N_uncached={n_unc}')
    if n_unconv > 0:
        text_lines.append(f'N_unconverged={n_unconv}')
    ax2.text(0.02, 0.95, '\n'.join(text_lines), transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {output_path}")


# =========================================================================
# SECONDARY METHOD (commented out for Disheng discussion)
# =========================================================================
# from sklearn.cluster import KMeans
#
# def classify_composite_kmeans(df):
#     """Composite metrics + k-means (k=2) classification.
#
#     Features per mouse:
#       - dprime
#       - fa_rate (catch trial false alarm rate)
#       - rt_cv (response time coefficient of variation)
#       - anticipatory_lick_frac
#
#     Steps:
#       1. Standardize features with StandardScaler
#       2. k-means (k=2, random_state=RANDOM_SEED)
#       3. Label cluster with higher dprime as "visual"
#     """
#     features = ['dprime', 'fa_rate', 'rt_cv', 'anticipatory_lick_frac']
#     valid = df.dropna(subset=features).copy()
#     if len(valid) < 4:
#         print("WARNING: too few mice with complete metrics for k-means")
#         return df
#
#     X = valid[features].values
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     km = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
#     labels = km.fit_predict(X_scaled)
#
#     # Cluster with higher mean dprime => "visual"
#     means = [valid.loc[labels == i, 'dprime'].mean() for i in range(2)]
#     visual_cluster = int(np.argmax(means))
#
#     valid['km_strategy'] = np.where(
#         labels == visual_cluster, 'visual', 'timing'
#     )
#     df = df.merge(
#         valid[['mouse_id', 'km_strategy']],
#         on='mouse_id', how='left'
#     )
#     return df


def main():
    t_start = time.perf_counter()
    print("=" * 70)
    print("BEHAVIORAL STRATEGY CLASSIFICATION (Piet et al. 2023)")
    print("=" * 70)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_FIG), exist_ok=True)

    existing_mice = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        existing_mice = set(existing_df['mouse_id'].values)
        print(f"Checkpoint: {len(existing_mice)} mice already in {OUTPUT_CSV}")

    print("Loading cache...", flush=True)
    cache = load_cache()

    print("Getting mouse-session pairs...", flush=True)
    session_map = get_mouse_session_map(cache)

    familiar = session_map[
        session_map['experience_level'].str.contains('Familiar', case=False, na=False)
    ].groupby('mouse_id')['ecephys_session_id'].first().rename('familiar_session_id')

    novel = session_map[
        session_map['experience_level'].str.contains('Novel', case=False, na=False)
    ].groupby('mouse_id')['ecephys_session_id'].first().rename('novel_session_id')

    pairs = familiar.to_frame().join(novel.to_frame(), how='outer').reset_index()
    print(f"Found {len(pairs)} mice (outer join: includes mice with only 1 session type)")

    CSV_COLUMNS = [
        'mouse_id', 'familiar_session_id', 'novel_session_id',
        'n_sessions_used', 'n_trials_pooled', 'hit_rate', 'fa_rate', 'dprime',
        'rt_median_s', 'rt_cv', 'anticipatory_lick_frac',
        'beta_visual', 'beta_timing', 'strategy_index', 'strategy_label',
        'strategy_index_familiar', 'strategy_index_novel', 'strategy_stability_r',
    ]

    all_rows = []
    n_skipped = 0

    for _, pair_row in pairs.iterrows():
        mouse_id = pair_row['mouse_id']
        if mouse_id in existing_mice:
            n_skipped += 1
            continue

        familiar_sid = pair_row['familiar_session_id']
        novel_sid = pair_row['novel_session_id']

        print(f"Processing mouse {mouse_id} "
              f"(fam={int(familiar_sid) if not pd.isna(familiar_sid) else 'NA'}, "
              f"nov={int(novel_sid) if not pd.isna(novel_sid) else 'NA'})...",
              flush=True)

        result = classify_single_mouse(cache, mouse_id, familiar_sid, novel_sid)
        all_rows.append(result)

    if len(all_rows) > 0:
        batch_df = pd.DataFrame(all_rows)
        for col in CSV_COLUMNS:
            if col not in batch_df.columns:
                batch_df[col] = np.nan
        batch_df = batch_df[CSV_COLUMNS]
        batch_df.to_csv(OUTPUT_CSV, index=False)

    df = pd.read_csv(OUTPUT_CSV)

    both_sessions = df.dropna(subset=['strategy_index_familiar', 'strategy_index_novel'])
    if len(both_sessions) >= 3:
        r, p = stats.pearsonr(
            both_sessions['strategy_index_familiar'],
            both_sessions['strategy_index_novel']
        )
        df['strategy_stability_r'] = np.nan
        print(f"\nSplit-session stability: r={r:.3f}, p={p:.4f} (n={len(both_sessions)} mice)")
    else:
        r = np.nan
        print(f"\nSplit-session stability: insufficient mice with both sessions "
              f"(n={len(both_sessions)})")

    df.to_csv(OUTPUT_CSV, index=False)

    make_figure(df, OUTPUT_FIG)

    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*70}")
    print("STRATEGY CLASSIFICATION SUMMARY")
    print(f"{'='*70}")

    n_vis = (df['strategy_label'] == 'visual').sum()
    n_tim = (df['strategy_label'] == 'timing').sum()
    n_unc = (df['strategy_label'] == 'uncached').sum()
    n_unconv = (df['strategy_label'] == 'unconverged').sum()

    print(f"\nTotal mice:       {len(df)}")
    print(f"  Visual:         {n_vis}")
    print(f"  Timing:         {n_tim}")
    print(f"  Uncached:       {n_unc}")
    print(f"  Unconverged:    {n_unconv}")

    si_valid = df.loc[df['strategy_label'].isin(['visual', 'timing']), 'strategy_index']
    if len(si_valid) > 0:
        print(f"\nStrategy index (classified mice):")
        print(f"  Mean:   {si_valid.mean():.3f}")
        print(f"  Std:    {si_valid.std():.3f}")
        print(f"  Range:  [{si_valid.min():.3f}, {si_valid.max():.3f}]")
        print(f"  Median: {si_valid.median():.3f}")

    dp_valid = df['dprime'].dropna()
    if len(dp_valid) > 0:
        print(f"\nd-prime (all mice with data):")
        print(f"  Mean:   {dp_valid.mean():.3f}")
        print(f"  Range:  [{dp_valid.min():.3f}, {dp_valid.max():.3f}]")

    alf_valid = df['anticipatory_lick_frac'].dropna()
    if len(alf_valid) > 0:
        print(f"\nAnticipatory lick fraction:")
        print(f"  Mean:   {alf_valid.mean():.3f}")
        print(f"  Std:    {alf_valid.std():.3f}")
        print(f"  Range:  [{alf_valid.min():.3f}, {alf_valid.max():.3f}]")

    if not np.isnan(r):
        print(f"\nSplit-session stability: r={r:.3f} (n={len(both_sessions)})")

    bennett_note = (
        "Bennett et al. state mice 'overwhelmingly used visual cues.' "
        f"Our classification: {n_vis}/{n_vis + n_tim} = "
        f"{n_vis / max(n_vis + n_tim, 1) * 100:.1f}% visual."
    )
    if n_vis > n_tim:
        print(f"\n[CONSISTENT] {bennett_note}")
    else:
        print(f"\n[CHECK] {bennett_note}")

    print(f"\nElapsed time: {elapsed:.1f}s")
    print(f"Output CSV:   {OUTPUT_CSV}")
    print(f"Output Fig:   {OUTPUT_FIG}")


if __name__ == '__main__':
    main()
