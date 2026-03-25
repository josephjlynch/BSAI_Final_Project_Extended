"""
Reset Index Computation (Molano-Mazón et al. 2023)
===================================================

Computes the Reset Index (RI) per mouse from the Allen Visual Behavior
Neuropixels dataset, following Equation 6 of:

  Molano-Mazón et al. (2023). "Recurrent networks endowed with structural
  priors explain suboptimal animal behavior." Current Biology 33, 622-638.

Method:
    1. For each mouse, pool engaged trials from all cached sessions.
    2. Tag each trial with the outcome of the *previous* trial (hit/miss).
    3. Split trials into after-hit and after-miss subsets.
    4. Fit an extended GLM with 10 trial-history lags on each subset:
         P(lick) = sigmoid(beta_stim * is_change
                           + sum_{k=1}^{10} beta_k * outcome_{t-k})
       where outcome_{t-k} = +1 (hit) or -1 (miss).
    5. Compute T = sum of |beta_k| for each subset.
    6. RI = 1 - |T_after-miss| / |T_after-hit|

Outputs:
    results/tables/strategy_classification.csv — RI columns appended
    results/figures/history_kernels.png
    results/figures/transition_kernels.png

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python compute_reset_index.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    GLM_HISTORY_LAGS,
    RESET_INDEX_MIN_TRIALS,
    RANDOM_SEED,
    STRATEGY_INDEX_THRESHOLD,
    ENGAGEMENT_REWARD_RATE_MIN,
)
from src.data_loading import (
    load_cache,
    get_mouse_session_map,
    get_change_detection_trials,
)

STRATEGY_CSV = 'results/tables/strategy_classification.csv'
OUTPUT_HISTORY_FIG = 'results/figures/history_kernels.png'
OUTPUT_TRANSITION_FIG = 'results/figures/transition_kernels.png'
SESSION_IDS_FILE = 'session_ids.txt'
CACHE_BASE = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0'
    '/behavior_ecephys_sessions'
)


def is_session_cached(sid):
    nwb = os.path.join(CACHE_BASE, str(sid), f'ecephys_session_{sid}.nwb')
    return os.path.exists(nwb)


def build_history_matrix(outcomes, n_lags):
    """Build a trial-history feature matrix from a sequence of outcomes.

    Parameters
    ----------
    outcomes : np.ndarray of shape (n_trials,)
        Encoded outcomes: +1 (hit) or -1 (miss).
    n_lags : int
        Number of history lags to include.

    Returns
    -------
    X_history : np.ndarray of shape (n_trials, n_lags)
        Column k contains outcome_{t-(k+1)}.  Rows with insufficient
        history (first n_lags trials) are filled with 0.
    """
    n = len(outcomes)
    X = np.zeros((n, n_lags), dtype=float)
    for k in range(n_lags):
        lag = k + 1
        X[lag:, k] = outcomes[:-lag] if lag < n else 0.0
    return X


def fit_history_glm(is_change, responded, outcomes, n_lags):
    """Fit the extended GLM on a trial subset.

    Parameters
    ----------
    is_change : np.ndarray (n_trials,)
        Binary: 1 if image changed, 0 otherwise.
    responded : np.ndarray (n_trials,)
        Binary: 1 if mouse licked, 0 otherwise.
    outcomes : np.ndarray (n_trials,)
        Encoded previous-trial outcomes: +1 (hit) or -1 (miss).
    n_lags : int
        Number of history lags.

    Returns
    -------
    betas : np.ndarray (n_lags,) or None
        History kernel weights (beta_1 through beta_{n_lags}).
    beta_stim : float or None
        Stimulus coefficient.
    converged : bool
    """
    X_hist = build_history_matrix(outcomes, n_lags)
    valid_mask = np.arange(len(is_change)) >= n_lags
    X_hist = X_hist[valid_mask]
    is_change_v = is_change[valid_mask]
    responded_v = responded[valid_mask]

    if len(responded_v) < n_lags + 2:
        return None, None, False

    if len(np.unique(responded_v)) < 2:
        return None, None, False

    X = np.column_stack([is_change_v.reshape(-1, 1), X_hist])
    y = responded_v.astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    converged = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            clf = LogisticRegression(
                penalty=None, max_iter=2000,
                random_state=RANDOM_SEED, solver='lbfgs',
            )
            clf.fit(X_scaled, y)
        except Exception:
            return None, None, False

        if any('converge' in str(w.message).lower() for w in caught
               if issubclass(w.category, UserWarning)):
            converged = False

    beta_stim = clf.coef_[0][0]
    betas = clf.coef_[0][1:]
    return betas, beta_stim, converged


def compute_ri_for_mouse(cache, mouse_id, session_ids):
    """Compute Reset Index and history kernels for a single mouse.

    Returns a dict of result columns, or None on total failure.
    """
    row = {'mouse_id': mouse_id}

    all_trials_list = []
    for sid in session_ids:
        if not is_session_cached(sid):
            continue
        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)
        except Exception as e:
            print(f"  WARNING: could not load session {sid}: {e}")
            continue

        trial_data = get_change_detection_trials(session, engaged_only=True)
        engaged = trial_data['engaged_trials']

        engaged = engaged[engaged['sdt_category'].isin(
            ['hit', 'miss', 'false_alarm', 'correct_reject']
        )].copy()

        if len(engaged) == 0:
            continue

        engaged['responded'] = engaged['sdt_category'].isin(
            ['hit', 'false_alarm']
        ).astype(int)
        engaged['is_change'] = engaged['sdt_category'].isin(
            ['hit', 'miss']
        ).astype(int)

        engaged['outcome_encoded'] = np.where(
            engaged['sdt_category'] == 'hit', 1.0,
            np.where(engaged['sdt_category'] == 'miss', -1.0, 0.0)
        )

        engaged['previous_sdt'] = engaged['sdt_category'].shift(1)
        engaged['previous_outcome'] = engaged['outcome_encoded'].shift(1)

        all_trials_list.append(engaged)

    if len(all_trials_list) == 0:
        row['reset_index'] = np.nan
        row['ri_n_after_hit'] = 0
        row['ri_n_after_miss'] = 0
        row['ri_flag'] = 'no_data'
        return row, None, None

    pooled = pd.concat(all_trials_list, ignore_index=True)

    change_trials = pooled[pooled['is_change'] == 1].copy()

    after_hit = change_trials[change_trials['previous_sdt'] == 'hit'].copy()
    after_miss = change_trials[change_trials['previous_sdt'] == 'miss'].copy()

    row['ri_n_after_hit'] = len(after_hit)
    row['ri_n_after_miss'] = len(after_miss)

    if len(after_hit) < RESET_INDEX_MIN_TRIALS or len(after_miss) < RESET_INDEX_MIN_TRIALS:
        min_count = min(len(after_hit), len(after_miss))
        row['reset_index'] = np.nan
        row['ri_flag'] = f'insufficient_trials (min={min_count}, need={RESET_INDEX_MIN_TRIALS})'
        print(f"  Mouse {mouse_id}: after-hit={len(after_hit)}, "
              f"after-miss={len(after_miss)} — below threshold {RESET_INDEX_MIN_TRIALS}")
        return row, None, None

    betas_hit, _, conv_hit = fit_history_glm(
        after_hit['is_change'].values,
        after_hit['responded'].values,
        after_hit['outcome_encoded'].values,
        GLM_HISTORY_LAGS,
    )
    betas_miss, _, conv_miss = fit_history_glm(
        after_miss['is_change'].values,
        after_miss['responded'].values,
        after_miss['outcome_encoded'].values,
        GLM_HISTORY_LAGS,
    )

    if betas_hit is None or betas_miss is None:
        row['reset_index'] = np.nan
        row['ri_flag'] = 'glm_failed'
        return row, None, None

    T_hit = np.sum(np.abs(betas_hit))
    T_miss = np.sum(np.abs(betas_miss))

    if T_hit > 0:
        ri = 1.0 - T_miss / T_hit
    else:
        ri = np.nan

    row['reset_index'] = ri

    flags = []
    if not conv_hit:
        flags.append('after-hit GLM did not converge')
    if not conv_miss:
        flags.append('after-miss GLM did not converge')
    row['ri_flag'] = '; '.join(flags) if flags else 'ok'

    return row, betas_hit, betas_miss


def make_history_kernel_figure(mouse_results, strategy_df, output_path):
    """Per-mouse history kernel weight profiles, grouped by Piet strategy label.

    Reproduces the spirit of Molano-Mazón Figure 1E using real mouse data.
    """
    lags = np.arange(1, GLM_HISTORY_LAGS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    strategy_colors = {'visual': '#2176AE', 'timing': '#D64933', 'other': '#888888'}

    for ax, (subset_label, kernel_key) in zip(
        axes, [('After-Hit Trials', 'betas_hit'), ('After-Miss Trials', 'betas_miss')]
    ):
        for mouse_id, betas_hit, betas_miss in mouse_results:
            if betas_hit is None or betas_miss is None:
                continue
            betas = betas_hit if kernel_key == 'betas_hit' else betas_miss

            strategy = 'other'
            if strategy_df is not None and mouse_id in strategy_df['mouse_id'].values:
                label = strategy_df.loc[
                    strategy_df['mouse_id'] == mouse_id, 'strategy_label'
                ].iloc[0]
                if label in ('visual', 'timing'):
                    strategy = label

            color = strategy_colors[strategy]
            ax.plot(lags, betas, color=color, alpha=0.3, linewidth=0.8)

        for strategy in ['visual', 'timing']:
            all_betas = []
            for mouse_id, betas_hit, betas_miss in mouse_results:
                if betas_hit is None or betas_miss is None:
                    continue
                betas = betas_hit if kernel_key == 'betas_hit' else betas_miss
                if strategy_df is not None and mouse_id in strategy_df['mouse_id'].values:
                    label = strategy_df.loc[
                        strategy_df['mouse_id'] == mouse_id, 'strategy_label'
                    ].iloc[0]
                    if label == strategy:
                        all_betas.append(betas)
            if len(all_betas) >= 2:
                mean_betas = np.mean(all_betas, axis=0)
                sem_betas = np.std(all_betas, axis=0) / np.sqrt(len(all_betas))
                color = strategy_colors[strategy]
                ax.plot(lags, mean_betas, color=color, linewidth=2.5,
                        label=f'{strategy} mean (n={len(all_betas)})')
                ax.fill_between(lags, mean_betas - sem_betas, mean_betas + sem_betas,
                                color=color, alpha=0.2)

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel('Trial Lag', fontsize=11)
        ax.set_title(subset_label, fontsize=12)
        ax.set_xticks(lags)
        ax.legend(fontsize=9)

    axes[0].set_ylabel('History Kernel Weight (GLM beta)', fontsize=11)
    fig.suptitle('History Kernel Profiles by Piet Strategy\n'
                 '(Molano-Mazón Eq. 3, cf. Figure 1E)', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def make_transition_kernel_figure(mouse_results, output_path):
    """Lag-by-lag weights after-hit vs. after-miss, averaged across mice.

    Reproduces the spirit of Molano-Mazón Figure 5D.
    """
    lags = np.arange(1, GLM_HISTORY_LAGS + 1)

    all_hit_betas = []
    all_miss_betas = []
    for _, betas_hit, betas_miss in mouse_results:
        if betas_hit is not None and betas_miss is not None:
            all_hit_betas.append(betas_hit)
            all_miss_betas.append(betas_miss)

    if len(all_hit_betas) < 2:
        print(f"WARNING: only {len(all_hit_betas)} mice with valid kernels, "
              f"skipping transition kernel figure")
        return

    hit_arr = np.array(all_hit_betas)
    miss_arr = np.array(all_miss_betas)
    n = len(hit_arr)

    hit_mean = np.mean(hit_arr, axis=0)
    hit_sem = np.std(hit_arr, axis=0) / np.sqrt(n)
    miss_mean = np.mean(miss_arr, axis=0)
    miss_sem = np.std(miss_arr, axis=0) / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lags, hit_mean, 'o-', color='#2176AE', linewidth=2,
            markersize=6, label=f'After-Hit (n={n})')
    ax.fill_between(lags, hit_mean - hit_sem, hit_mean + hit_sem,
                    color='#2176AE', alpha=0.2)

    ax.plot(lags, miss_mean, 's-', color='#D64933', linewidth=2,
            markersize=6, label=f'After-Miss (n={n})')
    ax.fill_between(lags, miss_mean - miss_sem, miss_mean + miss_sem,
                    color='#D64933', alpha=0.2)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Trial Lag', fontsize=11)
    ax.set_ylabel('Mean History Kernel Weight', fontsize=11)
    ax.set_title('Transition Kernel: After-Hit vs. After-Miss\n'
                 '(Molano-Mazón Eq. 3, cf. Figure 5D)', fontsize=13)
    ax.set_xticks(lags)
    ax.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved: {output_path}")


def main():
    t_start = time.perf_counter()
    print("=" * 70)
    print("RESET INDEX COMPUTATION (Molano-Mazón et al. 2023, Eq. 6)")
    print("=" * 70)
    print(f"  GLM_HISTORY_LAGS      = {GLM_HISTORY_LAGS}")
    print(f"  RESET_INDEX_MIN_TRIALS = {RESET_INDEX_MIN_TRIALS}")
    print(f"  RANDOM_SEED           = {RANDOM_SEED}")

    strategy_df = None
    if os.path.exists(STRATEGY_CSV):
        strategy_df = pd.read_csv(STRATEGY_CSV)
        print(f"\nLoaded existing strategy classification: {STRATEGY_CSV} "
              f"({len(strategy_df)} mice)")
    else:
        print(f"\nNo existing {STRATEGY_CSV} found. RI results will be saved "
              f"as a standalone CSV.")

    print("\nLoading cache...", flush=True)
    cache = load_cache()

    session_map = get_mouse_session_map(cache)
    mice_sessions = session_map.groupby('mouse_id')['ecephys_session_id'].apply(list)
    print(f"Found {len(mice_sessions)} mice in session table")

    mouse_results = []
    ri_rows = []

    for mouse_id, sids in mice_sessions.items():
        print(f"Processing mouse {mouse_id} ({len(sids)} sessions)...", flush=True)
        row, betas_hit, betas_miss = compute_ri_for_mouse(cache, mouse_id, sids)
        ri_rows.append(row)
        mouse_results.append((mouse_id, betas_hit, betas_miss))

    ri_df = pd.DataFrame(ri_rows)

    if strategy_df is not None:
        for col in ['reset_index', 'ri_n_after_hit', 'ri_n_after_miss', 'ri_flag']:
            if col in strategy_df.columns:
                strategy_df = strategy_df.drop(columns=[col])
        merged = strategy_df.merge(ri_df, on='mouse_id', how='left')
        merged.to_csv(STRATEGY_CSV, index=False)
        print(f"\nAppended RI columns to {STRATEGY_CSV}")
        output_df = merged
    else:
        os.makedirs(os.path.dirname(STRATEGY_CSV), exist_ok=True)
        ri_df.to_csv(STRATEGY_CSV, index=False)
        print(f"\nSaved standalone RI results to {STRATEGY_CSV}")
        output_df = ri_df

    make_history_kernel_figure(mouse_results, strategy_df, OUTPUT_HISTORY_FIG)
    make_transition_kernel_figure(mouse_results, OUTPUT_TRANSITION_FIG)

    elapsed = time.perf_counter() - t_start

    print(f"\n{'=' * 70}")
    print("RESET INDEX SUMMARY")
    print(f"{'=' * 70}")

    ri_valid = output_df['reset_index'].dropna()
    ri_ok = output_df[output_df['ri_flag'] == 'ok']
    ri_insuff = output_df[output_df['ri_flag'].str.startswith('insufficient', na=False)]
    ri_failed = output_df[output_df['ri_flag'] == 'glm_failed']
    ri_nodata = output_df[output_df['ri_flag'] == 'no_data']

    print(f"\nTotal mice:              {len(output_df)}")
    print(f"  RI computed (ok):      {len(ri_ok)}")
    print(f"  Insufficient trials:   {len(ri_insuff)}")
    print(f"  GLM failed:            {len(ri_failed)}")
    print(f"  No data (uncached):    {len(ri_nodata)}")

    if len(ri_valid) > 0:
        print(f"\nReset Index (valid mice, n={len(ri_valid)}):")
        print(f"  Mean:   {ri_valid.mean():.3f}")
        print(f"  Std:    {ri_valid.std():.3f}")
        print(f"  Median: {ri_valid.median():.3f}")
        print(f"  Range:  [{ri_valid.min():.3f}, {ri_valid.max():.3f}]")

        if strategy_df is not None and 'strategy_label' in output_df.columns:
            for label in ['visual', 'timing']:
                subset = output_df[
                    (output_df['strategy_label'] == label) &
                    output_df['reset_index'].notna()
                ]
                if len(subset) > 0:
                    print(f"\n  {label.capitalize()} strategy mice (n={len(subset)}):")
                    print(f"    RI mean:   {subset['reset_index'].mean():.3f}")
                    print(f"    RI median: {subset['reset_index'].median():.3f}")

    n_kernels = sum(1 for _, bh, bm in mouse_results if bh is not None and bm is not None)
    print(f"\nHistory kernel pairs computed: {n_kernels}")
    print(f"\nElapsed time: {elapsed:.1f}s")
    print(f"Output CSV:              {STRATEGY_CSV}")
    print(f"History kernel figure:   {OUTPUT_HISTORY_FIG}")
    print(f"Transition kernel figure: {OUTPUT_TRANSITION_FIG}")


if __name__ == '__main__':
    main()
