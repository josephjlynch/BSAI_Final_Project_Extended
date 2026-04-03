"""
CCG Pilot Validation (Week 6)
=============================

Runs full jitter-corrected cross-correlogram analysis on the top 3
pilot sessions by CCG-eligible unit count.  Produces per-session
adjacency CSVs, raw CCG NPZ derivatives, a summary CSV, a 3x3
diagnostic figure, and an ANALYSIS_LOG.md entry.

All parameters imported from src/constants.py.  CCG pipeline from
src/connectivity.py (port of Disheng Tang's ccg_library.py, Tang et
al. 2024 Nature Communications).

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python compute_ccg_pilot.py
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.constants import (
    CCG_BIN_SIZE_MS,
    CCG_JITTER_MEMORY,
    CCG_JITTER_WINDOW_BINS,
    CCG_MIN_FIRING_RATE_HZ,
    CCG_N_SIGMA,
    CCG_N_SURROGATES,
    CCG_STIMULUS_EPOCH_MS,
    CCG_WINDOW_BINS,
    COMMON_INPUT_PEAK_WIDTH_MS,
    MONOSYNAPTIC_PEAK_WIDTH_MS,
    RANDOM_SEED,
)
from src.connectivity import (
    compute_ccg_corrected,
    get_significant_connections,
    spike_times_to_trial_tensor,
)
from src.data_loading import (
    VISUAL_AREAS,
    load_annotated_units,
    load_extracted_spike_times,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SUMMARY_CSV = 'results/tables/unit_annotation_summary.csv'
TRIAL_TABLES_DIR = 'results/derivatives/trial_tables'
SPIKE_TIMES_DIR = 'results/derivatives/spike_times'
UNIT_TABLES_DIR = 'results/derivatives/unit_tables'

OUTPUT_TABLES_DIR = 'results/tables'
OUTPUT_DERIVATIVES_DIR = 'results/derivatives'
OUTPUT_FIGURES_DIR = 'results/figures'

N_PILOT_SESSIONS = 3

# Connection-type classification thresholds (in bins)
MONO_THRESHOLD_BINS = MONOSYNAPTIC_PEAK_WIDTH_MS / CCG_BIN_SIZE_MS
COMMON_THRESHOLD_BINS = COMMON_INPUT_PEAK_WIDTH_MS / CCG_BIN_SIZE_MS

# Pilot-only: fewer surrogates than CCG_N_SURROGATES in constants.py for speed.
PILOT_N_SURROGATES = 25


# ===================================================================
# Step 1: Session Selection
# ===================================================================

def select_pilot_sessions(n: int = N_PILOT_SESSIONS) -> pd.DataFrame:
    """Select top-n sessions by CCG-eligible count that have trial tables."""
    summary = pd.read_csv(SUMMARY_CSV)
    available_trials = {
        int(f.replace('.parquet', ''))
        for f in os.listdir(TRIAL_TABLES_DIR)
        if f.endswith('.parquet')
    }
    candidates = summary[summary['session_id'].isin(available_trials)].copy()
    candidates = candidates.sort_values('n_ccg_eligible', ascending=False)
    top = candidates.head(n).reset_index(drop=True)

    print("=" * 72)
    print("PILOT SESSION SELECTION")
    print("=" * 72)
    for _, row in top.iterrows():
        print(f"  Session {int(row['session_id'])}: "
              f"{int(row['n_ccg_eligible'])} CCG-eligible units, "
              f"experience={row['experience_level']}")
    print()
    return top


# ===================================================================
# Step 2: Spike Tensor Construction
# ===================================================================

def build_spike_tensor(
    session_id: int,
) -> Tuple[np.ndarray, List[int], int, pd.DataFrame]:
    """Load spikes, filter to CCG-eligible, build trial-aligned tensor.

    Returns (tensor, unit_ids, n_trials, unit_annotation_df).
    """
    spike_dict, spike_meta = load_extracted_spike_times(
        session_id, derivatives_dir=SPIKE_TIMES_DIR, ccg_only=False,
    )

    stim_fr_path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}_stim_fr.npz')
    if os.path.exists(stim_fr_path):
        stim_data = np.load(stim_fr_path, allow_pickle=True)
        stim_uids = stim_data['unit_ids'].astype(np.int64)
        stim_eligible = stim_data['ccg_eligible_stim'].astype(bool)
        eligible_set = set(stim_uids[stim_eligible].tolist())
    else:
        eligible_set = set(
            spike_meta.loc[spike_meta['ccg_eligible'].astype(bool), 'unit_id']
            .astype(int).tolist()
        )

    unit_annot = load_annotated_units(
        session_id, derivatives_dir=UNIT_TABLES_DIR,
    )
    annot_eligible = set(
        unit_annot.loc[unit_annot['ccg_eligible'].astype(bool), 'unit_id']
        .astype(int).tolist()
    )
    eligible_set = eligible_set & annot_eligible & set(spike_dict.keys())

    visual_unit_ids = set(
        unit_annot.loc[
            unit_annot['structure_acronym'].isin(VISUAL_AREAS), 'unit_id'
        ].astype(int).tolist()
    )
    eligible_set = eligible_set & visual_unit_ids
    eligible_ids = sorted(eligible_set)

    trial_df = pd.read_parquet(
        os.path.join(TRIAL_TABLES_DIR, f'{session_id}.parquet')
    )
    active_engaged = trial_df[
        (trial_df['active'] == True)
        & (trial_df['engaged'] == True)
        & (trial_df['trial_type'] != 'omission')
    ]
    trial_starts = active_engaged['start_time_s'].values.astype(np.float64)

    print(f"  Building spike tensor: {len(eligible_ids)} units x "
          f"{len(trial_starts)} trials x {int(CCG_STIMULUS_EPOCH_MS)}ms")

    tensor = spike_times_to_trial_tensor(
        spike_times_dict=spike_dict,
        unit_ids=eligible_ids,
        trial_start_times=trial_starts,
        trial_duration_ms=CCG_STIMULUS_EPOCH_MS,
        bin_size_ms=CCG_BIN_SIZE_MS,
    )

    annot_for_session = unit_annot[
        unit_annot['unit_id'].isin(eligible_ids)
    ].set_index('unit_id').reindex(eligible_ids).reset_index()

    return tensor, eligible_ids, len(trial_starts), annot_for_session


# ===================================================================
# Step 3: CCG Computation
# ===================================================================

def run_ccg(
    spike_tensor: np.ndarray,
) -> np.ndarray:
    """Compute jitter-corrected CCG using parameters from constants.py."""
    return compute_ccg_corrected(
        spike_tensor,
        window=CCG_WINDOW_BINS,
        num_jitter=PILOT_N_SURROGATES,
        L=CCG_JITTER_WINDOW_BINS,
        memory=CCG_JITTER_MEMORY,
        seed=RANDOM_SEED,
        use_parallel=True,
        num_cores=4,
    )


# ===================================================================
# Step 4: Significance + Connection Classification
# ===================================================================

def classify_connections(
    sig_ccg: np.ndarray,
    sig_conf: np.ndarray,
    sig_off: np.ndarray,
    sig_dur: np.ndarray,
    unit_ids: List[int],
    unit_annot: pd.DataFrame,
) -> pd.DataFrame:
    """Build adjacency table from significance matrices."""
    annot_lookup = unit_annot.set_index('unit_id')
    rows: List[Dict] = []
    N = sig_ccg.shape[0]
    for i in range(N):
        for j in range(N):
            if i == j or np.isnan(sig_ccg[i, j]):
                continue
            pre_uid = unit_ids[i]
            post_uid = unit_ids[j]
            sign = 1 if sig_ccg[i, j] > 0 else -1
            offset_bins = sig_off[i, j]
            peak_lag_ms = offset_bins * CCG_BIN_SIZE_MS

            if offset_bins <= MONO_THRESHOLD_BINS:
                conn_type = 'monosynaptic'
            elif offset_bins > COMMON_THRESHOLD_BINS:
                conn_type = 'common_input'
            else:
                conn_type = 'intermediate'

            pre_row = annot_lookup.loc[pre_uid] if pre_uid in annot_lookup.index else None
            post_row = annot_lookup.loc[post_uid] if post_uid in annot_lookup.index else None

            rows.append({
                'pre_unit_id': pre_uid,
                'post_unit_id': post_uid,
                'sign': sign,
                'connection_type': conn_type,
                'peak_lag_ms': peak_lag_ms,
                'z_score': sig_conf[i, j],
                'pre_area': pre_row['structure_acronym'] if pre_row is not None else 'unknown',
                'pre_layer': pre_row['cortical_layer'] if pre_row is not None else 'unknown',
                'pre_waveform': pre_row['waveform_type'] if pre_row is not None else 'unknown',
                'post_area': post_row['structure_acronym'] if post_row is not None else 'unknown',
                'post_layer': post_row['cortical_layer'] if post_row is not None else 'unknown',
                'post_waveform': post_row['waveform_type'] if post_row is not None else 'unknown',
            })

    return pd.DataFrame(rows)


def compute_summary_stats(
    adj_df: pd.DataFrame,
    session_id: int,
    n_units: int,
    n_trials: int,
    elapsed_s: float,
) -> Dict:
    """Compute per-session summary statistics."""
    n_conn = len(adj_df)
    n_exc = int((adj_df['sign'] == 1).sum()) if n_conn else 0
    n_inh = int((adj_df['sign'] == -1).sum()) if n_conn else 0
    n_mono = int((adj_df['connection_type'] == 'monosynaptic').sum()) if n_conn else 0
    n_common = int((adj_df['connection_type'] == 'common_input').sum()) if n_conn else 0
    n_possible = n_units * (n_units - 1)
    edge_density = n_conn / n_possible if n_possible > 0 else 0.0

    within = 0
    between = 0
    if n_conn:
        same_area = adj_df['pre_area'] == adj_df['post_area']
        within = int(same_area.sum())
        between = int((~same_area).sum())
    pct_within = (within / n_conn * 100) if n_conn else 0.0
    pct_between = (between / n_conn * 100) if n_conn else 0.0

    return {
        'session_id': session_id,
        'n_units_eligible': n_units,
        'n_trials': n_trials,
        'elapsed_s': round(elapsed_s, 1),
        'n_connections_total': n_conn,
        'n_excitatory': n_exc,
        'n_inhibitory': n_inh,
        'n_monosynaptic': n_mono,
        'n_common_input': n_common,
        'edge_density': round(edge_density, 6),
        'pct_within_area': round(pct_within, 1),
        'pct_between_area': round(pct_between, 1),
    }


# ===================================================================
# Step 7: Figure
# ===================================================================

def make_pilot_figure(
    session_results: List[Tuple[int, pd.DataFrame, np.ndarray, np.ndarray]],
) -> None:
    """3x3 figure: lag dist, type breakdown, area matrix per session."""
    n_sess = len(session_results)
    fig, axes = plt.subplots(3, n_sess, figsize=(6 * n_sess, 15))
    if n_sess == 1:
        axes = axes[:, np.newaxis]

    for col, (sid, adj_df, sig_off_mat, sig_ccg_mat) in enumerate(session_results):
        # Row 1: lag distribution
        ax = axes[0, col]
        if len(adj_df):
            lags = adj_df['peak_lag_ms'].values
            ax.hist(lags, bins=50, edgecolor='black', linewidth=0.3)
        ax.set_xlabel('Peak lag (ms)')
        ax.set_ylabel('Count')
        ax.set_title(f'Session {sid}\nLag distribution (N={len(adj_df)})')

        # Row 2: connection type breakdown
        ax = axes[1, col]
        if len(adj_df):
            categories = []
            counts = []
            for sign_label, sign_val in [('Exc', 1), ('Inh', -1)]:
                for ctype in ['monosynaptic', 'intermediate', 'common_input']:
                    mask = (adj_df['sign'] == sign_val) & (adj_df['connection_type'] == ctype)
                    categories.append(f'{sign_label}\n{ctype[:4]}')
                    counts.append(int(mask.sum()))
            colors = ['#2166ac', '#4393c3', '#92c5de',
                      '#d6604d', '#f4a582', '#fddbc7']
            ax.bar(range(len(categories)), counts, color=colors,
                   edgecolor='black', linewidth=0.3)
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, fontsize=7)
        ax.set_ylabel('Count')
        ax.set_title(f'Connection type breakdown')

        # Row 3: area connection matrix
        ax = axes[2, col]
        area_labels = VISUAL_AREAS
        n_areas = len(area_labels)
        matrix = np.zeros((n_areas, n_areas))
        if len(adj_df):
            area_to_idx = {a: i for i, a in enumerate(area_labels)}
            for _, row in adj_df.iterrows():
                pre_i = area_to_idx.get(row['pre_area'])
                post_i = area_to_idx.get(row['post_area'])
                if pre_i is not None and post_i is not None:
                    matrix[pre_i, post_i] += 1
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal')
        ax.set_xticks(range(n_areas))
        ax.set_xticklabels(area_labels, fontsize=7, rotation=45)
        ax.set_yticks(range(n_areas))
        ax.set_yticklabels(area_labels, fontsize=7)
        ax.set_xlabel('Post area')
        ax.set_ylabel('Pre area')
        ax.set_title('Area connection matrix')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        'CCG Pilot Validation — '
        f'{n_sess} sessions, '
        f'window={CCG_WINDOW_BINS}bins, '
        f'jitter={PILOT_N_SURROGATES}surr, '
        f'sigma={CCG_N_SIGMA}',
        fontsize=12, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUTPUT_FIGURES_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(OUTPUT_FIGURES_DIR, 'ccg_pilot_results.png'),
        dpi=200,
    )
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================

def process_single_session(
    session_id: int,
) -> Tuple[Dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Full CCG pipeline for one session. Saves per-session outputs."""
    print(f"\n{'=' * 72}")
    print(f"SESSION {session_id}")
    print(f"{'=' * 72}")

    # Check for existing outputs (idempotency)
    adj_csv = os.path.join(
        OUTPUT_TABLES_DIR, f'ccg_pilot_{session_id}_adjacency.csv')
    npz_path = os.path.join(
        OUTPUT_DERIVATIVES_DIR, f'ccg_pilot_{session_id}_ccg.npz')

    if os.path.exists(adj_csv) and os.path.exists(npz_path):
        print(f"  Outputs already exist, loading from disk...")
        adj_df = pd.read_csv(adj_csv)
        npz_data = np.load(npz_path, allow_pickle=True)
        sig_off_mat = npz_data['sig_off']
        sig_ccg_mat = npz_data['sig_ccg']
        unit_ids = npz_data['unit_ids'].tolist()

        trial_df = pd.read_parquet(
            os.path.join(TRIAL_TABLES_DIR, f'{session_id}.parquet'))
        n_trials = int((
            (trial_df['active'] == True)
            & (trial_df['engaged'] == True)
            & (trial_df['trial_type'] != 'omission')
        ).sum())

        summary = compute_summary_stats(
            adj_df, session_id, len(unit_ids), n_trials, elapsed_s=0.0)
        summary['elapsed_s'] = 0.0
        return summary, adj_df, sig_off_mat, sig_ccg_mat

    # Step 2: Build spike tensor
    t0 = time.perf_counter()
    tensor, unit_ids, n_trials, unit_annot = build_spike_tensor(session_id)
    N = tensor.shape[0]
    print(f"  Tensor shape: {tensor.shape} "
          f"(built in {time.perf_counter() - t0:.1f}s)")
    print(f"  Spike density: "
          f"{tensor.sum() / max(tensor.size, 1):.4f} spikes/bin")

    # Step 3: CCG computation
    print(f"  Computing CCG: {N} units, {n_trials} trials, "
          f"window={CCG_WINDOW_BINS}, surrogates={PILOT_N_SURROGATES}, "
          f"jitter_L={CCG_JITTER_WINDOW_BINS}...")
    t_ccg = time.perf_counter()
    ccg_corrected = run_ccg(tensor)
    elapsed_ccg = time.perf_counter() - t_ccg
    print(f"  CCG computed in {elapsed_ccg:.1f}s "
          f"({elapsed_ccg / 3600:.2f}h)")

    # Step 4: Significance
    print(f"  Detecting significant connections (n_sigma={CCG_N_SIGMA})...")
    sig_ccg, sig_conf, sig_off, sig_dur = get_significant_connections(
        ccg_corrected, n_sigma=CCG_N_SIGMA)

    n_sig = int(np.count_nonzero(~np.isnan(sig_ccg)) - N)
    print(f"  Significant connections: {n_sig}")

    # Firing rates from tensor (for NPZ)
    N_units, n_tr, T = tensor.shape
    firing_rates = (
        np.count_nonzero(tensor, axis=(1, 2)) / (n_tr * T / 1000.0)
    )

    # Step 4b: Classify connections
    adj_df = classify_connections(
        sig_ccg, sig_conf, sig_off, sig_dur, unit_ids, unit_annot)

    elapsed_total = time.perf_counter() - t0

    # Step 5: Save adjacency CSV
    os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)
    adj_df.to_csv(adj_csv, index=False)
    print(f"  Saved: {adj_csv}")

    # Step 6: Save raw CCG NPZ
    os.makedirs(OUTPUT_DERIVATIVES_DIR, exist_ok=True)
    np.savez_compressed(
        npz_path,
        ccg_corrected=ccg_corrected,
        sig_ccg=sig_ccg,
        sig_conf=sig_conf,
        sig_off=sig_off,
        sig_dur=sig_dur,
        unit_ids=np.array(unit_ids),
        firing_rates=firing_rates,
    )
    print(f"  Saved: {npz_path}")

    summary = compute_summary_stats(
        adj_df, session_id, len(unit_ids), n_trials, elapsed_total)
    return summary, adj_df, sig_off, sig_ccg


def main() -> None:
    t_start = time.perf_counter()

    # Step 1: Session selection
    pilot_df = select_pilot_sessions(N_PILOT_SESSIONS)
    session_ids = pilot_df['session_id'].astype(int).tolist()

    summary_rows: List[Dict] = []
    figure_data: List[Tuple[int, pd.DataFrame, np.ndarray, np.ndarray]] = []

    for sid in session_ids:
        summary, adj_df, sig_off_mat, sig_ccg_mat = process_single_session(sid)
        summary_rows.append(summary)
        figure_data.append((sid, adj_df, sig_off_mat, sig_ccg_mat))

    # Step 7: Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUTPUT_TABLES_DIR, 'ccg_pilot_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved: {summary_csv_path}")

    # Step 8: Figure
    print("Generating 3x3 pilot figure...")
    make_pilot_figure(figure_data)
    print(f"Saved: {os.path.join(OUTPUT_FIGURES_DIR, 'ccg_pilot_results.png')}")

    # Step 9: Print summary
    total_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 72)
    print("CCG PILOT VALIDATION — FINAL SUMMARY")
    print("=" * 72)
    print(f"Sessions processed: {len(session_ids)}")
    for row in summary_rows:
        sid = row['session_id']
        print(f"\n  Session {sid}:")
        print(f"    Units:       {row['n_units_eligible']}")
        print(f"    Trials:      {row['n_trials']}")
        print(f"    Elapsed:     {row['elapsed_s']:.1f}s "
              f"({row['elapsed_s'] / 3600:.2f}h)")
        print(f"    Connections: {row['n_connections_total']} total "
              f"({row['n_excitatory']} exc, {row['n_inhibitory']} inh)")
        print(f"    Types:       {row['n_monosynaptic']} mono, "
              f"{row['n_common_input']} common")
        print(f"    Edge density:{row['edge_density']:.6f}")
        print(f"    Within-area: {row['pct_within_area']:.1f}%  "
              f"Between-area: {row['pct_between_area']:.1f}%")

        # QC warnings
        ed = row['edge_density']
        if ed < 0.001:
            print(f"    WARNING: Edge density < 0.001 (very sparse)")
        elif ed > 0.05:
            print(f"    WARNING: Edge density > 0.05 (unusually dense)")

    print(f"\nTotal elapsed: {total_elapsed:.1f}s "
          f"({total_elapsed / 3600:.2f}h)")
    print(f"\nParameters used:")
    print(f"  CCG_BIN_SIZE_MS       = {CCG_BIN_SIZE_MS}")
    print(f"  CCG_WINDOW_BINS       = {CCG_WINDOW_BINS}")
    print(f"  PILOT_N_SURROGATES    = {PILOT_N_SURROGATES}  "
          f"(pilot speed; constants CCG_N_SURROGATES={CCG_N_SURROGATES} for full run)")
    print(f"  CCG_JITTER_WINDOW_BINS= {CCG_JITTER_WINDOW_BINS}")
    print(f"  CCG_JITTER_MEMORY     = {CCG_JITTER_MEMORY}")
    print(f"  CCG_N_SIGMA           = {CCG_N_SIGMA}")
    print(f"  CCG_STIMULUS_EPOCH_MS = {CCG_STIMULUS_EPOCH_MS}")
    print(f"  CCG_MIN_FIRING_RATE_HZ= {CCG_MIN_FIRING_RATE_HZ}")
    print(f"  MONO_THRESHOLD        = {MONOSYNAPTIC_PEAK_WIDTH_MS}ms "
          f"({MONO_THRESHOLD_BINS} bins)")
    print(f"  COMMON_THRESHOLD      = {COMMON_INPUT_PEAK_WIDTH_MS}ms "
          f"({COMMON_THRESHOLD_BINS} bins)")
    print(f"  RANDOM_SEED           = {RANDOM_SEED}")
    print("=" * 72)


if __name__ == '__main__':
    main()
