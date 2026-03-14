"""
DEPRECATED: Uses Visual Coding stimulus paradigm (Natural_Images/gabor).
Incompatible with Visual Behavior change detection task.
Retained for reference only. Do not run.
"""
import sys
sys.exit("ERROR: Deprecated script. See docstring.")

# --- Original code below (unreachable) ---

"""
Multi-Session Analysis Script
==============================
Standalone equivalent of tutorial.ipynb Cell 18.
Processes all 65 sessions across all 6 visual areas with checkpointing.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python run_multi_session.py

Resumes automatically from checkpoint if interrupted.
Results saved to: ./results/multi_session/checkpoint_results_6area.csv
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loading import (
    VISUAL_AREAS, AREA_NAMES,
    load_cache, load_session_data,
    compute_firing_rates_by_area,
    get_stimulus_times
)
from connectivity import compute_all_connectivity_matrices, summarize_connectivity
from graph_metrics import compute_all_metrics

# =============================================================================
# CONSTANTS (match tutorial.ipynb Cell 3)
# =============================================================================

CACHE_DIR       = './data/allen_cache/'
SESSION_IDS_FILE = './session_ids.txt'
CHECKPOINT_FILE = './results/multi_session/checkpoint_results_6area.csv'
BIN_SIZE        = 0.050   # 50 ms bins
MIN_RATE        = 0.1     # Hz minimum firing rate
MIN_NEURONS     = 5       # minimum good neurons per area
THRESHOLD       = 0.1     # correlation threshold for adjacency

np.random.seed(42)


def load_session_ids(path: str):
    """Read session IDs from session_ids.txt, skipping comment lines."""
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line.strip()) for line in lines if line.strip() and not line.startswith('#')]


def main():
    print("=" * 70)
    print("Multi-Session Analysis (6 Visual Areas)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load Allen cache once
    # ------------------------------------------------------------------
    print("\n[1/4] Loading Allen cache (one-time)...")
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = load_cache(CACHE_DIR)
    print("[OK] Cache loaded")

    # ------------------------------------------------------------------
    # Step 2: Load session IDs
    # ------------------------------------------------------------------
    print("\n[2/4] Loading session IDs...")
    valid_session_ids = load_session_ids(SESSION_IDS_FILE)
    print(f"[OK] {len(valid_session_ids)} sessions to process")

    # ------------------------------------------------------------------
    # Step 3: Resume from checkpoint if available
    # ------------------------------------------------------------------
    os.makedirs('./results/multi_session', exist_ok=True)

    if os.path.exists(CHECKPOINT_FILE):
        existing_df = pd.read_csv(CHECKPOINT_FILE)
        completed_sessions = set(existing_df['session_id'].values)
        all_results = existing_df.to_dict('records')
        print(f"\n[3/4] Resuming: {len(completed_sessions)} sessions already done")
    else:
        completed_sessions = set()
        all_results = []
        print("\n[3/4] Starting fresh (no checkpoint found)")

    sessions_to_process = [s for s in valid_session_ids if s not in completed_sessions]
    print(f"[...] {len(sessions_to_process)} sessions remaining")

    # ------------------------------------------------------------------
    # Step 4: Process sessions
    # ------------------------------------------------------------------
    print(f"\n[4/4] Processing sessions (verbose=False, cache reused)...\n")

    for session_id in tqdm(sessions_to_process, desc="Sessions"):
        try:
            sess_data = load_session_data(
                cache_dir=CACHE_DIR,
                session_id=session_id,
                quality_filter=True,
                min_neurons=MIN_NEURONS
            )

            if len(sess_data['areas_present']) == 0:
                continue

            stim = sess_data['stim_presentations']
            nat_starts, nat_ends = get_stimulus_times(stim, 'natural')
            gab_starts, gab_ends = get_stimulus_times(stim, 'gabor')

            if len(nat_starts) == 0 or len(gab_starts) == 0:
                continue

            rates_nat = compute_firing_rates_by_area(
                sess_data['spike_times_by_area'], nat_starts, nat_ends, bin_size=BIN_SIZE
            )
            rates_gab = compute_firing_rates_by_area(
                sess_data['spike_times_by_area'], gab_starts, gab_ends, bin_size=BIN_SIZE
            )

            fr_nat  = {a: r for a, (r, _) in rates_nat.items()}
            uid_nat = {a: u for a, (_, u) in rates_nat.items()}
            fr_gab  = {a: r for a, (r, _) in rates_gab.items()}
            uid_gab = {a: u for a, (_, u) in rates_gab.items()}

            mat_nat = compute_all_connectivity_matrices(fr_nat, uid_nat, min_rate_threshold=MIN_RATE)
            mat_gab = compute_all_connectivity_matrices(fr_gab, uid_gab, min_rate_threshold=MIN_RATE)

            conn_nat = summarize_connectivity(mat_nat)
            conn_gab = summarize_connectivity(mat_gab)

            result = {
                'session_id': session_id,
                'n_natural_stim': len(nat_starts),
                'n_gabor_stim': len(gab_starts),
            }

            for area in VISUAL_AREAS:
                result[f'n_{area}'] = sess_data['neuron_counts'].get(area, 0)

                if area in mat_nat['within']:
                    m_nat = compute_all_metrics(mat_nat['within'][area][0], threshold=THRESHOLD)
                    result[f'{area}_modularity_natural'] = m_nat['modularity']['modularity_Q']
                    result[f'{area}_clustering_natural'] = m_nat['clustering']['mean']
                    result[f'{area}_density_natural']    = m_nat['density']
                    key_nat = f'within_{area}'
                    if key_nat in conn_nat:
                        result[f'{area}_corr_natural'] = conn_nat[key_nat]['mean']

                if area in mat_gab['within']:
                    m_gab = compute_all_metrics(mat_gab['within'][area][0], threshold=THRESHOLD)
                    result[f'{area}_modularity_gabor'] = m_gab['modularity']['modularity_Q']
                    result[f'{area}_clustering_gabor'] = m_gab['clustering']['mean']
                    result[f'{area}_density_gabor']    = m_gab['density']
                    key_gab = f'within_{area}'
                    if key_gab in conn_gab:
                        result[f'{area}_corr_gabor'] = conn_gab[key_gab]['mean']

            all_results.append(result)
            completed_sessions.add(session_id)
            pd.DataFrame(all_results).to_csv(CHECKPOINT_FILE, index=False)

        except Exception as e:
            print(f"\nSession {session_id} failed: {str(e)[:100]}")
            continue

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    print(f"\n[OK] Processed {len(results_df)} sessions successfully")
    print(f"Results saved to: {CHECKPOINT_FILE}")

    print(f"\nPer-area data availability:")
    for area in VISUAL_AREAS:
        col = f'{area}_modularity_natural'
        if col in results_df.columns:
            n_valid = results_df[col].notna().sum()
            print(f"  {AREA_NAMES[area]} ({area}): {n_valid}/{len(results_df)} sessions")
        else:
            print(f"  {AREA_NAMES[area]} ({area}): 0/{len(results_df)} sessions")

    print("\nNext: Open tutorial.ipynb and run cells from Part 5 onward.")


if __name__ == '__main__':
    main()
