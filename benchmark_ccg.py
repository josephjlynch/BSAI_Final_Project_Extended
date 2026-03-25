"""
CCG Timing Benchmark
====================

Benchmarks the real CCG implementation (src/connectivity.py) on one
cached session. The implementation faithfully ports ccg_library.py by
Disheng Tang (HChoiLab/functional-network, Nature Communications 2024).

Architecture:
  - Spike trains binned at 1ms (not 0.5ms)
  - Organized as (N x n_trials x T_per_trial) tensor
  - CCG computed per trial using as_strided matrix multiply
  - Jitter correction: pattern jitter, averaged across trials
  - Significance: z-score > n_sigma

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python benchmark_ccg.py [session_id]
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    CCG_BIN_SIZE_MS, CCG_WINDOW_BINS, CCG_WINDOW_MS,
    CCG_N_SURROGATES, CCG_JITTER_WINDOW_BINS, CCG_N_SIGMA,
    CCG_MIN_FIRING_RATE_HZ,
    PRESENCE_RATIO_MIN, ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX,
    RANDOM_SEED
)
from src.connectivity import (
    compute_all_ccgs_single_trial,
    PatternJitter,
    spike_times_to_trial_tensor,
)
from src.data_loading import load_cache

SESSION_IDS_FILE = 'session_ids.txt'
OUTPUT_FILE = 'results/tables/ccg_timing_estimate.txt'
N_TRIAL_BENCHMARK = 5   # compute on a small trial subset for timing; extrapolate
N_JITTER_BENCHMARK = 20 # fewer surrogates for benchmark; extrapolate to full


def load_session_ids():
    ids = []
    with open(SESSION_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(int(line))
    return ids


def find_cached_session(ids):
    cache_base = (
        'data/allen_cache/visual-behavior-neuropixels-0.5.0'
        '/behavior_ecephys_sessions'
    )
    if len(sys.argv) > 1:
        return int(sys.argv[1])
    for sid in ids:
        nwb = os.path.join(cache_base, str(sid), f'ecephys_session_{sid}.nwb')
        if os.path.exists(nwb):
            return sid
    return None


def main():
    ids = load_session_ids()
    first_sid = find_cached_session(ids)
    if first_sid is None:
        print("ERROR: No cached sessions found.")
        sys.exit(1)

    print(f"Using cached session {first_sid}", flush=True)
    print("Loading cache...", flush=True)
    cache = load_cache()

    print("Loading session NWB...", flush=True)
    session = cache.get_ecephys_session(ecephys_session_id=first_sid)
    print("Session loaded.", flush=True)

    session_units = session.get_units()
    quality_mask = (
        (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
        (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
        (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
    )
    if 'quality' in session_units.columns:
        quality_mask = quality_mask & (session_units['quality'] == 'good')
    filtered = session_units[quality_mask]
    print(f"Bennett-filtered units (this session): {len(filtered)}", flush=True)

    spike_times = session.spike_times
    print(f"Spike trains loaded: {len(spike_times)} units", flush=True)

    # --- Get engaged trials ---
    trials = session.trials
    reward_rate = session.get_reward_rate()
    engaged_mask = reward_rate >= 2.0
    engaged_trials = trials[engaged_mask]
    print(f"Engaged trials: {len(engaged_trials)}", flush=True)

    # --- 2Hz firing rate filter (Tang et al. 2024 Methods) ---
    unit_ids_bennett = [uid for uid in filtered.index if uid in spike_times]
    n_before_fr = len(unit_ids_bennett)

    unit_ids = []
    for uid in unit_ids_bennett:
        st = spike_times[uid]
        if len(st) < 2:
            continue
        duration_s = float(st[-1] - st[0])
        if duration_s > 0 and (len(st) / duration_s) >= CCG_MIN_FIRING_RATE_HZ:
            unit_ids.append(uid)

    print(f"Units before 2Hz filter: {n_before_fr}", flush=True)
    print(f"Units after 2Hz filter:  {len(unit_ids)} "
          f"({n_before_fr - len(unit_ids)} removed)", flush=True)

    # Benchmark subset: use up to 10 units for a manageable n^2 pairs test
    benchmark_n = min(10, len(unit_ids))
    benchmark_units = unit_ids[:benchmark_n]
    print(f"Benchmark units: {benchmark_n} (from {len(unit_ids)} CCG-eligible)", flush=True)

    # --- Build trial spike tensor ---
    # Disheng: use 250ms stimulus window only; gray screen (500ms) is separate
    trial_starts = engaged_trials['change_time_no_display_delay'].dropna().values
    trial_starts = trial_starts[:N_TRIAL_BENCHMARK]
    trial_duration_ms = 250.0

    print(f"Building spike tensor: {benchmark_n} units × {N_TRIAL_BENCHMARK} trials "
          f"× {int(trial_duration_ms)}ms...", flush=True)

    spike_dict = {uid: np.asarray(spike_times[uid]) for uid in benchmark_units
                  if uid in spike_times}

    t0 = time.perf_counter()
    tensor = spike_times_to_trial_tensor(
        spike_dict, benchmark_units,
        trial_starts, trial_duration_ms,
        bin_size_ms=CCG_BIN_SIZE_MS)
    t_tensor = time.perf_counter() - t0

    N, n_tr, T = tensor.shape
    print(f"Tensor shape: {tensor.shape} (built in {t_tensor:.3f}s)", flush=True)

    firing_rates = (np.count_nonzero(tensor, axis=(1, 2))
                    / (n_tr * T / 1000.0))

    # --- Benchmark: single trial CCG ---
    t0 = time.perf_counter()
    ccg_trial = compute_all_ccgs_single_trial(
        tensor[:, 0, :], firing_rates, CCG_WINDOW_BINS)
    t_single_trial = time.perf_counter() - t0
    n_pairs = N * (N - 1)
    print(f"\nSingle trial CCG ({N}x{N}, {n_pairs} pairs): {t_single_trial*1000:.1f}ms", flush=True)

    # --- Benchmark: pattern jitter surrogates for one trial ---
    t0 = time.perf_counter()
    pj = PatternJitter(
        num_sample=N_JITTER_BENCHMARK, spike_train=tensor[:, 0, :],
        L=CCG_JITTER_WINDOW_BINS, memory=False, seed=RANDOM_SEED)
    surrogates = pj.jitter()
    t_jitter = time.perf_counter() - t0
    t_per_surrogate = t_jitter / N_JITTER_BENCHMARK
    print(f"{N_JITTER_BENCHMARK} surrogates: {t_jitter*1000:.1f}ms "
          f"({t_per_surrogate*1000:.2f}ms per surrogate)", flush=True)

    # --- Extrapolate ---
    # Per trial: 1 CCG + N_SURROGATES surrogate CCGs
    t_per_trial = t_single_trial + CCG_N_SURROGATES * t_per_surrogate
    # For full N_units session (scale from benchmark_n):
    n_all = len(unit_ids)
    n_pairs_all = n_all * (n_all - 1)
    # Single-trial CCG scales as O(n_pairs) so scale from benchmark
    t_ccg_full = t_single_trial * (n_pairs_all / max(n_pairs, 1))
    t_jitter_full = t_per_surrogate * CCG_N_SURROGATES * (n_pairs_all / max(n_pairs, 1))
    t_per_trial_full = t_ccg_full + t_jitter_full

    n_engaged_total = len(engaged_trials)
    est_total_hours = (t_per_trial_full * n_engaged_total) / 3600

    report_lines = [
        "CCG TIMING BENCHMARK",
        "=" * 60,
        f"Implementation: src/connectivity (port of ccg_library.py, Disheng Tang 2024)",
        f"Session:         {first_sid}",
        f"",
        f"--- Architecture ---",
        f"Input format:    (N x n_trials x T) binary spike tensor",
        f"Bin size:        {CCG_BIN_SIZE_MS} ms",
        f"Trial window:    {int(trial_duration_ms)} ms = {T} bins",
        f"CCG window:      {CCG_WINDOW_BINS} bins ({CCG_WINDOW_MS:.0f} ms, one-sided)",
        f"Significance:    z-score > {CCG_N_SIGMA}σ (mean±σ of surrogate distribution)",
        f"Jitter:          pattern jitter, L={CCG_JITTER_WINDOW_BINS} bins",
        f"",
        f"--- Unit selection ---",
        f"Bennett quality filter:     {n_before_fr} units",
        f"After 2Hz FR threshold:     {n_all} units ({n_before_fr - n_all} removed)",
        f"",
        f"--- Benchmark (N={benchmark_n} units, {N_TRIAL_BENCHMARK} trials) ---",
        f"Single trial CCG ({n_pairs} pairs):  {t_single_trial*1000:.1f} ms",
        f"Per surrogate:                      {t_per_surrogate*1000:.2f} ms",
        f"Per trial (CCG + {CCG_N_SURROGATES} surrogates):    {t_per_trial*1000:.0f} ms",
        f"",
        f"--- Scaled to full session ---",
        f"CCG-eligible units (post-2Hz): {n_all}",
        f"All ordered pairs:             {n_pairs_all:,}",
        f"Engaged trials:                {n_engaged_total}",
        f"Per trial (full N):            {t_per_trial_full/60:.1f} min",
        f"Estimated total (all trials):  {est_total_hours:.1f} hours",
        f"",
        f"NOTE: Parallelizing across trials (joblib, n_cores=-1) reduces",
        f"wall-clock time proportionally to number of CPU cores.",
    ]

    print("\n" + "\n".join(report_lines))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
