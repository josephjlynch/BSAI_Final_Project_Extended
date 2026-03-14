import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    CCG_BIN_SIZE_MS, CCG_WINDOW_MS, CCG_N_SURROGATES,
    CCG_JITTER_WINDOW_MS, PRESENCE_RATIO_MIN,
    ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX
)
from src.data_loading import load_cache

SESSION_IDS_FILE = 'session_ids.txt'
OUTPUT_FILE = 'results/tables/ccg_timing_estimate.txt'


def compute_ccg_placeholder(spike_times_a, spike_times_b,
                            bin_size_s, window_s):
    """Placeholder CCG: histogram of spike time differences."""
    diffs = []
    for t in spike_times_a:
        mask = (spike_times_b >= t - window_s) & (spike_times_b <= t + window_s)
        diffs.extend(spike_times_b[mask] - t)
    if len(diffs) == 0:
        n_bins = int(2 * window_s / bin_size_s)
        return np.zeros(n_bins), np.zeros(n_bins + 1)
    diffs = np.array(diffs)
    bins = np.arange(-window_s, window_s + bin_size_s, bin_size_s)
    counts, edges = np.histogram(diffs, bins=bins)
    return counts, edges


def jitter_spikes(spike_times, jitter_window_s):
    """Uniformly jitter each spike within +/- jitter_window."""
    jitter = np.random.uniform(-jitter_window_s, jitter_window_s, len(spike_times))
    return np.sort(spike_times + jitter)


def main():
    ids = []
    with open(SESSION_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(int(line))

    cache_base = 'data/allen_cache/visual-behavior-neuropixels-0.5.0/behavior_ecephys_sessions'
    if len(sys.argv) > 1:
        first_sid = int(sys.argv[1])
    else:
        first_sid = None
        for sid in ids:
            nwb_path = os.path.join(cache_base, str(sid), f'ecephys_session_{sid}.nwb')
            if os.path.exists(nwb_path):
                first_sid = sid
                break
        if first_sid is None:
            print("ERROR: No cached sessions found.")
            sys.exit(1)

    print(f"Using cached session {first_sid}", flush=True)
    print("Loading cache...", flush=True)
    cache = load_cache()
    print("Loading unit table...", flush=True)
    unit_table = cache.get_unit_table()
    print(f"Loading session NWB...", flush=True)
    session = cache.get_ecephys_session(ecephys_session_id=first_sid)
    print("Session loaded.", flush=True)
    session_units = unit_table[unit_table['ecephys_session_id'] == first_sid]
    filtered = session_units[
        (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
        (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
        (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
    ]

    visp_units = filtered[filtered['structure_acronym'] == 'VISp']
    print(f"VISp units (Bennett-filtered): {len(visp_units)}")

    if len(visp_units) < 2:
        print("ERROR: Need at least 2 VISp units for benchmark")
        return

    spike_times = session.spike_times

    # Find two high-firing units (> 5 Hz).
    # Compute rate per unit from its own spike span to avoid scanning all units.
    selected = []
    for uid in visp_units.index:
        if uid in spike_times and len(spike_times[uid]) > 0:
            unit_span = float(spike_times[uid][-1] - spike_times[uid][0]) if len(spike_times[uid]) > 1 else 1.0
            rate = len(spike_times[uid]) / max(unit_span, 1.0)
            if rate > 5.0:
                selected.append((uid, rate))
        if len(selected) >= 2:
            break

    if len(selected) < 2:
        print("WARNING: Could not find 2 units with rate > 5 Hz, using top 2")
        rates = []
        for uid in visp_units.index:
            if uid in spike_times and len(spike_times[uid]) > 0:
                unit_span = float(spike_times[uid][-1] - spike_times[uid][0]) if len(spike_times[uid]) > 1 else 1.0
                rate = len(spike_times[uid]) / max(unit_span, 1.0)
                rates.append((uid, rate))
        rates.sort(key=lambda x: -x[1])
        selected = rates[:2]

    uid_a, rate_a = selected[0]
    uid_b, rate_b = selected[1]
    st_a = spike_times[uid_a]
    st_b = spike_times[uid_b]
    max_spikes_bench = 10000
    if len(st_a) > max_spikes_bench:
        st_a = st_a[:max_spikes_bench]
    if len(st_b) > max_spikes_bench:
        st_b = st_b[:max_spikes_bench]

    print(f"\nBenchmark pair:")
    print(f"  Unit A: {uid_a} ({rate_a:.1f} Hz, {len(st_a)} spikes)")
    print(f"  Unit B: {uid_b} ({rate_b:.1f} Hz, {len(st_b)} spikes)")

    bin_size_s = CCG_BIN_SIZE_MS / 1000.0
    window_s = CCG_WINDOW_MS / 1000.0
    jitter_s = CCG_JITTER_WINDOW_MS / 1000.0

    t0 = time.time()
    ccg, edges = compute_ccg_placeholder(st_a, st_b, bin_size_s, window_s)
    t_single = time.time() - t0
    print(f"\nSingle CCG time: {t_single:.3f}s")

    n_benchmark = min(100, CCG_N_SURROGATES)
    t0 = time.time()
    for _ in range(n_benchmark):
        jittered_b = jitter_spikes(st_b, jitter_s)
        compute_ccg_placeholder(st_a, jittered_b, bin_size_s, window_s)
    t_surrogates = time.time() - t0
    t_per_surrogate = t_surrogates / n_benchmark
    print(f"{n_benchmark} surrogates time: {t_surrogates:.3f}s "
          f"({t_per_surrogate:.4f}s per surrogate)")

    t_per_pair = t_single + CCG_N_SURROGATES * t_per_surrogate
    n_visp = len(visp_units)
    n_pairs_visp = n_visp * (n_visp - 1) // 2
    n_all = len(filtered)
    n_pairs_all = n_all * (n_all - 1) // 2

    est_visp_hours = (n_pairs_visp * t_per_pair) / 3600
    est_all_hours = (n_pairs_all * t_per_pair) / 3600

    report = [
        "CCG TIMING BENCHMARK",
        "=" * 50,
        f"Session: {first_sid}",
        f"Benchmark pair: {uid_a} ({rate_a:.1f} Hz) vs {uid_b} ({rate_b:.1f} Hz)",
        f"",
        f"Single CCG time: {t_single:.4f}s",
        f"Per surrogate: {t_per_surrogate:.4f}s",
        f"Surrogates per pair: {CCG_N_SURROGATES}",
        f"Time per pair (CCG + surrogates): {t_per_pair:.3f}s",
        f"",
        f"VISp units: {n_visp}",
        f"VISp pairs: {n_pairs_visp:,}",
        f"Estimated VISp-only time: {est_visp_hours:.1f} hours",
        f"",
        f"All filtered units this session: {n_all}",
        f"All pairs: {n_pairs_all:,}",
        f"Estimated all-pairs time: {est_all_hours:.1f} hours",
        f"",
        f"NOTE: This is a naive placeholder CCG. Optimized implementations",
        f"(e.g., elephant, or vectorized numpy) will be significantly faster.",
        f"This estimate provides an upper bound for planning purposes.",
    ]

    print("\n" + "\n".join(report))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(report) + "\n")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
