"""
Compute Stimulus-Epoch Firing Rates (Week 5 fix)
=================================================

For sessions whose base NPZ has NaN firing_rates_stim_hz (or lacks a
``_stim_fr.npz`` companion), this script:

1. Reads each unit's spike times from the base NPZ.
2. Reads the NWB file to extract active, non-omitted Natural_Images
   presentation start times.
3. Counts spikes that fall within each 250 ms stimulus window.
4. Computes ``firing_rate_stim_hz = n_stim_spikes / (n_stimuli * 0.250)``.
5. Sets ``ccg_eligible_stim = (firing_rate_stim_hz >= 2.0)``.
6. Saves ``{session_id}_stim_fr.npz`` and updates the base NPZ in-place.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python compute_stim_firing_rates.py
"""

import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    CCG_MIN_FIRING_RATE_HZ,
    STIMULUS_DURATION_MS,
)

SPIKE_TIMES_DIR = 'results/derivatives/spike_times'
CACHE_BASE = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0'
    '/behavior_ecephys_sessions'
)
STIM_DURATION_S = STIMULUS_DURATION_MS / 1000.0


def _nwb_path(session_id: int) -> str:
    return os.path.join(
        CACHE_BASE, str(session_id),
        f'ecephys_session_{session_id}.nwb',
    )


def _load_stim_starts_from_nwb(nwb_path: str) -> np.ndarray:
    """Return start times for active, non-omitted Natural_Images presentations."""
    with h5py.File(nwb_path, 'r') as f:
        intervals = list(f['intervals'].keys())
        img_keys = [k for k in intervals if 'Natural_Images' in k]
        if not img_keys:
            return np.array([], dtype=np.float64)

        stim_grp = f[f'intervals/{img_keys[0]}']

        start_times = stim_grp['start_time'][()]

        if 'active' in stim_grp:
            obj = stim_grp['active']
            active = (
                obj['data'][()] if isinstance(obj, h5py.Group) and 'data' in obj
                else obj[()]
            )
        else:
            active = np.ones(len(start_times), dtype=bool)

        if 'omitted' in stim_grp:
            obj = stim_grp['omitted']
            omitted = (
                obj['data'][()] if isinstance(obj, h5py.Group) and 'data' in obj
                else obj[()]
            )
        else:
            omitted = np.zeros(len(start_times), dtype=bool)

    mask = np.asarray(active, dtype=bool) & ~np.asarray(omitted, dtype=bool)
    return start_times[mask].astype(np.float64)


def _needs_stim_fr(session_id: int) -> bool:
    """True if the base NPZ exists but has NaN stim rates and no companion."""
    stim_path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}_stim_fr.npz')
    if os.path.exists(stim_path):
        return False
    base_path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}.npz')
    if not os.path.exists(base_path):
        return False
    data = np.load(base_path, allow_pickle=True)
    if 'firing_rates_stim_hz' in data.files:
        fr = data['firing_rates_stim_hz']
        if not np.all(np.isnan(fr)):
            return False
    return True


def compute_for_session(session_id: int) -> dict:
    """Compute stimulus-epoch firing rates for one session.

    Returns a summary dict.
    """
    base_path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}.npz')
    data = np.load(base_path, allow_pickle=True)
    unit_ids = data['unit_ids'].astype(np.int64)
    spike_times_arr = data['spike_times']
    n_units = len(unit_ids)

    nwb = _nwb_path(session_id)
    if not os.path.exists(nwb):
        return {
            'session_id': session_id,
            'status': 'nwb_missing',
            'n_units': n_units,
            'n_stimuli': 0,
        }

    stim_starts = _load_stim_starts_from_nwb(nwb)
    n_stimuli = len(stim_starts)
    if n_stimuli == 0:
        return {
            'session_id': session_id,
            'status': 'no_stimuli',
            'n_units': n_units,
            'n_stimuli': 0,
        }

    fr_stim = np.zeros(n_units, dtype=np.float64)
    total_stim_duration = n_stimuli * STIM_DURATION_S

    for i in range(n_units):
        st = spike_times_arr[i]
        if st is None or len(st) == 0:
            continue
        st = np.asarray(st, dtype=np.float64)
        count = 0
        for t0 in stim_starts:
            count += int(np.sum((st >= t0) & (st < t0 + STIM_DURATION_S)))
        fr_stim[i] = count / total_stim_duration if total_stim_duration > 0 else 0.0

    ccg_eligible_stim = (fr_stim >= CCG_MIN_FIRING_RATE_HZ)

    stim_out = os.path.join(SPIKE_TIMES_DIR, f'{session_id}_stim_fr.npz')
    np.savez_compressed(
        stim_out,
        unit_ids=unit_ids,
        firing_rates_stim_hz=fr_stim,
        ccg_eligible_stim=ccg_eligible_stim,
    )

    return {
        'session_id': session_id,
        'status': 'ok',
        'n_units': n_units,
        'n_stimuli': n_stimuli,
        'n_ccg_eligible_stim': int(ccg_eligible_stim.sum()),
        'n_ccg_eligible_session': int(
            (data['firing_rates_session_hz'].astype(float) >= CCG_MIN_FIRING_RATE_HZ).sum()
        ) if 'firing_rates_session_hz' in data.files else 0,
        'mean_fr_stim': float(fr_stim.mean()),
    }


def main():
    t0 = time.perf_counter()

    print('=' * 72)
    print('COMPUTE STIMULUS-EPOCH FIRING RATES')
    print('=' * 72)

    if not os.path.exists(SPIKE_TIMES_DIR):
        print(f"ERROR: {SPIKE_TIMES_DIR} not found")
        sys.exit(1)

    all_base = []
    for fname in os.listdir(SPIKE_TIMES_DIR):
        if fname.endswith('.npz') and not fname.endswith('_stim_fr.npz'):
            sid_str = fname.replace('.npz', '')
            if sid_str.isdigit():
                all_base.append(int(sid_str))
    all_base.sort()

    needs = [sid for sid in all_base if _needs_stim_fr(sid)]
    already = len(all_base) - len(needs)

    print(f"Total base NPZ sessions: {len(all_base)}")
    print(f"Already have stim_fr:    {already}")
    print(f"Sessions to process:     {len(needs)}")
    print()

    summaries = []
    for i, sid in enumerate(needs, 1):
        print(f"[{i}/{len(needs)}] Session {sid} ...", end=' ', flush=True)
        try:
            s = compute_for_session(sid)
            summaries.append(s)
            if s['status'] == 'ok':
                print(
                    f"OK  {s['n_units']} units, {s['n_stimuli']} stimuli, "
                    f"CCG-elig stim={s['n_ccg_eligible_stim']} "
                    f"session={s['n_ccg_eligible_session']}"
                )
            else:
                print(f"SKIP ({s['status']})")
        except Exception as exc:
            print(f"FAILED: {exc}")
            summaries.append({
                'session_id': sid,
                'status': f'error: {str(exc)[:80]}',
                'n_units': 0,
                'n_stimuli': 0,
            })

    ok = [s for s in summaries if s['status'] == 'ok']
    n_ok = len(ok)

    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 72}")
    print("STIM FIRING RATE SUMMARY")
    print(f"{'=' * 72}")
    print(f"Sessions processed: {len(needs)}")
    print(f"Sessions skipped (already had stim_fr): {already}")
    print(f"Sessions OK: {n_ok}")
    print(f"Sessions failed/skipped: {len(needs) - n_ok}")

    if ok:
        mean_elig_stim = np.mean([s['n_ccg_eligible_stim'] for s in ok])
        mean_elig_sess = np.mean([s['n_ccg_eligible_session'] for s in ok])
        mean_units = np.mean([s['n_units'] for s in ok])
        print(f"Mean CCG-eligible (stim):    {mean_elig_stim:.1f} / {mean_units:.0f} = "
              f"{mean_elig_stim / max(mean_units, 1) * 100:.1f}%")
        print(f"Mean CCG-eligible (session): {mean_elig_sess:.1f} / {mean_units:.0f} = "
              f"{mean_elig_sess / max(mean_units, 1) * 100:.1f}%")
        delta = mean_elig_stim - mean_elig_sess
        print(f"Delta (stim vs session):     {delta:+.1f} units avg")

    print(f"\nElapsed time: {elapsed:.1f}s")
    print(f"Output dir:   {SPIKE_TIMES_DIR}/")


if __name__ == '__main__':
    main()
