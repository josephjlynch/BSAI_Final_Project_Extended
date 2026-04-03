"""
Download and Extract Sessions to Reach 90 Working Sessions
===========================================================

Downloads uncached sessions one at a time via AllenSDK, immediately
extracts spike times to NPZ, then releases the session object.

Stops when:
  - 90 working sessions (NPZ files) exist, OR
  - Free disk drops below DISK_FLOOR_GB, OR
  - All 50 remaining sessions have been attempted

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python download_and_extract.py
"""

import os
import sys
import shutil
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    CCG_MIN_FIRING_RATE_HZ,
    PRESENCE_RATIO_MIN, ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX,
)
from src.data_loading import load_cache, VISUAL_AREAS, SUBCORTICAL_AREAS

TARGET_SESSIONS   = 90
DISK_FLOOR_GB     = 8.0
SESSION_IDS_FILE  = 'session_ids.txt'
DERIVATIVES_DIR   = 'results/derivatives/spike_times'
SUMMARY_CSV       = 'results/tables/spike_extraction_summary.csv'
CACHE_BASE        = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0'
    '/behavior_ecephys_sessions'
)


def free_gb():
    stat = shutil.disk_usage('/')
    return stat.free / 1e9


def is_nwb_cached(sid):
    nwb = os.path.join(CACHE_BASE, str(sid), f'ecephys_session_{sid}.nwb')
    return os.path.exists(nwb)


def npz_done(sid):
    return os.path.exists(os.path.join(DERIVATIVES_DIR, f'{sid}.npz'))


def n_working():
    return sum(1 for f in os.listdir(DERIVATIVES_DIR)
               if f.endswith('.npz') and not f.endswith('_stim_fr.npz'))


def compute_firing_rates(spike_times_dict, unit_ids):
    rates = np.zeros(len(unit_ids), dtype=np.float64)
    for i, uid in enumerate(unit_ids):
        st = spike_times_dict.get(uid, np.array([]))
        if len(st) >= 2:
            d = st[-1] - st[0]
            rates[i] = len(st) / d if d > 0 else 0.0
    return rates


def extract_session(session, sid, meta_row, unit_table):
    su = unit_table[unit_table['ecephys_session_id'] == sid].copy()
    mask = (
        (su['presence_ratio'] > PRESENCE_RATIO_MIN) &
        (su['isi_violations'] < ISI_VIOLATIONS_MAX) &
        (su['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
    )
    if 'quality' in su.columns:
        mask = mask & (su['quality'] == 'good')
    su = su[mask].copy()

    if 'waveform_duration' in su.columns:
        su['waveform_type'] = np.where(su['waveform_duration'] > 0.4, 'RS', 'FS')
    else:
        su['waveform_type'] = 'unknown'

    unit_ids = su.index.values.astype(np.int64)
    n = len(unit_ids)
    areas = su['structure_acronym'].values.astype(str) \
        if 'structure_acronym' in su.columns else np.full(n, 'unknown', dtype=object)
    wt = su['waveform_type'].values.astype(str)

    spk = session.spike_times
    st_arr = np.empty(n, dtype=object)
    for i, uid in enumerate(unit_ids):
        st_arr[i] = np.asarray(spk.get(uid, np.array([], dtype=np.float64)), dtype=np.float64)

    fr = compute_firing_rates(spk, unit_ids)
    ccg_el = (fr >= CCG_MIN_FIRING_RATE_HZ).astype(bool)
    fr_stim = np.full(n, np.nan, dtype=np.float64)

    mouse_id = int(meta_row.get('mouse_id', -1)) if 'mouse_id' in meta_row.index else -1
    exp_lvl  = str(meta_row.get('experience_level', 'unknown')) if 'experience_level' in meta_row.index else 'unknown'

    return {
        'unit_ids': unit_ids, 'spike_times': st_arr, 'areas': areas,
        'waveform_types': wt, 'firing_rates_session_hz': fr,
        'ccg_eligible': ccg_el, 'firing_rates_stim_hz': fr_stim,
        'session_id': np.int64(sid), 'mouse_id': np.int64(mouse_id),
        'experience_level': exp_lvl,
    }, {
        'session_id': sid, 'mouse_id': mouse_id, 'experience_level': exp_lvl,
        'n_units_quality': n, 'n_units_ccg_eligible': int(ccg_el.sum()),
        'n_visual_units': int(np.isin(areas, VISUAL_AREAS).sum()),
        'n_subcortical_units': int(np.isin(areas, SUBCORTICAL_AREAS).sum()),
    }


def main():
    print('=' * 70)
    print('DOWNLOAD AND EXTRACT — TARGET 90 WORKING SESSIONS')
    print('=' * 70)
    print(f'Current working sessions: {n_working()}')
    print(f'Free disk:                {free_gb():.1f} GB')
    print(f'Target:                   {TARGET_SESSIONS}')
    print(f'Disk floor:               {DISK_FLOOR_GB} GB')
    print()

    all_sids = [int(x.strip()) for x in open(SESSION_IDS_FILE)
                if x.strip() and not x.strip().startswith('#')]
    to_do = [s for s in all_sids if not npz_done(s)]
    print(f'Sessions without NPZ: {len(to_do)}')
    print()

    print('Loading cache (unit_table + session_table loaded ONCE)...', flush=True)
    cache = load_cache()
    unit_table   = cache.get_unit_table()
    session_table = cache.get_ecephys_session_table()
    print(f'unit_table: {len(unit_table):,}  session_table: {len(session_table):,}')
    print()

    os.makedirs(DERIVATIVES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)

    summary_rows = []
    n_ok = n_failed = 0

    for sid in tqdm(to_do, desc='Downloading + extracting'):
        cur = n_working()
        disk = free_gb()

        if cur >= TARGET_SESSIONS:
            tqdm.write(f'Reached target {TARGET_SESSIONS} working sessions. Stopping.')
            break
        if disk < DISK_FLOOR_GB:
            tqdm.write(f'Disk floor reached ({disk:.1f} GB free). Stopping.')
            break

        npz_path = os.path.join(DERIVATIVES_DIR, f'{sid}.npz')
        t0 = time.perf_counter()
        meta_row = session_table.loc[sid] if sid in session_table.index \
            else pd.Series({'mouse_id': -1, 'experience_level': 'unknown'})

        try:
            tqdm.write(f'  {sid}: downloading... (disk={disk:.1f}GB, done={cur})', end='\r')
            session = cache.get_ecephys_session(ecephys_session_id=sid)
            npz_data, summary = extract_session(session, sid, meta_row, unit_table)
            np.savez_compressed(npz_path, **npz_data)
            elapsed = time.perf_counter() - t0
            mb = os.path.getsize(npz_path) / 1e6
            summary['extraction_time_s'] = round(elapsed, 2)
            summary['file_size_mb']      = round(mb, 2)
            summary['status']            = 'ok'
            n_ok += 1
            tqdm.write(
                f'  {sid}: OK  {summary["n_units_quality"]} units '
                f'({summary["n_units_ccg_eligible"]} CCG)  '
                f'{mb:.1f} MB  {elapsed:.0f}s  disk={free_gb():.1f}GB'
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            tqdm.write(f'  FAILED {sid}: {str(e)[:120]}')
            summary = {
                'session_id': sid,
                'mouse_id': int(meta_row.get('mouse_id', -1)) if 'mouse_id' in meta_row.index else -1,
                'experience_level': str(meta_row.get('experience_level','unknown')) if 'experience_level' in meta_row.index else 'unknown',
                'n_units_quality': 0, 'n_units_ccg_eligible': 0,
                'n_visual_units': 0, 'n_subcortical_units': 0,
                'extraction_time_s': round(elapsed, 2),
                'file_size_mb': 0.0,
                'status': f'failed: {str(e)[:80]}',
            }
            n_failed += 1

        summary_rows.append(summary)

        # Flush summary every 3 sessions
        if len(summary_rows) % 3 == 0:
            batch = pd.DataFrame(summary_rows)
            if os.path.exists(SUMMARY_CSV):
                batch.to_csv(SUMMARY_CSV, mode='a', header=False, index=False)
            else:
                batch.to_csv(SUMMARY_CSV, index=False)
            summary_rows = []

    if summary_rows:
        batch = pd.DataFrame(summary_rows)
        if os.path.exists(SUMMARY_CSV):
            batch.to_csv(SUMMARY_CSV, mode='a', header=False, index=False)
        else:
            batch.to_csv(SUMMARY_CSV, index=False)

    print()
    print('=' * 70)
    print('DOWNLOAD SUMMARY')
    print('=' * 70)
    print(f'Sessions OK this run:      {n_ok}')
    print(f'Sessions failed this run:  {n_failed}')
    print(f'Total working sessions:    {n_working()}')
    print(f'Free disk remaining:       {free_gb():.1f} GB')


if __name__ == '__main__':
    main()
