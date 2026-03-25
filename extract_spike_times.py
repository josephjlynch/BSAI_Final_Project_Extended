"""
Spike Time Extraction (Week 3)
===============================

Pre-extracts spike times for ALL 4-criteria quality-filtered units
from each locally cached session into lightweight per-session NPZ files
at results/derivatives/spike_times/{session_id}.npz.

CRITICAL: Extracts ALL quality-filtered units, NOT just CCG-eligible ones.
The 2Hz firing rate filter is a CCG-specific constraint applied at
CCG computation time (Week 4+). Downstream analyses (GLM encoding models,
population decoding, firing-rate-matched controls, dimensionality reduction)
require units below 2Hz. Discarding them at extraction is irreversible.

Firing rate note: firing_rates_session_hz is computed session-wide from
first-to-last spike. Tang et al. 2024 specifies '2Hz during all stimuli.'
Stimulus-epoch firing rate (firing_rates_stim_hz) will be computed in
Week 4 after trial alignment; the ccg_eligible flag will be refined then.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python extract_spike_times.py
"""

import os
import sys
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

SESSION_IDS_FILE = 'session_ids.txt'
DERIVATIVES_DIR = 'results/derivatives/spike_times'
SUMMARY_CSV = 'results/tables/spike_extraction_summary.csv'
CACHE_BASE = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0'
    '/behavior_ecephys_sessions'
)


def is_session_cached(sid):
    nwb = os.path.join(CACHE_BASE, str(sid), f'ecephys_session_{sid}.nwb')
    return os.path.exists(nwb)


def compute_session_firing_rates(spike_times_dict, unit_ids):
    """Compute session-wide firing rate for each unit.

    Rate = len(spikes) / (t_last - t_first).
    Returns np.array of float64, same order as unit_ids.
    """
    rates = np.zeros(len(unit_ids), dtype=np.float64)
    for i, uid in enumerate(unit_ids):
        st = spike_times_dict.get(uid, np.array([]))
        if len(st) >= 2:
            duration = st[-1] - st[0]
            rates[i] = len(st) / duration if duration > 0 else 0.0
        elif len(st) == 1:
            rates[i] = 0.0
        else:
            rates[i] = 0.0
    return rates


def extract_session(session, session_id, session_meta_row, unit_table):
    """Extract spike times for one session and return the NPZ data dict.

    Parameters
    ----------
    session : BehaviorEcephysSession
    session_id : int
    session_meta_row : pd.Series
        Row from session_table for this session.
    unit_table : pd.DataFrame
        Full unit table (loaded once by caller).

    Returns
    -------
    npz_data : dict  (keys match NPZ save spec)
    summary  : dict  (one row for spike_extraction_summary.csv)
    """
    session_units = unit_table[
        unit_table['ecephys_session_id'] == session_id
    ].copy()

    mask = (
        (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
        (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
        (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
    )
    if 'quality' in session_units.columns:
        mask = mask & (session_units['quality'] == 'good')
    session_units = session_units[mask].copy()

    if 'waveform_duration' in session_units.columns:
        session_units['waveform_type'] = np.where(
            session_units['waveform_duration'] > 0.4, 'RS', 'FS'
        )
    else:
        session_units['waveform_type'] = 'unknown'

    unit_ids = session_units.index.values.astype(np.int64)
    n_units = len(unit_ids)

    areas = session_units['structure_acronym'].values.astype(str) \
        if 'structure_acronym' in session_units.columns \
        else np.full(n_units, 'unknown', dtype=object)

    waveform_types = session_units['waveform_type'].values.astype(str)

    spike_times_dict = session.spike_times

    spike_times_arr = np.empty(n_units, dtype=object)
    for i, uid in enumerate(unit_ids):
        st = spike_times_dict.get(uid, np.array([], dtype=np.float64))
        spike_times_arr[i] = np.asarray(st, dtype=np.float64)

    fr_session = compute_session_firing_rates(spike_times_dict, unit_ids)
    ccg_eligible = (fr_session >= CCG_MIN_FIRING_RATE_HZ).astype(bool)

    firing_rates_stim_hz = np.full(n_units, np.nan, dtype=np.float64)

    mouse_id = int(session_meta_row.get('mouse_id', -1)) \
        if 'mouse_id' in session_meta_row.index else -1
    experience_level = str(session_meta_row.get('experience_level', 'unknown')) \
        if 'experience_level' in session_meta_row.index else 'unknown'

    npz_data = {
        'unit_ids': unit_ids,
        'spike_times': spike_times_arr,
        'areas': areas,
        'waveform_types': waveform_types,
        'firing_rates_session_hz': fr_session,
        'ccg_eligible': ccg_eligible,
        'firing_rates_stim_hz': firing_rates_stim_hz,
        'session_id': np.int64(session_id),
        'mouse_id': np.int64(mouse_id),
        'experience_level': experience_level,
    }

    n_ccg = int(ccg_eligible.sum())
    n_visual = int(np.isin(areas, VISUAL_AREAS).sum())
    n_subcortical = int(np.isin(areas, SUBCORTICAL_AREAS).sum())

    summary = {
        'session_id': session_id,
        'mouse_id': mouse_id,
        'experience_level': experience_level,
        'n_units_quality': n_units,
        'n_units_ccg_eligible': n_ccg,
        'n_visual_units': n_visual,
        'n_subcortical_units': n_subcortical,
        'extraction_time_s': np.nan,
        'file_size_mb': np.nan,
        'status': 'ok',
    }

    return npz_data, summary


def main():
    t_global = time.perf_counter()

    print('=' * 70)
    print('SPIKE TIME EXTRACTION (Week 3)')
    print('=' * 70)

    os.makedirs(DERIVATIVES_DIR, exist_ok=True)
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
    # Check NPZ files on disk (primary) — catches sessions written before
    # the summary CSV was flushed on a previous interrupted run.
    for fname in os.listdir(DERIVATIVES_DIR):
        if fname.endswith('.npz'):
            try:
                already_done.add(int(fname.replace('.npz', '')))
            except ValueError:
                pass
    # Also pull any 'ok' rows from summary CSV that may not have NPZ files yet
    if os.path.exists(SUMMARY_CSV):
        prev = pd.read_csv(SUMMARY_CSV)
        done_ok = prev[prev['status'] == 'ok']['session_id'].values
        already_done.update(int(x) for x in done_ok)
    print(f'Already extracted:    {len(already_done)} (skipping)')

    to_process = [sid for sid in cached_sids if sid not in already_done]
    print(f'To process:           {len(to_process)}')
    print()

    print('Loading cache (unit_table + session_table loaded ONCE)...',
          flush=True)
    cache = load_cache()
    unit_table = cache.get_unit_table()
    session_table = cache.get_ecephys_session_table()
    print(f'unit_table rows: {len(unit_table):,}')
    print(f'session_table rows: {len(session_table):,}')
    print()

    summary_rows = []
    n_total_units = 0
    n_total_ccg = 0
    n_failed = 0
    n_skipped = len(already_done)

    for sid in tqdm(to_process, desc='Extracting sessions'):
        npz_path = os.path.join(DERIVATIVES_DIR, f'{sid}.npz')
        t0 = time.perf_counter()

        if sid in session_table.index:
            meta_row = session_table.loc[sid]
        else:
            meta_row = pd.Series({'mouse_id': -1, 'experience_level': 'unknown'})

        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)
            npz_data, summary = extract_session(
                session, sid, meta_row, unit_table
            )

            np.savez_compressed(npz_path, **npz_data)

            elapsed = time.perf_counter() - t0
            file_size_mb = os.path.getsize(npz_path) / 1e6

            summary['extraction_time_s'] = round(elapsed, 2)
            summary['file_size_mb'] = round(file_size_mb, 2)

            n_total_units += summary['n_units_quality']
            n_total_ccg += summary['n_units_ccg_eligible']

            tqdm.write(
                f'  {sid}: {summary["n_units_quality"]} units '
                f'({summary["n_units_ccg_eligible"]} CCG-eligible), '
                f'{file_size_mb:.1f} MB, {elapsed:.1f}s'
            )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            tqdm.write(f'  FAILED {sid}: {e}')
            summary = {
                'session_id': sid,
                'mouse_id': int(meta_row.get('mouse_id', -1))
                    if 'mouse_id' in meta_row.index else -1,
                'experience_level': str(meta_row.get('experience_level', 'unknown'))
                    if 'experience_level' in meta_row.index else 'unknown',
                'n_units_quality': 0,
                'n_units_ccg_eligible': 0,
                'n_visual_units': 0,
                'n_subcortical_units': 0,
                'extraction_time_s': round(elapsed, 2),
                'file_size_mb': 0.0,
                'status': f'failed: {str(e)[:80]}',
            }
            n_failed += 1

        summary_rows.append(summary)

        if len(summary_rows) % 5 == 0:
            batch_df = pd.DataFrame(summary_rows)
            if os.path.exists(SUMMARY_CSV) and (n_skipped > 0 or len(summary_rows) > 5):
                batch_df.to_csv(SUMMARY_CSV, mode='a', header=False, index=False)
            else:
                batch_df.to_csv(SUMMARY_CSV, index=False)
            summary_rows = []

    if summary_rows:
        batch_df = pd.DataFrame(summary_rows)
        if os.path.exists(SUMMARY_CSV) and (n_skipped > 0 or n_total_units > 0):
            batch_df.to_csv(SUMMARY_CSV, mode='a', header=False, index=False)
        else:
            batch_df.to_csv(SUMMARY_CSV, index=False)

    summary_df = pd.read_csv(SUMMARY_CSV)
    total_size_mb = summary_df.loc[summary_df['status'] == 'ok', 'file_size_mb'].sum()
    total_extracted = int(summary_df.loc[summary_df['status'] == 'ok', 'n_units_quality'].sum())
    total_ccg = int(summary_df.loc[summary_df['status'] == 'ok', 'n_units_ccg_eligible'].sum())
    n_ok = int((summary_df['status'] == 'ok').sum())

    elapsed_global = time.perf_counter() - t_global

    print(f'\n{"=" * 70}')
    print('SPIKE EXTRACTION SUMMARY')
    print(f'{"=" * 70}')
    print(f'\nSessions processed this run: {len(to_process)}')
    print(f'Sessions OK (total):         {n_ok}')
    print(f'Sessions failed this run:    {n_failed}')
    print(f'Sessions skipped (done):     {n_skipped}')
    print(f'\nTotal quality-filtered units: {total_extracted:,}')
    print(f'Total CCG-eligible (>=2Hz):   {total_ccg:,}')
    pct_ccg = total_ccg / max(total_extracted, 1) * 100
    print(f'CCG-eligible fraction:        {pct_ccg:.1f}%')
    print(f'\nTotal file size:              {total_size_mb:.1f} MB '
          f'({total_size_mb / 1024:.2f} GB)')
    print(f'Mean file size per session:   '
          f'{total_size_mb / max(n_ok, 1):.1f} MB')
    print(f'\nElapsed time:                 {elapsed_global:.1f}s')
    print(f'Output dir:                   {DERIVATIVES_DIR}/')
    print(f'Summary CSV:                  {SUMMARY_CSV}')

    if n_failed > 0:
        failed = summary_df[~summary_df['status'].str.startswith('ok')]
        print(f'\nFailed sessions:')
        for _, row in failed.iterrows():
            print(f'  {int(row["session_id"])}: {row["status"]}')


if __name__ == '__main__':
    main()
