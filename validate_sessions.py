"""
Validate Sessions + Column Discovery
=====================================

Performs two tasks:
1. Column discovery: prints all available columns from unit_table,
   stimulus_presentations, and trials for documentation in ANALYSIS_LOG.md.
2. Validation loop: opens each of the 103 sessions and checks data integrity.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python validate_sessions.py
"""

import sys
import os
import csv

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    N_SESSIONS_EXPECTED, N_MICE_EXPECTED,
    PRESENCE_RATIO_MIN, ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX,
    EXPECTED_UNIT_COUNT, RS_FS_THRESHOLD_MS
)
from src.data_loading import load_cache

SESSION_IDS_FILE = 'session_ids.txt'
VALIDATION_CSV = 'results/session_validation.csv'


def load_session_ids():
    ids = []
    with open(SESSION_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(int(line))
    return ids


def column_discovery(cache, session_ids):
    """Print all available columns for documentation."""
    print("=" * 70)
    print("COLUMN DISCOVERY")
    print("=" * 70)

    unit_table = cache.get_unit_table()
    print("\n=== UNIT TABLE COLUMNS ===")
    print(unit_table.columns.tolist())

    assert 'waveform_duration' in unit_table.columns, \
        "waveform_duration column missing from unit_table"
    print("\n[OK] waveform_duration confirmed")

    assert 'presence_ratio' in unit_table.columns
    assert 'isi_violations' in unit_table.columns
    assert 'amplitude_cutoff' in unit_table.columns
    print("[OK] presence_ratio, isi_violations, amplitude_cutoff confirmed")

    if 'quality' in unit_table.columns:
        print(f"\nquality column values:")
        print(unit_table['quality'].value_counts())
    else:
        print("\nquality column: NOT PRESENT")

    opto_cols = [c for c in unit_table.columns if 'opto' in c.lower()
                 or 'cell_type' in c.lower() or 'tagged' in c.lower()]
    print(f"\nOptotagging columns: {opto_cols if opto_cols else 'NONE'}")

    session_table = cache.get_ecephys_session_table()
    clean = session_table[
        session_table['abnormal_histology'].isna() &
        session_table['abnormal_activity'].isna()
    ]
    print(f"\n=== Genotype distribution ({len(clean)} sessions) ===")
    print(clean['genotype'].value_counts().to_string())

    first_sid = session_ids[0]
    print(f"\nLoading session {first_sid} for column inspection...")
    session = cache.get_ecephys_session(ecephys_session_id=first_sid)

    print("\n=== STIMULUS_PRESENTATIONS COLUMNS ===")
    print(session.stimulus_presentations.columns.tolist())

    print("\n=== TRIALS COLUMNS ===")
    print(session.trials.columns.tolist())

    print("\n=== SESSION PERFORMANCE METRICS ===")
    perf = session.get_performance_metrics()
    print(perf)

    clean_units = unit_table[
        unit_table['ecephys_session_id'].isin(session_ids)
    ]
    filtered = clean_units[
        (clean_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
        (clean_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
        (clean_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
    ]
    total_units = len(filtered)
    print(f"\n=== UNIT QUALITY SUMMARY ===")
    print(f"Total units (raw) across {len(session_ids)} sessions: {len(clean_units)}")
    print(f"Total units (Bennett-filtered): {total_units}")
    print(f"Bennett reports: {EXPECTED_UNIT_COUNT}")

    if 'waveform_duration' in filtered.columns:
        rs_count = (filtered['waveform_duration'] > RS_FS_THRESHOLD_MS).sum()
        fs_count = (filtered['waveform_duration'] <= RS_FS_THRESHOLD_MS).sum()
        print(f"\nRS/FS distribution (Bennett-filtered):")
        print(f"  RS (>{RS_FS_THRESHOLD_MS}ms): {rs_count} ({100*rs_count/total_units:.1f}%)")
        print(f"  FS (<={RS_FS_THRESHOLD_MS}ms): {fs_count} ({100*fs_count/total_units:.1f}%)")

    return {
        'unit_table_columns': unit_table.columns.tolist(),
        'stim_columns': session.stimulus_presentations.columns.tolist(),
        'trials_columns': session.trials.columns.tolist(),
        'opto_cols': opto_cols,
        'genotype_counts': clean['genotype'].value_counts().to_dict(),
        'total_units_filtered': total_units,
        'rs_count': rs_count if 'waveform_duration' in filtered.columns else None,
        'fs_count': fs_count if 'waveform_duration' in filtered.columns else None,
        'performance_metrics': perf,
    }


def validate_sessions(cache, session_ids):
    """Open each session and check data integrity."""
    print("\n" + "=" * 70)
    print("SESSION VALIDATION LOOP")
    print("=" * 70)

    unit_table = cache.get_unit_table()
    os.makedirs('results', exist_ok=True)

    results = []
    passed = 0
    failed = 0
    failed_ids = []

    for i, sid in enumerate(session_ids):
        status = 'pass'
        errors = []
        unit_count = 0

        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)

            spike_times = session.spike_times
            if len(spike_times) == 0:
                errors.append('spike_times_empty')

            stim = session.stimulus_presentations
            for col in ['is_change', 'active', 'omitted']:
                if col not in stim.columns:
                    errors.append(f'stim_missing_{col}')

            trials = session.trials
            for col in ['hit', 'miss', 'false_alarm', 'correct_reject']:
                if col not in trials.columns:
                    errors.append(f'trials_missing_{col}')

            session_units = unit_table[
                unit_table['ecephys_session_id'] == sid
            ]
            filtered_units = session_units[
                (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
                (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
                (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
            ]
            unit_count = len(filtered_units)
            if unit_count == 0:
                errors.append('zero_units_after_filter')

        except Exception as e:
            errors.append(f'load_error: {str(e)}')

        if errors:
            status = 'fail'
            failed += 1
            failed_ids.append(sid)
        else:
            passed += 1

        results.append({
            'session_id': sid,
            'status': status,
            'unit_count': unit_count,
            'errors': '; '.join(errors) if errors else '',
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(session_ids):
            print(f"  Validated {i+1}/{len(session_ids)} "
                  f"(passed={passed}, failed={failed})")

    with open(VALIDATION_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['session_id', 'status', 'unit_count', 'errors'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Passed: {passed}/{len(session_ids)}")
    print(f"Failed: {failed}/{len(session_ids)}")
    if failed_ids:
        print(f"Failed IDs: {failed_ids}")
    print(f"Results saved to {VALIDATION_CSV}")

    return results


if __name__ == '__main__':
    print("Loading cache...")
    cache = load_cache()

    session_ids = load_session_ids()
    print(f"Loaded {len(session_ids)} session IDs from {SESSION_IDS_FILE}")

    discovery = column_discovery(cache, session_ids)
    validation = validate_sessions(cache, session_ids)
