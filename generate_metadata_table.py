"""
Generate Session Metadata Table
================================

For each of the 103 sessions, extracts performance metrics (d-prime,
hit_rate, false_alarm_rate) and session-level metadata. Saves to
results/tables/session_metadata.csv (Table S1 of the paper).

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python generate_metadata_table.py
"""

import sys
import os

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    N_SESSIONS_EXPECTED, PRESENCE_RATIO_MIN,
    ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX
)
from src.data_loading import load_cache

SESSION_IDS_FILE = 'session_ids.txt'
OUTPUT_CSV = 'results/tables/session_metadata.csv'


def load_session_ids():
    ids = []
    with open(SESSION_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(int(line))
    return ids


def main():
    print("Loading cache...")
    cache = load_cache()

    session_ids = load_session_ids()
    print(f"Loaded {len(session_ids)} session IDs")

    session_table = cache.get_ecephys_session_table()
    unit_table = cache.get_unit_table()

    rows = []

    for sid in tqdm(session_ids, desc="Generating metadata"):
        row = {'ecephys_session_id': sid}

        if sid in session_table.index:
            st = session_table.loc[sid]
            for col in ['mouse_id', 'genotype', 'sex', 'age_in_days',
                        'experience_level', 'session_number', 'image_set',
                        'equipment_name', 'project_code']:
                if col in session_table.columns:
                    row[col] = st[col]
            if 'unit_count' in session_table.columns:
                row['unit_count_raw'] = st['unit_count']
            if 'structure_acronyms' in session_table.columns:
                val = st['structure_acronyms']
                row['structure_acronyms'] = (
                    ', '.join(sorted(val)) if isinstance(val, (list, set)) else str(val)
                )

        session_units = unit_table[unit_table['ecephys_session_id'] == sid]
        filtered = session_units[
            (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
            (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
            (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
        ]
        row['unit_count_filtered'] = len(filtered)

        try:
            session = cache.get_ecephys_session(ecephys_session_id=sid)
            perf = session.get_performance_metrics()
            if isinstance(perf, dict):
                for key, val in perf.items():
                    row[f'perf_{key}'] = val
            elif isinstance(perf, pd.DataFrame):
                for col in perf.columns:
                    row[f'perf_{col}'] = perf[col].iloc[0] if len(perf) > 0 else None
            elif isinstance(perf, pd.Series):
                for key, val in perf.items():
                    row[f'perf_{key}'] = val
        except Exception as e:
            row['perf_error'] = str(e)

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved metadata for {len(df)} sessions to {OUTPUT_CSV}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == '__main__':
    main()
