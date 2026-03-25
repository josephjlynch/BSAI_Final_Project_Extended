"""
Generate Session Metadata Table
================================

For each of the 103 sessions, extracts performance metrics (d-prime,
hit_rate, false_alarm_rate), per-area unit counts, SDT cross-validation,
reaction time statistics, and subcortical coverage flags. Saves to
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
    ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX, EXPECTED_UNIT_COUNT
)
from src.data_loading import (
    load_cache, VISUAL_AREAS, THALAMIC_AREAS, MIDBRAIN_AREAS,
    HIPPOCAMPAL_AREAS, OTHER_SUBCORTICAL_AREAS, ALL_BENNETT_AREAS,
    get_change_detection_trials, label_sdt_category
)

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
        quality_mask = (
            (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
            (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
            (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
        )
        if 'quality' in session_units.columns:
            quality_mask = quality_mask & (session_units['quality'] == 'good')
        filtered = session_units[quality_mask]
        row['unit_count_filtered'] = len(filtered)

        # --- 2a. Per-area unit counts ---
        area_counts = filtered['structure_acronym'].value_counts()
        for area in ALL_BENNETT_AREAS:
            row[f'n_{area}'] = area_counts.get(area, 0)
        row['n_grey'] = area_counts.get('grey', 0)
        known_areas = set(ALL_BENNETT_AREAS) | {'grey'}
        row['n_other'] = sum(
            c for a, c in area_counts.items() if a not in known_areas
        )
        row['n_visual_total'] = sum(row.get(f'n_{a}', 0) for a in VISUAL_AREAS)

        # --- 2d. Subcortical flags ---
        row['has_thalamic'] = (row.get('n_LGd', 0) + row.get('n_LP', 0)) > 0
        row['has_midbrain'] = (row.get('n_SCm', 0) + row.get('n_MRN', 0)) > 0
        row['n_thalamic_total'] = row.get('n_LGd', 0) + row.get('n_LP', 0)
        row['n_midbrain_total'] = row.get('n_SCm', 0) + row.get('n_MRN', 0)

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

            # --- 2b. SDT cross-validation ---
            labeled = label_sdt_category(session.trials)
            sdt_counts = labeled['sdt_category'].value_counts()
            for sdt_name, perf_key in [
                ('hit', 'hit_trial_count'),
                ('miss', 'miss_trial_count'),
                ('false_alarm', 'false_alarm_trial_count'),
                ('correct_reject', 'correct_reject_trial_count'),
            ]:
                expected = row.get(f'perf_{perf_key}')
                actual = sdt_counts.get(sdt_name, 0)
                if expected is not None and actual != expected:
                    print(f"  WARNING session {sid}: {sdt_name} count mismatch: "
                          f"sdt_category={actual}, perf_metrics={expected}")

            # --- 2c. Reaction time statistics ---
            try:
                trial_data = get_change_detection_trials(session, engaged_only=True)
                engaged_trials = trial_data['engaged_trials']
                engaged_hits = engaged_trials[engaged_trials['sdt_category'] == 'hit']
                if len(engaged_hits) > 0:
                    has_rt = 'response_time' in engaged_hits.columns
                    has_ct = 'change_time_no_display_delay' in engaged_hits.columns
                    if has_rt and has_ct:
                        rt = (engaged_hits['response_time']
                              - engaged_hits['change_time_no_display_delay']).dropna()
                        if len(rt) > 0:
                            row['rt_median_s'] = rt.median()
                            row['rt_mean_s'] = rt.mean()
                            row['rt_std_s'] = rt.std()
            except Exception as e:
                row['rt_error'] = str(e)

        except Exception as e:
            row['perf_error'] = str(e)

        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved metadata for {len(df)} sessions to {OUTPUT_CSV}")

    # --- 2e. End-of-script structured summary ---
    print(f"\n{'='*70}")
    print("SESSION METADATA SUMMARY")
    print(f"{'='*70}")

    print(f"\nSessions processed: {len(df)}")

    area_cols = [c for c in df.columns if c.startswith('n_') and c != 'n_other']
    if area_cols:
        print(f"\n--- Unit counts per area (summed across {len(df)} sessions) ---")
        for col in sorted(area_cols):
            total = df[col].sum()
            print(f"  {col:>20s}: {int(total):>7d}")
        if 'n_grey' in df.columns:
            total_grey = int(df['n_grey'].sum())
            total_all = int(df['unit_count_filtered'].sum())
            print(f"\n  Total filtered:       {total_all}")
            print(f"  Total grey:           {total_grey}")
            print(f"  Total excl grey:      {total_all - total_grey}")
            print(f"  Bennett reports:      {EXPECTED_UNIT_COUNT}")

    if 'has_thalamic' in df.columns:
        n_thal = df['has_thalamic'].sum()
        n_mid = df['has_midbrain'].sum()
        print(f"\n--- Subcortical coverage ---")
        print(f"  Sessions with thalamic units:  {int(n_thal)}/{len(df)}")
        print(f"  Sessions with midbrain units:  {int(n_mid)}/{len(df)}")

    rt_col = 'rt_median_s'
    if rt_col in df.columns:
        rt_vals = df[rt_col].dropna()
        if len(rt_vals) > 0:
            print(f"\n--- Reaction time (engaged hit trials) ---")
            print(f"  Grand median across sessions: {rt_vals.median():.3f} s")
            print(f"  IQR: [{rt_vals.quantile(0.25):.3f}, "
                  f"{rt_vals.quantile(0.75):.3f}] s")
            print(f"  Range: [{rt_vals.min():.3f}, {rt_vals.max():.3f}] s")

    n_errors = (
        df['perf_error'].notna().sum() if 'perf_error' in df.columns else 0
    )
    print(f"\n--- Data quality ---")
    print(f"  Sessions with perf_error: {n_errors}/{len(df)}")

    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")


if __name__ == '__main__':
    main()
