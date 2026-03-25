"""
Diagnostic: Per-area unit count verification
=============================================
GATE STEP -- must run before any Week 2 code changes.
Resolves the 82,133 vs 76,091 (Bennett) unit count discrepancy.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python diagnostic_unit_counts.py
"""

import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    N_SESSIONS_EXPECTED, PRESENCE_RATIO_MIN,
    ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX, EXPECTED_UNIT_COUNT
)
from src.data_loading import (
    load_cache, VISUAL_AREAS, THALAMIC_AREAS,
    MIDBRAIN_AREAS, HIPPOCAMPAL_AREAS
)

SESSION_IDS_FILE = 'session_ids.txt'

NAMED_AREAS = (
    VISUAL_AREAS + THALAMIC_AREAS + MIDBRAIN_AREAS + HIPPOCAMPAL_AREAS
)


def load_session_ids():
    ids = []
    with open(SESSION_IDS_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                ids.append(int(line))
    return ids


def main():
    print("=" * 70)
    print("GATE DIAGNOSTIC: Per-area unit count verification")
    print("=" * 70)

    cache = load_cache()
    session_ids = load_session_ids()
    assert len(session_ids) == N_SESSIONS_EXPECTED

    unit_table = cache.get_unit_table()
    session_units = unit_table[unit_table['ecephys_session_id'].isin(session_ids)]
    print(f"\nRaw units across {len(session_ids)} sessions: {len(session_units)}")

    filtered = session_units[
        (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
        (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
        (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
    ]
    total_filtered = len(filtered)
    print(f"Bennett-filtered units: {total_filtered}")
    print(f"Bennett reports:        {EXPECTED_UNIT_COUNT}")
    print(f"Discrepancy:            {total_filtered - EXPECTED_UNIT_COUNT}")

    area_counts = filtered['structure_acronym'].value_counts().sort_values(
        ascending=False
    )
    print(f"\n{'='*70}")
    print(f"ALL UNIQUE STRUCTURE ACRONYMS ({len(area_counts)} structures)")
    print(f"{'='*70}")
    for area, count in area_counts.items():
        marker = ""
        if area == 'grey':
            marker = "  <-- GREY (unannotated)"
        elif area not in NAMED_AREAS:
            marker = "  <-- not in our constants"
        print(f"  {area:>12s}: {count:>6d}{marker}")

    n_grey = area_counts.get('grey', 0)
    n_named = filtered[filtered['structure_acronym'] != 'grey'].shape[0]
    print(f"\n{'='*70}")
    print("DISCREPANCY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total filtered:            {total_filtered}")
    print(f"Units in 'grey':           {n_grey}")
    print(f"Units in named areas:      {n_named}")
    print(f"Bennett's count:           {EXPECTED_UNIT_COUNT}")
    print(f"Ours minus grey:           {n_named}")
    print(f"Difference from Bennett:   {n_named - EXPECTED_UNIT_COUNT}")

    pct_diff = abs(n_named - EXPECTED_UNIT_COUNT) / EXPECTED_UNIT_COUNT * 100
    if pct_diff < 1.0:
        print(f"\n>> EXPLANATION A CONFIRMED: excluding 'grey' brings count to "
              f"{n_named} ({pct_diff:.2f}% from Bennett's {EXPECTED_UNIT_COUNT}).")
        print(">> Proceed with Week 2 changes.")
    elif pct_diff < 5.0:
        print(f"\n>> PARTIAL MATCH: {pct_diff:.2f}% difference after excluding grey.")
        print(">> Check quality column (Bennett: 'non-noise units').")
    else:
        print(f"\n>> Grey alone does not explain the gap ({pct_diff:.2f}%).")
        print(">> Checking quality column (Bennett: 'non-noise units')...")

    # --- Quality column check (Bennett: "non-noise units") ---
    print(f"\n{'='*70}")
    print("QUALITY COLUMN ANALYSIS (Bennett: 'non-noise units')")
    print(f"{'='*70}")
    if 'quality' in filtered.columns:
        q_counts = filtered['quality'].value_counts()
        for q_val, q_n in q_counts.items():
            print(f"  quality == '{q_val}': {q_n}")

        good_filtered = session_units[
            (session_units['quality'] == 'good') &
            (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
            (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
            (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
        ]
        n_good = len(good_filtered)
        n_good_no_grey = len(
            good_filtered[good_filtered['structure_acronym'] != 'grey']
        )
        pct_good = abs(n_good - EXPECTED_UNIT_COUNT) / EXPECTED_UNIT_COUNT * 100
        print(f"\n  3-criteria + quality=='good':  {n_good}")
        print(f"  3-criteria + good, excl grey:  {n_good_no_grey}")
        print(f"  Bennett reports:               {EXPECTED_UNIT_COUNT}")
        print(f"  Difference (with grey):        {n_good - EXPECTED_UNIT_COUNT} "
              f"({pct_good:.2f}%)")

        if pct_good < 1.0:
            print(f"\n>> RESOLUTION: Adding quality=='good' filter brings count to "
                  f"{n_good} ({pct_good:.2f}% from Bennett).")
            print(">> Bennett's 'non-noise' criterion = quality == 'good'.")
            print(">> ACTION: Update get_units_with_areas() to include quality filter.")
            print(">> Proceed with Week 2 changes.")
        else:
            print(f"\n>> WARNING: {pct_good:.2f}% difference persists even with "
                  "quality=='good' filter. Investigate further.")
    else:
        print("  quality column NOT FOUND in unit table.")

    print(f"\n{'='*70}")
    print("THRESHOLD SENSITIVITY (> vs >=)")
    print(f"{'='*70}")
    for label, filt in [
        ("Current (strict >/>/</<)", filtered),
        ("presence_ratio >= 0.9", session_units[
            (session_units['presence_ratio'] >= PRESENCE_RATIO_MIN) &
            (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
            (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
        ]),
        ("isi_violations <= 0.5", session_units[
            (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
            (session_units['isi_violations'] <= ISI_VIOLATIONS_MAX) &
            (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
        ]),
        ("amplitude_cutoff <= 0.1", session_units[
            (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
            (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
            (session_units['amplitude_cutoff'] <= AMPLITUDE_CUTOFF_MAX)
        ]),
    ]:
        n = len(filt)
        n_no_grey = len(filt[filt['structure_acronym'] != 'grey'])
        print(f"  {label:>35s}: {n:>6d} total, {n_no_grey:>6d} excl grey")

    print(f"\n{'='*70}")
    print("SUBCORTICAL COVERAGE")
    print(f"{'='*70}")
    for sid in session_ids:
        s_units = filtered[filtered['ecephys_session_id'] == sid]
        areas = set(s_units['structure_acronym'].unique())
        has_thal = bool(areas & set(THALAMIC_AREAS))
        has_mid = bool(areas & set(MIDBRAIN_AREAS))
        if not has_thal or not has_mid:
            missing = []
            if not has_thal:
                missing.append("thalamic")
            if not has_mid:
                missing.append("midbrain")
            print(f"  Session {sid}: MISSING {', '.join(missing)}")

    n_with_thal = sum(
        1 for sid in session_ids
        if set(
            filtered[filtered['ecephys_session_id'] == sid][
                'structure_acronym'
            ].unique()
        ) & set(THALAMIC_AREAS)
    )
    n_with_mid = sum(
        1 for sid in session_ids
        if set(
            filtered[filtered['ecephys_session_id'] == sid][
                'structure_acronym'
            ].unique()
        ) & set(MIDBRAIN_AREAS)
    )
    print(f"\n  Sessions with thalamic units: {n_with_thal}/{len(session_ids)}")
    print(f"  Sessions with midbrain units: {n_with_mid}/{len(session_ids)}")


if __name__ == '__main__':
    main()
