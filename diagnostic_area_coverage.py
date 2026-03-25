"""
Diagnostic: Area coverage across sessions for CCG network analysis
===================================================================

Answers Disheng's directive: "it'll be better if most of your sessions
have the same areas."

For each of the 103 sessions, determines which brain areas have at least
one CCG-eligible unit (Bennett 4-criteria filter + 2Hz firing rate threshold).
Groups sessions by area set, cross-references with mouse familiar/novel
pairing, and identifies core areas for consistent network comparison.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python diagnostic_area_coverage.py
"""

import sys
import os
from collections import Counter

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import (
    N_SESSIONS_EXPECTED, PRESENCE_RATIO_MIN,
    ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX,
    CCG_MIN_FIRING_RATE_HZ
)
from src.data_loading import (
    load_cache, get_mouse_session_map,
    VISUAL_AREAS, THALAMIC_AREAS, MIDBRAIN_AREAS, HIPPOCAMPAL_AREAS
)

SESSION_IDS_FILE = 'session_ids.txt'
OUTPUT_CSV = 'results/tables/area_coverage_by_session.csv'


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
    print("AREA COVERAGE DIAGNOSTIC (CCG-eligible units)")
    print("=" * 70)

    cache = load_cache()
    session_ids = load_session_ids()
    assert len(session_ids) == N_SESSIONS_EXPECTED

    print("Loading unit table...", flush=True)
    unit_table = cache.get_unit_table()
    session_units = unit_table[unit_table['ecephys_session_id'].isin(session_ids)]
    print(f"Raw units across {len(session_ids)} sessions: {len(session_units)}")

    # --- Bennett 4-criteria quality filter ---
    quality_mask = (
        (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
        (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
        (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
    )
    if 'quality' in session_units.columns:
        quality_mask = quality_mask & (session_units['quality'] == 'good')
    filtered = session_units[quality_mask].copy()
    print(f"Bennett-filtered units: {len(filtered)}")

    # --- 2Hz firing rate filter ---
    if 'firing_rate' in filtered.columns:
        fr_col = 'firing_rate'
    else:
        fr_col = None
        print("WARNING: 'firing_rate' column not in unit table. "
              "Skipping 2Hz filter (all Bennett-filtered units included).")

    if fr_col:
        n_before = len(filtered)
        filtered = filtered[filtered[fr_col] >= CCG_MIN_FIRING_RATE_HZ].copy()
        print(f"After 2Hz FR filter: {len(filtered)} "
              f"({n_before - len(filtered)} removed)")

    # --- Mouse-session mapping ---
    session_map = get_mouse_session_map(cache)
    session_map = session_map[
        session_map['ecephys_session_id'].isin(session_ids)
    ]
    sid_to_mouse = dict(zip(
        session_map['ecephys_session_id'], session_map['mouse_id']))
    sid_to_exp = dict(zip(
        session_map['ecephys_session_id'],
        session_map.get('experience_level', pd.Series(dtype=str))))

    # --- Per-session area sets ---
    rows = []
    for sid in session_ids:
        s_units = filtered[filtered['ecephys_session_id'] == sid]
        areas = sorted(s_units['structure_acronym'].unique())
        area_set = frozenset(areas)

        n_units = len(s_units)
        visual_present = [a for a in VISUAL_AREAS if a in area_set]
        thalamic_present = [a for a in THALAMIC_AREAS if a in area_set]
        midbrain_present = [a for a in MIDBRAIN_AREAS if a in area_set]

        rows.append({
            'session_id': sid,
            'mouse_id': sid_to_mouse.get(sid, ''),
            'experience_level': sid_to_exp.get(sid, ''),
            'n_units_ccg': n_units,
            'n_areas': len(areas),
            'area_set': '|'.join(areas),
            'has_all_visual': len(visual_present) == len(VISUAL_AREAS),
            'n_visual': len(visual_present),
            'has_thalamic': len(thalamic_present) > 0,
            'has_midbrain': len(midbrain_present) > 0,
        })

    df = pd.DataFrame(rows)

    # --- Save CSV ---
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved per-session area coverage to {OUTPUT_CSV}")

    # --- Group sessions by area set ---
    print(f"\n{'='*70}")
    print("AREA SET GROUPING")
    print(f"{'='*70}")

    area_set_groups = df.groupby('area_set').size().sort_values(ascending=False)
    print(f"Distinct area sets: {len(area_set_groups)}")
    print(f"\nTop area sets by session count:")
    for i, (aset, count) in enumerate(area_set_groups.items()):
        if i >= 10:
            print(f"  ... and {len(area_set_groups) - 10} more")
            break
        areas_short = aset if len(aset) < 80 else aset[:77] + "..."
        print(f"  {count:>3d} sessions: {areas_short}")

    # --- Per-area frequency ---
    print(f"\n{'='*70}")
    print("PER-AREA FREQUENCY (CCG-eligible units, >=1 unit in session)")
    print(f"{'='*70}")

    all_areas = []
    for _, row in df.iterrows():
        all_areas.extend(row['area_set'].split('|'))
    area_freq = Counter(all_areas)
    area_freq_sorted = sorted(area_freq.items(), key=lambda x: -x[1])

    for area, count in area_freq_sorted:
        pct = count / len(session_ids) * 100
        marker = ""
        if area in VISUAL_AREAS:
            marker = " [visual]"
        elif area in THALAMIC_AREAS:
            marker = " [thalamic]"
        elif area in MIDBRAIN_AREAS:
            marker = " [midbrain]"
        elif area in HIPPOCAMPAL_AREAS:
            marker = " [hippocampal]"
        print(f"  {area:>12s}: {count:>3d}/{len(session_ids)} ({pct:5.1f}%){marker}")

    # --- Core areas (>= 90% of sessions) ---
    core_areas = [a for a, c in area_freq_sorted if c / len(session_ids) >= 0.9]
    print(f"\nCore areas (>=90% of sessions): {core_areas}")

    # --- Visual area coverage ---
    print(f"\n{'='*70}")
    print("VISUAL AREA COVERAGE")
    print(f"{'='*70}")
    n_all_visual = df['has_all_visual'].sum()
    print(f"Sessions with all 6 visual areas: {n_all_visual}/{len(session_ids)}")
    for nv in sorted(df['n_visual'].unique(), reverse=True):
        count = (df['n_visual'] == nv).sum()
        print(f"  {nv} visual areas: {count} sessions")

    # --- Mouse pair consistency ---
    print(f"\n{'='*70}")
    print("MOUSE PAIR AREA CONSISTENCY")
    print(f"{'='*70}")

    mice = df.groupby('mouse_id')
    n_mice_total = 0
    n_mice_matched = 0
    n_mice_mismatched = 0
    mismatched_mice = []

    for mouse_id, group in mice:
        if len(group) < 2:
            continue
        n_mice_total += 1
        area_sets = group['area_set'].unique()
        if len(area_sets) == 1:
            n_mice_matched += 1
        else:
            n_mice_mismatched += 1
            mismatched_mice.append(mouse_id)

    print(f"Mice with paired sessions: {n_mice_total}")
    print(f"Mice with matching area sets: {n_mice_matched}")
    print(f"Mice with mismatched area sets: {n_mice_mismatched}")
    if mismatched_mice and len(mismatched_mice) <= 10:
        for mid in mismatched_mice:
            mdata = df[df['mouse_id'] == mid][['session_id', 'experience_level', 'area_set']]
            print(f"\n  Mouse {mid}:")
            for _, r in mdata.iterrows():
                print(f"    {r['experience_level']}: {r['area_set']}")

    # --- Subcortical coverage ---
    print(f"\n{'='*70}")
    print("SUBCORTICAL COVERAGE")
    print(f"{'='*70}")
    n_thal = df['has_thalamic'].sum()
    n_mid = df['has_midbrain'].sum()
    n_both = ((df['has_thalamic']) & (df['has_midbrain'])).sum()
    n_neither = ((~df['has_thalamic']) & (~df['has_midbrain'])).sum()
    print(f"Sessions with thalamic (LGd/LP):     {n_thal}/{len(session_ids)}")
    print(f"Sessions with midbrain (SCm/MRN):     {n_mid}/{len(session_ids)}")
    print(f"Sessions with both:                   {n_both}/{len(session_ids)}")
    print(f"Sessions with neither:                {n_neither}/{len(session_ids)}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total sessions: {len(session_ids)}")
    print(f"CCG-eligible units (Bennett + 2Hz): {len(filtered)}")
    print(f"Distinct area sets: {len(area_set_groups)}")
    print(f"Core areas (>=90%): {core_areas}")
    print(f"Sessions with all 6 visual: {n_all_visual}/{len(session_ids)}")
    print(f"Mouse pair consistency: {n_mice_matched}/{n_mice_total} matched")
    print(f"Subcortical: {n_thal} thalamic, {n_mid} midbrain, "
          f"{n_neither} neither")


if __name__ == '__main__':
    main()
