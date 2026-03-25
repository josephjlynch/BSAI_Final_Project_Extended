"""
PFC Structure Query
===================

Checks the Allen Visual Behavior Neuropixels dataset for prefrontal cortex
(PFC) units and generates a structured three-source report on PFC data
availability for the project.

Usage (from BSAI_Final_Project_Extended/ with venv activated):
    python query_pfc_structures.py
"""

import sys
import os
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loading import load_cache

SESSION_IDS_FILE = 'session_ids.txt'
OUTPUT_FILE = 'results/tables/pfc_query_results.txt'

PFC_STRUCTURES = [
    'PL', 'ILA', 'ACAd', 'ACAv', 'ORBl', 'ORBm', 'ORBvl',
    'MOs', 'MOp', 'FRP'
]


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
    print("PFC STRUCTURE QUERY")
    print("=" * 70)

    cache = load_cache()
    session_ids = load_session_ids()

    unit_table = cache.get_unit_table()
    session_units = unit_table[unit_table['ecephys_session_id'].isin(session_ids)]

    print(f"\nTotal units across {len(session_ids)} sessions: {len(session_units)}")

    # --- Query for PFC structures ---
    pfc_mask = session_units['structure_acronym'].isin(PFC_STRUCTURES)
    pfc_units = session_units[pfc_mask]
    n_pfc = len(pfc_units)

    print(f"\nPFC structures queried: {PFC_STRUCTURES}")
    print(f"PFC units found: {n_pfc}")

    if n_pfc > 0:
        print("\nPFC unit breakdown:")
        for area, count in pfc_units['structure_acronym'].value_counts().items():
            print(f"  {area}: {count}")
    else:
        print("  -> CONFIRMED: Zero PFC units in the Visual Behavior Neuropixels dataset.")
        print("     (Probes enter through visual cortex craniotomy; PFC not targeted.)")

    # --- All unique structures for reference ---
    all_areas = session_units['structure_acronym'].value_counts().sort_values(ascending=False)
    print(f"\n--- All {len(all_areas)} unique structure_acronym values ---")
    for area, count in all_areas.items():
        print(f"  {area:>12s}: {count:>6d}")

    # --- Generate structured report ---
    report_lines = [
        f"PFC Data Availability Report",
        f"Generated: {date.today().isoformat()}",
        f"{'='*70}",
        f"",
        f"SOURCE 1: Allen Visual Behavior Neuropixels Dataset",
        f"{'-'*70}",
        f"Dataset: Allen Brain Observatory Visual Behavior Neuropixels (DANDI:000713)",
        f"Sessions queried: {len(session_ids)}",
        f"Total units: {len(session_units)}",
        f"PFC structures queried: {', '.join(PFC_STRUCTURES)}",
        f"PFC units found: {n_pfc}",
        f"",
        f"Result: {'NEGATIVE' if n_pfc == 0 else 'POSITIVE'} -- "
        f"{'No PFC units present. Probes enter through visual cortex craniotomy.' if n_pfc == 0 else f'{n_pfc} PFC units detected.'}",
        f"",
        f"",
        f"SOURCE 2: DANDI Archive",
        f"{'-'*70}",
        f"Identified dataset: dandiset/001260",
        f"URL: https://dandiarchive.org/dandiset/001260",
        f"Description: Only PFC dataset identified on DANDI with mouse",
        f"  electrophysiology recordings from prefrontal regions.",
        f"Access method: DANDI API / dandi download",
        f"Integration plan: Download and assess in Week 3; determine",
        f"  compatibility with Visual Behavior task paradigm.",
        f"",
        f"Status: IDENTIFIED -- requires download and compatibility assessment.",
        f"",
        f"",
        f"SOURCE 3: Gale et al. (2024) Backward Masking",
        f"{'-'*70}",
        f"Paper: Gale et al. (2024) backward masking study",
        f"Mouse line: VGAT-ChR2 mice with VISp silencing",
        f"PFC recordings: NONE -- study targets visual cortex silencing only.",
        f"",
        f"Result: NEGATIVE -- no PFC data available from this source.",
        f"",
        f"",
        f"NEXT STEPS",
        f"{'-'*70}",
        f"1. Download dandiset/001260 in Week 3",
        f"2. Assess whether task paradigm and recording sites are compatible",
        f"   with the Visual Behavior change detection framework",
        f"3. If compatible, integrate PFC data as an additional analysis",
        f"   dimension for the network neuroscience framework",
        f"4. If incompatible, document the limitation and focus on",
        f"   thalamo-cortical and midbrain circuits available in the",
        f"   Visual Behavior dataset",
    ]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write('\n'.join(report_lines) + '\n')

    print(f"\nReport saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
