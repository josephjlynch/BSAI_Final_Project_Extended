"""
Generate Session List
=====================

Queries the Allen Visual Behavior Neuropixels ecephys session table,
filters by Bennett et al. (2025) QC criteria, and writes the 103
clean session IDs to session_ids.txt.

Also includes a download loop for all sessions (re-runnable).

Usage:
    python generate_session_list.py [--download]
"""

import sys
import os
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.constants import N_SESSIONS_EXPECTED, N_MICE_EXPECTED
from src.data_loading import load_cache

SESSION_IDS_FILE = 'session_ids.txt'
DOWNLOAD_LOG_FILE = 'results/download_log.csv'

HEADER = """\
# Allen Brain Observatory Visual Behavior Neuropixels
# 103 Valid Session IDs (Bennett et al. QC-passed)
#
# Selection: All sessions without QC flags (abnormal_histology,
# abnormal_activity) from the 153 released sessions.
# 54 mice, 103 sessions. Most mice have 2 sessions (familiar + novel day).
#
# Data Source: DANDI:000713
# Citation: Bennett et al. (2025 preprint)
"""


def generate_session_list():
    print("Loading cache...")
    cache = load_cache()

    print("Querying ecephys session table...")
    table = cache.get_ecephys_session_table()
    print(f"  Total sessions in table: {len(table)}")

    clean = table[
        table['abnormal_histology'].isna() &
        table['abnormal_activity'].isna()
    ]

    n_sessions = len(clean)
    n_mice = clean['mouse_id'].nunique()
    print(f"  Clean sessions: {n_sessions}")
    print(f"  Unique mice: {n_mice}")

    assert n_sessions == N_SESSIONS_EXPECTED, (
        f"Expected {N_SESSIONS_EXPECTED} sessions, got {n_sessions}"
    )
    assert n_mice == N_MICE_EXPECTED, (
        f"Expected {N_MICE_EXPECTED} mice, got {n_mice}"
    )

    session_ids = sorted(clean.index.tolist())

    with open(SESSION_IDS_FILE, 'w') as f:
        f.write(HEADER)
        for sid in session_ids:
            f.write(f"{sid}\n")

    print(f"\nWrote {len(session_ids)} session IDs to {SESSION_IDS_FILE}")

    print("\n=== Genotype breakdown ===")
    print(clean['genotype'].value_counts().to_string())

    print("\n=== Experience level breakdown ===")
    print(clean['experience_level'].value_counts().to_string())

    return cache, session_ids


def download_sessions(cache, session_ids):
    from tqdm import tqdm

    os.makedirs('results', exist_ok=True)

    print(f"\nDownloading {len(session_ids)} sessions...")
    print("(Already-cached sessions will return immediately)")

    with open(DOWNLOAD_LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'status', 'error'])

        ok_count = 0
        fail_count = 0

        for sid in tqdm(session_ids, desc="Downloading sessions"):
            try:
                cache.get_ecephys_session(ecephys_session_id=sid)
                writer.writerow([sid, 'ok', ''])
                ok_count += 1
            except Exception as e:
                writer.writerow([sid, 'failed', str(e)])
                fail_count += 1
                print(f"\n  FAILED: {sid}: {e}")

    print(f"\nDownload complete: {ok_count} ok, {fail_count} failed")
    print(f"Log saved to {DOWNLOAD_LOG_FILE}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate session list and optionally download")
    parser.add_argument('--download', action='store_true',
                        help='Download all 103 sessions after generating list')
    args = parser.parse_args()

    cache, session_ids = generate_session_list()

    if args.download:
        download_sessions(cache, session_ids)
    else:
        print("\nTo download sessions, re-run with --download flag")
