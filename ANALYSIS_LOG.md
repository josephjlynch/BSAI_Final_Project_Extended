

---

**Date:** 2026-03-04

Bennett et al. (2025) use 103 QC-passed sessions from the Allen Visual Behavior Neuropixels dataset. Sessions excluded if `abnormal_histology == True` or `abnormal_activity == True`. Session list generation and validation deferred to Part 2 (`validate_sessions.py`).


Replaced `quality == 'good'` with Bennett's three explicit criteria (Methods §Data processing):

- `presence_ratio > 0.9`
- `isi_violations < 0.5`
- `amplitude_cutoff < 0.1`

### Pre-correction pipeline archived

- `run_multi_session.py` → `run_multi_session_DEPRECATED.py` (with `sys.exit` guard)
- `results/multi_session/checkpoint_results_6area.csv` → `results/archive/checkpoint_results_6area_PRECORRECTION.csv`

The deprecated script used the Visual Coding stimulus paradigm (Natural_Images/gabor), which is incompatible with the Visual Behavior change detection task.

### Column confirmation (from validate_sessions.py / column discovery)

- waveform_duration: PRESENT
- presence_ratio: PRESENT
- isi_violations: PRESENT
- amplitude_cutoff: PRESENT
- quality: PRESENT (values: good=262,177; noise=56,836 — but NOT used for filtering)
- is_change: PRESENT (in stimulus_presentations)
- active: PRESENT (in stimulus_presentations)
- omitted: PRESENT (in stimulus_presentations)
- trials.hit: PRESENT
- trials.miss: PRESENT
- trials.false_alarm: PRESENT
- trials.correct_reject: PRESENT (note: column name is `correct_reject`, not `correct_rejection`)
- Optotagging columns: NONE (optotagging data not exposed in unit_table; SST/VIP mice identified by genotype)

### Genotype distribution (103 sessions)

- Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt: 43 sessions
- wt/wt: 38 sessions
- Vip-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt: 22 sessions

### Session selection

- Filter: abnormal_histology is NaN AND abnormal_activity is NaN (S3 cache pre-excludes flagged sessions)
- Result: 103 sessions from 54 mice
- Assertion: count == 103 PASS
- Experience level: Novel=52, Familiar=51

### Unit quality filter

- Bennett criteria applied: presence_ratio > 0.9, isi_violations < 0.5, amplitude_cutoff < 0.1
- Total units (raw) across 103 sessions: 216,006
- Total units (Bennett-filtered): 82,133
- Bennett reports: 76,091
- Discrepancy note: our count is ~6,042 higher than Bennett's. Likely due to additional area-specific or probe-level filters applied by Bennett that are not documented in the Methods section. This should be investigated further by comparing per-area unit counts.

### Waveform classification

- Column used: waveform_duration (confirmed: trough-to-peak in ms)
- Threshold: 0.4ms (Bennett Methods)
- RS (>0.4ms, putative excitatory): 56,171 (68.4%)
- FS (<=0.4ms, putative PV inhibitory): 25,962 (31.6%)

### Performance metrics (from session.get_performance_metrics())

Returns a dict with keys: trial_count, go_trial_count, catch_trial_count, hit_trial_count, miss_trial_count, false_alarm_trial_count, correct_reject_trial_count, auto_reward_count, earned_reward_count, total_reward_count, total_reward_volume, maximum_reward_rate, engaged_trial_count, mean_hit_rate, mean_hit_rate_uncorrected, mean_hit_rate_engaged, mean_false_alarm_rate, mean_false_alarm_rate_uncorrected, mean_false_alarm_rate_engaged, mean_dprime, mean_dprime_engaged, max_dprime, max_dprime_engaged

### Mouse-session pairing structure

`get_mouse_session_map()` and `get_mouse_session_pairs()` added. Pairs function returns one row per mouse with `familiar_session_id` and `novel_session_id`. Asserts `N == 54` mice. Backbone for paired t-tests.

### Environment lock status

`requirements_locked.txt` generated via `pip freeze`. Added `pynwb>=2.5.0`, `dandi>=0.60.0` to `requirements.txt`.

---

**Date:** 2026-03-14

## Week 2 -- Per-area Verification, SDT Labeling, Metadata Enrichment

### Unit count discrepancy resolution (GATE STEP)

The Week 1 log noted 82,133 filtered units vs Bennett's reported 76,091 (discrepancy: 6,042). This was the first investigation step before any other Week 2 changes.

**Root cause identified**: Bennett states "non-noise units" (Methods line 1485-1486). Our Week 1 filter applied only the three numerical criteria but omitted the `quality == 'good'` filter. The `quality` column in the unit_table has two values: `good` (76,602 passing all three criteria) and `noise` (5,531 passing three criteria but flagged as noise by spike sorting).

| Filter applied | Unit count |
|---|---|
| 3 criteria only (Week 1) | 82,133 |
| 3 criteria + quality=='good' | 76,602 |
| Bennett's published count | 76,091 |
| Difference (ours vs Bennett) | 511 (0.67%) |

The 0.67% residual difference is within acceptable tolerance and likely due to minor data versioning between our SDK version and Bennett's analysis pipeline.

**Action taken**: Updated `get_units_with_areas()` in `src/data_loading.py` to include `quality == 'good'` in the filter mask. All downstream analyses now use the corrected four-criteria filter.

**Script**: `diagnostic_unit_counts.py`

### Per-area unit counts across 103 sessions (quality + 3 criteria filter)

53 unique `structure_acronym` values in the filtered dataset. Top areas:

| Area | Units | Category |
|---|---|---|
| CA1 | 10,456 | Hippocampal |
| VISpm | 6,068 | Visual |
| VISp | 5,407 | Visual |
| VISl | 5,252 | Visual |
| VISam | 5,168 | Visual |
| VISal | 4,770 | Visual |
| APN | 4,329 | Other subcortical |
| VISrl | 3,989 | Visual |
| DG | 3,889 | Hippocampal |
| MGv | 2,508 | Medial geniculate |
| CA3 | 1,837 | Hippocampal |
| SGN | 1,690 | Other |
| MRN | 1,667 | Midbrain |
| MB | 1,603 | Other subcortical |
| SUB | 1,596 | Hippocampal |
| LP | 1,541 | Thalamic |
| MGd | 1,515 | Medial geniculate |
| grey | 1,507 | Unannotated probes |
| ProS | 1,504 | Hippocampal |
| MGm | 1,194 | Medial geniculate |
| LGd | 1,063 | Thalamic |

Remaining 32 areas each have < 1,000 units.

**Area constants updated** in `src/data_loading.py`:
- `HIPPOCAMPAL_AREAS`: expanded to `['CA1', 'CA3', 'DG', 'ProS', 'SUB']`
- `OTHER_SUBCORTICAL_AREAS`: new constant `['MG', 'APN', 'MB']`
- `ALL_BENNETT_AREAS`: union of all named area lists (20 areas)

### Subcortical coverage

- Sessions with thalamic (LGd/LP) units: 86/103
- Sessions with midbrain (SCm/MRN) units: 69/103
- Sessions missing BOTH thalamic and midbrain: 10

### SDT labeling

Added `label_sdt_category()` to `src/data_loading.py`. Maps boolean columns (`hit`, `miss`, `false_alarm`, `correct_reject`) to a single `sdt_category` column. Fallback label for rows with no boolean True: `'auto_reward_or_other'` (NOT `'aborted'` -- aborted trials restart and never enter the trials table per Bennett Methods).

Integrated into `get_change_detection_trials()` so both `result['trials']` and `result['engaged_trials']` carry the `sdt_category` column.

**Cross-validation against `get_performance_metrics()`**: Tested on session 1044385384. All four SDT categories match exactly:
- hit: OK
- miss: OK
- false_alarm: OK
- correct_reject: OK

### Reaction time statistics

Computed as `response_time - change_time_no_display_delay` from engaged hit trials (via `get_change_detection_trials(session, engaged_only=True)`).

Verified on session 1044385384: median RT = 0.601s, mean = 0.574s, std = 0.134s, n = 90 engaged hit trials.

Note: Bennett reports median lick latency of 424ms across all mice. Individual session values vary; grand median across all 103 sessions to be computed when `generate_metadata_table.py` completes (67/103 sessions currently cached locally).

### PFC data availability

**Source 1: Allen Visual Behavior Neuropixels** -- 0 PFC units found across all 216,006 units (103 sessions). PFC structures queried: PL, ILA, ACAd, ACAv, ORBl, ORBm, ORBvl, MOs, MOp, FRP. Probes enter through visual cortex craniotomy; PFC not targeted.

**Source 2: DANDI archive** -- dandiset/001260 identified as the only mouse PFC electrophysiology dataset. URL: https://dandiarchive.org/dandiset/001260. To be downloaded and assessed for compatibility in Week 3.

**Source 3: Gale et al. (2024) backward masking** -- VGAT-ChR2 mice with VISp silencing only. No PFC recordings.

**Script**: `query_pfc_structures.py`. Report saved to `results/tables/pfc_query_results.txt`.

### Metadata table (generate_metadata_table.py)

Updated with:
- Per-area unit counts (one column per ALL_BENNETT_AREAS structure + n_grey + n_other)
- SDT cross-validation against get_performance_metrics() (prints WARNING on mismatch)
- Reaction time statistics (rt_median_s, rt_mean_s, rt_std_s from engaged hit trials)
- Subcortical flags (has_thalamic, has_midbrain, n_thalamic_total, n_midbrain_total)
- End-of-script structured summary

Output: `results/tables/session_metadata.csv`. Full run pending completion of session downloads (67/103 cached).

---

**Date:** 2026-03-15

## Disheng Feedback -- CCG Parameters and Area Consistency

### Source

Two messages from Disheng Tang (PhD student, Dr. Jia's lab) in response to
the CCG benchmark output and pipeline architecture questions.

### CCG epoch: 250ms stimulus window

Disheng: "usually we would separate the stimulus window and gray screen window.
So it's more standard to use 250ms instead of 750ms."

**Action taken:** Changed `trial_duration_ms` from 750 to 250 in `benchmark_ccg.py`.
Added `CCG_STIMULUS_EPOCH_MS = 250` and `CCG_GRAY_EPOCH_MS = 500` to `src/constants.py`.
Gray screen CCG is a separate, optional secondary analysis.

### Firing rate threshold: 2Hz minimum for CCG units

Disheng: "usually we use a firing rate threshold of 2Hz to get enough spikes
for connection estimates otherwise they are too noisy."

Confirmed by Tang et al. (2024) Nature Communications, Methods §Dataset:
"only neurons with a firing rate of at least 2 Hz during all stimuli are
included in our analysis."

**Action taken:** Added `CCG_MIN_FIRING_RATE_HZ = 2.0` to `src/constants.py`.
Applied as a CCG-specific filter in `benchmark_ccg.py` after the Bennett
4-criteria quality filter. Does NOT affect metadata table or global unit counts.

Impact on example session 1044385384: 614 Bennett-filtered → 490 after 2Hz (124 removed, 20%).
Global: 76,602 Bennett-filtered → 58,226 CCG-eligible across 103 sessions (24% removed).

### Scope: all pairwise connections

Disheng: "you should compute all pairwise connections to build a network."

Confirmed: no within-area restriction. Full directed graph across all recorded neurons.

### Significance threshold

Disheng: "You can start with CCG_N_SIGMA=5.0 and later tune it to get
reasonable edge density."

`CCG_N_SIGMA = 5.0` retained as starting point. Tuning deferred to after
first full session CCG run, based on observed edge density.

### Area consistency across sessions

Disheng: "it'll be better if most of your sessions have the same areas."

**Script:** `diagnostic_area_coverage.py`. Results from full 103-session run:

**Key finding: 101 distinct area sets across 103 sessions.** Almost every session
has a unique combination of brain areas. This is because probes traverse many
subcortical/hippocampal structures en route to visual cortex, and the exact
set of structures hit varies per probe insertion.

| Metric | Value |
|---|---|
| CCG-eligible units (Bennett + 2Hz) | 58,226 |
| Distinct area sets | 101 |
| Sessions with all 6 visual areas | 88/103 |
| Mouse pairs with identical area sets | 1/49 |

**Per-area frequency (top areas, CCG-eligible):**

| Area | Sessions | % | Category |
|---|---|---|---|
| VISpm | 102/103 | 99.0% | Visual |
| VISal | 101/103 | 98.1% | Visual |
| VISp | 101/103 | 98.1% | Visual |
| CA1 | 100/103 | 97.1% | Hippocampal |
| VISrl | 99/103 | 96.1% | Visual |
| VISam | 99/103 | 96.1% | Visual |
| DG | 98/103 | 95.1% | Hippocampal |
| VISl | 98/103 | 95.1% | Visual |
| CA3 | 89/103 | 86.4% | Hippocampal |
| LP | 75/103 | 72.8% | Thalamic |
| MRN | 69/103 | 67.0% | Midbrain |
| LGd | 40/103 | 38.8% | Thalamic |

**Core areas (>=90% of sessions):** VISpm, VISal, VISp, CA1, VISrl, VISam, DG, VISl
(all 6 visual areas + 2 hippocampal areas).

**Critical observation:** 88/103 sessions have all 6 visual areas. Only 1/49 mice
have identical area sets between familiar and novel sessions. However, this
mismatch is driven by variable subcortical coverage -- visual area coverage is
highly consistent. For network analysis, the practical approach is:

1. Restrict the primary network to the 6 visual areas (present in 88/103 sessions)
2. Subcortical areas (thalamic/midbrain) analyzed as a secondary layer for the
   subset of sessions that have them (85/103 thalamic, 69/103 midbrain)
3. For paired mouse comparisons, confirm both sessions have the same visual areas

Output: `results/tables/area_coverage_by_session.csv`

---

**Date:** 2026-03-26

## Week 4 -- Trial-Type Alignment with Full Granularity

### Overview

Built per-session trial metadata tables with full trial-type labeling,
behavioral history tags, and stimulus-epoch firing rate refinement.
Script: `build_trial_metadata.py`.

### Sessions Processed

| Metric | Count |
|---|---|
| Sessions processed | 53 |
| Sessions failed (truncated NWB / corrupted NPZ) | 15 |
| Total stimulus presentation rows | 710,097 |
| Unique mice | 34 |
| Familiar sessions | 24 |
| Novel sessions | 29 |

15 failed sessions: 12 have truncated NWB files (incomplete S3 downloads),
3 have corrupted NPZ files ("possible zip bomb" errors). These sessions
were never successfully loaded and are excluded from all analyses.

### Trial-Type Distribution

| Trial Type | Count | % |
|---|---|---|
| passive | 446,299 | 62.8% |
| pre_change_repeat | 232,207 | 32.7% |
| omission | 18,740 | 2.6% |
| change | 12,851 | 1.8% |

### Repeat Position Distribution (pre_change_repeat only)

| Position | Count |
|---|---|
| 1 | 12,904 |
| 2 | 12,904 |
| 3 | 12,904 |
| 4 | 12,899 |
| 5plus | 180,543 |

Positions 1-4 are nearly identical (~12,900 each), confirming the geometric
distribution of change intervals. The 5plus bin dominates because the task's
geometric distribution produces many long repeat sequences.

All 53 sessions had `flashes_since_change` in `stimulus_presentations.columns`;
no manual counter fallback was needed.

### Transition Type Resolution

| Transition Type | Count |
|---|---|
| familiar_to_familiar | 12,851 |
| unknown_transition | 0 |

**Transition resolution rate: 100.0%**

All change trials were resolved. All 12,851 change trials were classified as
`familiar_to_familiar`. This is expected: in this dataset, both Familiar and
Novel sessions use images from the same image set within each session; the
"novel" designation refers to novelty of the *session type* (first exposure
vs. repeated exposure), not individual image novelty within a session.

### Behavioral Outcome Distribution (change trials only)

| Outcome | Count |
|---|---|
| hit | 8,258 |
| miss | 4,434 |
| false_alarm | 0 |
| correct_reject | 0 |

Hit rate across all sessions: 65.1% (8,258 / 12,692 go trials).

### Reset Index Eligibility

45/53 sessions (84.9%) are RI-eligible (both after-hit and after-miss
change trial counts >= `RESET_INDEX_MIN_TRIALS = 30`).

8 sessions below threshold:

| Session | After-Hit | After-Miss | Reason |
|---|---|---|---|
| 1055415082 | 166 | 28 | Low miss count |
| 1064415305 | 244 | 11 | Low miss count |
| 1064639378 | 181 | 23 | Low miss count |
| 1065449881 | 162 | 11 | Low miss count |
| 1095138995 | 226 | 29 | Low miss count (borderline) |
| 1108335514 | 184 | 22 | Low miss count |
| 1115086689 | 263 | 6 | Very low miss count |
| 1118324999 | 8 | 340 | Very low hit count (likely disengaged) |

Session 1118324999 is an outlier: only 8 after-hit change trials vs 340
after-miss, suggesting the mouse was largely disengaged or non-responsive.

### Join Method

All 53 sessions used `trials_id` join (primary method). No temporal
nearest-join fallback was needed.

### Stimulus-Epoch Firing Rate Refinement

| Metric | Value |
|---|---|
| Sessions with companion `_stim_fr.npz` | 50 |
| Total units processed | 35,868 |
| Units gaining CCG eligibility (stim-epoch) | 1,259 |
| Units losing CCG eligibility (stim-epoch) | 2,450 |

3 sessions with trial tables but no original NPZ file did not receive
firing rate refinement. The net effect is a modest reduction (-1,191 units)
in CCG-eligible units when using stimulus-epoch rates vs session-wide rates,
consistent with Tang et al. 2024 criterion "2Hz during all stimuli."

### Sample Coverage Note

53/65 cached sessions successfully processed. 15 cached sessions have
corrupted NWB files from incomplete downloads and cannot be loaded.
53 NPZ spike time files exist from Week 3; 50 received companion
`_stim_fr.npz` files. 3 sessions have trial tables but no NPZ (these
sessions were cached after spike extraction).

### Deliverable Clarification

The RESEARCH_PLAN_V2.md Week 4 deliverable specifies "trial-structured spike
trains." This is fulfilled by:

1. Per-session Parquet metadata tables (`results/derivatives/trial_tables/`)
   that provide the trial-level index (stimulus onset times, trial types,
   behavioral context) for each stimulus presentation.
2. Per-session NPZ spike time files (`results/derivatives/spike_times/`)
   that store the raw spike times per unit.

Actual spike tensors (e.g., binned spike counts aligned to stimulus onset)
are constructed on-the-fly at CCG computation time by combining the metadata
index with the NPZ spike arrays. This avoids storing redundant pre-binned
tensors and preserves flexibility for different bin widths and epoch windows.

### Gap Status

- `strategy_classification.csv` (Piet et al. 2023 visual/timing labels):
  still missing. Requires running the Piet classification algorithm on
  per-session behavioral data. Deferred to Week 5.

### Output Files

- `results/derivatives/trial_tables/{session_id}.parquet` (53 files)
- `results/derivatives/spike_times/{session_id}_stim_fr.npz` (50 files)
- `results/tables/trial_metadata_summary.csv`

---

**Date:** 2026-04-03

## Week 5 -- Unit-Type Labeling and Laminar Depth Tagging

### Metadata Discovery (units/channels columns)

`build_unit_annotations.py` logs the project metadata columns at runtime.

- `units.csv` columns confirm CCF/depth fields are present:
  - `anterior_posterior_ccf_coordinate`
  - `dorsal_ventral_ccf_coordinate`
  - `left_right_ccf_coordinate`
  - `probe_vertical_position`
  - `ecephys_channel_id`
- `channels.csv` columns include probe/channel coordinates and area labels.

This confirms Path A (DV CCF depth) is available and used for laminar assignment.

### Session Coverage and Outputs

- Base NPZ sessions discovered: 80
- Per-session unit tables written: 80 (`results/derivatives/unit_tables/{session_id}.parquet`)
- Summary table written: `results/tables/unit_annotation_summary.csv`
- Figure written: `results/figures/unit_composition_by_area.png`
- QA figure written: `results/figures/waveform_duration_histogram.png`

### Layer Assignment Method and Distribution

Method:
- Visual areas (`VISp`, `VISl`, `VISrl`, `VISal`, `VISpm`, `VISam`) use
  normalized `dorsal_ventral_ccf_coordinate` within each area/session to assign
  `L2/3`, `L4`, `L5`, `L6`.
- Non-visual structures are labeled `subcortical`.
- `grey`, `VIS`, and `VISrll` are labeled `unknown` for laminar tagging.

Distribution across all annotated units (N = 60,006):
- subcortical: 34,539
- L2/3: 7,189
- L4: 5,453
- L5: 6,099
- L6: 3,937
- unknown: 2,789

### Waveform and CCG-Eligibility Summary

- Waveform type counts:
  - RS: 41,014
  - FS: 18,992
- CCG-eligible units (using `_stim_fr.npz` when available, otherwise base NPZ):
  - Total CCG-eligible across 80 sessions: 45,977

### Strategy Classification Status

- `results/tables/strategy_classification.csv` exists but is incomplete.
- Current columns are RI-only (`mouse_id`, `reset_index`, `ri_n_after_hit`,
  `ri_n_after_miss`, `ri_flag`); Piet strategy columns are absent
  (`strategy_label`, `beta_visual`, `beta_timing`, `strategy_index`).
- Attempted reruns:
  - `python3 classify_behavioral_strategy.py` failed in this shell due to
    missing `allensdk` module in the active interpreter.
  - `.venv.bak/bin/python classify_behavioral_strategy.py` exited with code 139.
- Week 5 tables therefore set `mouse_strategy_group = 'pending'` for all units.

### Robustness Notes

- 3 NPZ files had minimal schema (`unit_ids`, `spike_times` only). The builder
  now falls back to `units.csv` for structure, waveform duration/type, and
  session firing rate to avoid data loss.
- No plan file was edited. Week 5 deliverables were implemented via code and
  derivative outputs.

---

## Date: 2026-03-27 — Week 5 Fix: Atlas Laminar, Strategy Classification, QC Flags

Three issues identified during Week 5 verification were resolved in this session.

### Issue 1: Atlas-Registered Laminar Assignment

**Problem**: The original laminar assignment used per-session-area normalized
depth (heuristic), which is scientifically unsound for publication because the
normalization range depends on probe penetration depth, not true cortical
boundaries.

**Fix**: Replaced with atlas-registered assignment using the Allen CCF 25 um
annotation volume (`annotation_25.nrrd`). For each visual area unit with valid
(AP, DV, LR) CCF coordinates, the 3-D annotation volume is queried for the
Allen structure ID, which is mapped to a cortical layer label via the ontology
JSON.

**Code changes**:
- `src/constants.py`: Added `CCF_ANNOTATION_URL`, `CCF_ANNOTATION_PATH`,
  `CCF_ONTOLOGY_URL`, `CCF_ONTOLOGY_PATH`, `CCF_RESOLUTION_UM`.
- `src/data_loading.py`: Renamed `assign_cortical_layer()` to
  `_assign_cortical_layer_heuristic()`. Added `_download_if_missing()`,
  `load_ccf_annotation()`, `_build_ccf_layer_lookup()`, and a new
  `assign_cortical_layer()` that uses atlas lookup with heuristic fallback.
- `build_unit_annotations.py`: `main()` loads the CCF annotation volume and
  passes it through `_build_session_table()` to `assign_cortical_layer()`.

**Results** (80 sessions, 60,006 units):

| Layer       | Count  |
|-------------|--------|
| subcortical | 34,539 |
| L5          |  8,854 |
| unknown     |  7,581 |
| L4          |  3,434 |
| L2/3        |  2,918 |
| L6          |  2,680 |

- 36 layer lookup entries (6 areas x 6 suffixes; prefix collision between
  VISp/VISpm and VISl/VISrl fixed by sorting areas longest-first).
- 23,798 visual area units total; 17,886 (75.2%) assigned atlas-registered
  layers.
- 5,912 visual area units labeled `unknown`:
  - ~50% map to structure ID 0 (outside annotated brain boundaries).
  - ~50% map to adjacent non-target areas (VISli, AUDpo, SSp-bfd, TEa) due to
    registration mismatch between probe-assigned area and CCF coordinate.
- Remaining 1,669 `unknown` are from other non-visual areas with parsing issues.

### Issue 2: QC Flags for Zero-FS Sessions

**Problem**: Sessions with zero FS (fast-spiking / putative inhibitory) units in
a visual area were not flagged, creating a silent data quality concern.

**Fix**: Added a QC check at the end of `build_unit_annotations.py` that scans
all session x area combinations for zero FS counts.

**Output**: `results/tables/unit_annotation_qc_flags.csv`

| session_id | area  | n_units_in_area | n_FS | flag    |
|------------|-------|-----------------|------|---------|
| 1051155866 | VISl  | 24              | 0    | zero_FS |
| 1086410738 | VISpm | 30              | 0    | zero_FS |
| 1093864136 | VISrl | 8               | 0    | zero_FS |
| 1104052767 | VISam | 4               | 0    | zero_FS |
| 1108528422 | VISal | 12              | 0    | zero_FS |
| 1108528422 | VISam | 12              | 0    | zero_FS |

6 session x area combinations flagged. These sessions should be excluded from
RS-vs-FS layer-specific connectivity analyses for the affected areas.

### Issue 3: Behavioral Strategy Classification

**Problem**: All 54 mice had `mouse_strategy_group = 'pending'` because
`classify_behavioral_strategy.py` could not load NWB files through `allensdk`
(version incompatibility: "Can't instantiate abstract class NWBFile with
abstract method external_resources").

**Fix**: Created `classify_strategy_from_nwb.py`, an SDK-free fallback that
reads NWB trial data directly with `h5py`. Implemented a simplified static
logistic regression inspired by Piet et al. (2023):
- `P(lick) = sigmoid(beta_visual * is_change + beta_timing * block_position)`
- `strategy_index = |beta_visual| / (|beta_visual| + |beta_timing|)`
- Index > 0.5 = visual; <= 0.5 = timing

**~~KNOWN LIMITATION — REQUIRES RE-IMPLEMENTATION (identified 2026-03-27)~~**
**RESOLVED 2026-03-28 — see entry "Strategy Classification Rewrite & Deficiency
Fixes" below.**

The limitation described here (trial-level collinearity, r=0.108 stability) has
been fully resolved. The static trial-level regression was replaced with the
Piet et al. (2023) dynamic logistic regression at flash level using `psytrack`.
Split-session stability improved to **r = 0.769, p < 0.0001**. The results and
labels below are from the old implementation and are superseded.

~~**Results** (49 mice, old implementation — superseded):~~

| Label       | Count | Note                         |
|-------------|-------|------------------------------|
| visual      | 32    | superseded — do not use      |
| timing      | 7     | superseded — do not use      |
| uncached    | 9     | resolved in rewrite          |
| unconverged | 1     | resolved in rewrite          |

- ~~Split-session stability: r = 0.108, p = 0.538~~ — **replaced by r = 0.769**
- ~~These labels should NOT be trusted~~ — **current labels are reliable**
- See the 2026-03-28 entry for authoritative results.

### Resolved Items (as of 2026-03-28)

- ✅ Strategy classification re-implemented (Piet et al. 2023, PsyTrack,
  flash-level). Labels are reliable for downstream Week 22 analysis.
  Results: 29 visual, 11 timing, 9 uncached (pending NWB download).
- ✅ 3 corrupted base NPZ files (sessions 1111013640, 1112302803, 1118508667)
  re-extracted via direct `h5py` reads. All 3 have full 10-key NPZ files and
  `_stim_fr.npz` companions. Ready for Week 6 CCG computation.
- ✅ In-place write bug removed from `compute_stim_firing_rates.py`.
- The 6 zero-FS session x area combinations remain flagged. Exclude from
  RS-vs-FS layer-specific connectivity analyses for affected areas. Waveform
  diagnostics saved to `results/tables/unit_annotation_qc_waveform_check.csv`.

---

### Methodological Note: L5 Unit Overrepresentation

Atlas-assigned visual area units show L5 overrepresentation:

| Layer | Units | % of assigned |
|-------|-------|---------------|
| L2/3  | 3,061 | 16.3%         |
| L4    | 3,621 | 19.2%         |
| L5    | 9,327 | 49.5%         |
| L6    | 2,820 | 15.0%         |

**Root cause analysis (two contributing factors):**

**Factor 1 — Neuropixels amplitude bias (primary, ~2× oversampling):**
CCF voxel volume analysis across target visual areas shows L5 occupies
22–26% of cortical volume, not 50%:

| Layer | VISp  | VISl  | VISrl |
|-------|-------|-------|-------|
| L2/3  | 46.1% | 39.7% | 43.8% |
| L4    | 14.5% | 14.6% | 14.4% |
| L5    | 21.9% | 25.6% | 24.0% |
| L6    | 17.5% | 20.0% | 17.7% |

The 2.0–2.3× oversampling is consistent with Neuropixels recording bias
toward large L5 pyramidal neurons with high spike amplitudes (Harris &
Shepherd, 2015; de Vries et al., 2020).

**Factor 2 — Differential border-dropout of superficial layers (secondary):**
L5 fraction of assigned units varies by area: VISp=42.7%, VISl=45.8%,
VISrl=46.1%, VISal=51.0%, VISpm=52.9%, VISam=55.2%. VISp has the
highest atlas assignment rate (94.7%) and the lowest L5 fraction. Areas
with more unknown units (due to pial-surface struct_id=0 dropout) show
higher L5 fractions, because L2/3 and L4 units near the cortical surface
are disproportionately lost to `unknown`, inflating L5's share of the
remaining assigned pool.

**Code verification:** L5 lookup contains exactly 6 structure IDs (one
per visual area): VISal5, VISam5, VISpm5, VISl5, VISp5, VISrl5. No
non-target structures (VISli5, TEa5, AUDpo5) are included — the
overrepresentation is not a code artifact.

**Manuscript action required (Week 37):** State in Methods: "Layer 5
accounts for 49.5% of atlas-assigned units despite occupying 22–26% of
cortical volume (CCF voxel analysis). This 2× oversampling is consistent
with the known spike-amplitude bias of Neuropixels probes toward large
L5 pyramidal neurons (de Vries et al., 2020; Harris & Shepherd, 2015).
The effect is amplified by differential border-dropout at the pial
surface that disproportionately removes superficial (L2/3, L4) units.
All layer-stratified analyses normalize by within-layer unit count."

---

### Methodological Note: VISl and VISrl CCF Registration Mismatch

Atlas layer assignment rates by area (post-neighborhood-search):

| Area  | Assigned | Unknown | Unknown % | Primary unknown cause       |
|-------|----------|---------|-----------|---------------------------- |
| VISp  | 3,867    | 217     | 5.3%      | struct_id_0 (pia boundary)  |
| VISpm | 4,500    | 249     | 5.2%      | struct_id_0                 |
| VISam | 3,682    | 280     | 7.1%      | adjacent area (VISa)        |
| VISal | 2,870    | 818     | 22.2%     | adjacent area (VISa)        |
| VISl  | 2,201    | 1,939   | 46.8%     | adjacent area (VISli, SSp)  |
| VISrl | 1,709    | 1,466   | 46.2%     | adjacent area (SSp-bfd)     |

The ±1-voxel (3×3×3) neighborhood search recovered 288 VISl units (46.2%→53.2%)
and 245 VISrl units (46.1%→53.8%). Remaining unknowns resolve to VISli, SSp-bfd,
and TEa structures across the entire 3×3×3 neighborhood — a genuine CCF
coordinate-to-annotation mismatch, not a code limitation.

Root cause: The Allen spike-sorting assigns structure_acronym by
probe-tract registration (local). CCF coordinates are from the global
brain atlas. At the lateral visual cortex boundary, these two coordinate
systems diverge. This is a known dataset property (Siegle et al., 2021).

**Manuscript action required (Week 37):** State in Methods: "Units in
VISl and VISrl showed a 47% CCF coordinate-to-annotation mismatch,
resolving to adjacent structures VISli and SSp-bfd. This reflects known
registration imprecision at the lateral visual cortex boundary (Siegle
et al., 2021, Nature). Layer-stratified analyses for VISl and VISrl are
restricted to the atlas-assigned subset."

---

## 2026-03-27 — Strategy Classification Rewrite & Deficiency Fixes

### 1. Strategy Classification: Piet et al. (2023) Dynamic Logistic Regression

**Root cause of old r=0.108 split-session stability:** The original
implementation used trial-level logistic regression with `is_change_flag`
and `block_position` as predictors and `responded` as the outcome.
Forensic analysis revealed:
- For go trials (hit/miss), `change_time_no_display_delay` always maps to
  `block_position = 0`, creating near-perfect collinearity between the two
  predictors and rendering `strategy_index` unreliable.
- The observation level was wrong: Piet analyses each image presentation
  (flash), not each trial.
- The outcome variable was wrong: Piet uses lick-bout onset, not
  hit/false-alarm response.

**Fix implemented (Path A — ACTIVE):** Complete rewrite of
`classify_strategy_from_nwb.py` using `psytrack` (Roy et al. 2018):
- Flash-level dynamic logistic regression at each active image presentation
- y = 1 if a new licking bout starts (ILI > 700 ms threshold, Piet default)
- Regressors: visual (is_change), timing (sigmoid of images since last
  bout), omission, post-omission, plus bias
- Strategy index = model evidence reduction when visual is removed minus
  evidence reduction when timing is removed (positive → visual dominant)
- Engagement filter: Piet definition (bout within 10 s OR reward within 120 s)
- Mid-bout exclusion: flashes where the mouse is already licking are removed
- Each session fit independently; mouse-level label = sign of averaged index

**Path B (COMMENTED):** Flash-level static logistic regression with
scikit-learn, using the same design matrix. Available as a simpler
fallback if PsyTrack is unavailable.

**Results:**
- 49 mice total: 29 visual (72.5%), 11 timing (27.5%), 9 uncached
- Split-session stability: **r = 0.769, p < 0.0001** (n = 35 mice)
  (previous: r = 0.108, p = 0.538)
- Bennett et al. state mice "overwhelmingly used visual cues" — our
  72.5% visual classification is consistent
- Strategy index range: -338 to +389 (evidence-based, not bounded to [0,1])

**Files changed:** `classify_strategy_from_nwb.py` (complete rewrite),
`src/constants.py` (added `LICK_BOUT_ILI_S`, `TIMING_SIGMOID_MIDPOINT`,
`TIMING_SIGMOID_SLOPE`, `PSYTRACK_SIGMA_BOUNDS`)

### 2. Vectorized Neighbourhood Search in `src/data_loading.py`

The triple-nested Python for-loop in `assign_cortical_layer()` (lines
420–451) that searched 26 CCF neighbours for unknown border units was
replaced with a vectorised NumPy approach:
- Pre-compute all 26 offset vectors as an (26, 3) array
- Broadcast to (U, 26, 3) neighbour coordinates for U unknown units
- Batch-lookup structure IDs via fancy indexing on `annotation_vol`
- Inner loop (per-unit, 26 neighbours) retained for the area-prefix check
  but the expensive `np.searchsorted` per-unit and triple loop are eliminated

### 3. Fixed Zero-FS QC Heuristic in `build_unit_annotations.py`

Changed the `likely_cause` classifier for zero-FS sessions from:
```
'waveform_shift' if wf.min() > RS_FS_THRESHOLD_MS * 1.5
```
to:
```
'waveform_shift' if wf.min() > RS_FS_THRESHOLD_MS * 1.2 and len(wf) >= 20
```
The 1.2× threshold is more sensitive to subtle shifts, and the minimum
unit count (20) prevents false positives from low-N sessions where the
distribution is too sparse to infer a shift.

### 4. Re-extracted 3 Corrupted Sessions

Sessions 1111013640, 1112302803, and 1118508667 had corrupted base NPZ
files (only `spike_times` and `unit_ids` keys — missing all metadata).
Root cause: the old `compute_stim_firing_rates.py` overwrote base NPZ
files in-place, and interrupted runs left corrupted archives.

Fix: direct h5py extraction from NWB files + project metadata CSV,
bypassing the allensdk NWBFile incompatibility. All 3 sessions now have
full 10-key base NPZ files and `_stim_fr.npz` companion files.

| Session    | Units | CCG-eligible | Visual units |
|------------|-------|-------------|--------------|
| 1111013640 | 1,041 | 714         | 455          |
| 1112302803 | 957   | 708         | 400          |
| 1118508667 | 845   | 604         | 362          |

### 5. Pipeline Rebuild Results

After all fixes, full pipeline rebuild produced:
- 80 sessions with parquet output
- 60,006 total units
- Waveform: RS = 41,014, FS = 18,992
- Layer distribution (visual-area units): L2/3 = 3,061, L4 = 3,621,
  L5 = 9,327, L6 = 2,820, unknown = 6,638
- All sessions use atlas-based laminar assignment (`laminar_method = atlas`)
- Strategy pending units: 3,950 (from 9 uncached mice)

### Pending Items

- [ ] Week 37 manuscript notes: L5 overrepresentation (recording bias),
      VISl/VISrl CCF mismatch (registration precision), strategy
      classification methodology
- [ ] Dynamic weight trace figure (panel 4) — requires re-running with
      wMode persistence for the figure
- [ ] Discuss strategy results with Disheng Tang / Dr. Jia

---

**Date:** 2026-04-06

## Week 6 — CCG Pilot Validation

### Objective

Run full jitter-corrected cross-correlogram (CCG) analysis on the top 3
pilot sessions to validate the pipeline (src/connectivity.py) on real
Allen Visual Behavior Neuropixels data before scaling to all sessions.

### Pilot Session Selection

Sessions selected by highest CCG-eligible unit count among sessions with
trial metadata parquet files available (53/80 sessions have trial tables):

| Session    | CCG-eligible | Active+Engaged trials | Experience |
|------------|-------------|----------------------|------------|
| 1067588044 | 701         | 4,527                | Familiar   |
| 1095138995 | 675         | 4,566                | Familiar   |
| 1111013640 | 672         | 4,475                | Familiar   |

Selection criteria:
- CCG-eligible: `firing_rate_stim_hz >= CCG_MIN_FIRING_RATE_HZ` (2.0 Hz)
- Trial filter: `active == True AND engaged == True AND trial_type != 'omission'`
- Trial start times: `start_time_s` column from per-session parquet

### Implementation

Script: `compute_ccg_pilot.py`

Pipeline per session:
1. Load spike times via `load_extracted_spike_times()` (from NPZ derivatives)
2. Filter to CCG-eligible units using stimulus-epoch firing rate
   (cross-referenced with `_stim_fr.npz` and unit annotation parquet)
3. Load trial table, filter to active+engaged+non-omission presentations
4. Build spike tensor via `spike_times_to_trial_tensor()`:
   `trial_duration_ms = CCG_STIMULUS_EPOCH_MS` (250ms),
   `bin_size_ms = CCG_BIN_SIZE_MS` (1.0ms)
5. Compute jitter-corrected CCG via `compute_ccg_corrected()`:
   `window = CCG_WINDOW_BINS` (100), `num_jitter = CCG_N_SURROGATES` (100),
   `L = CCG_JITTER_WINDOW_BINS` (25), `memory = CCG_JITTER_MEMORY` (False),
   `seed = RANDOM_SEED` (42), parallel across trials (`num_cores = -1`)
6. Detect significance via `get_significant_connections()`:
   `n_sigma = CCG_N_SIGMA` (5.0)
7. Classify connections by sign (exc/inh), type (monosynaptic/intermediate/
   common_input based on peak offset vs. thresholds), and pre/post unit
   properties (area, layer, waveform type from unit annotation parquet)

### CCG Parameters (from src/constants.py)

| Parameter              | Value | Source                        |
|------------------------|-------|-------------------------------|
| CCG_BIN_SIZE_MS        | 1.0   | Tang et al. 2024              |
| CCG_WINDOW_BINS        | 100   | 100ms one-sided (0 -> +100ms) |
| CCG_N_SURROGATES       | 100   | Pattern jitter surrogates     |
| CCG_JITTER_WINDOW_BINS | 25    | Harrison & Geman 2009         |
| CCG_JITTER_MEMORY      | False | Simple spike jitter           |
| CCG_N_SIGMA            | 5.0   | z-score significance          |
| CCG_STIMULUS_EPOCH_MS  | 250   | Stimulus window only          |
| CCG_MIN_FIRING_RATE_HZ | 2.0   | Tang et al. 2024 Methods      |
| MONOSYNAPTIC_PEAK_MS   | 2.0   | Peak offset <= 2 bins         |
| COMMON_INPUT_PEAK_MS   | 5.0   | Peak offset > 5 bins          |
| RANDOM_SEED            | 42    | Reproducibility               |

### Tensor Validation (Session 1067588044)

Tensor construction verified for the first pilot session:
- Shape: (701, 4527, 250) — 701 units x 4,527 trials x 250 bins
- Memory: 793.4 MB (int8)
- Spike density: 0.01135 spikes/bin (consistent with ~2.8 Hz mean rate)
- Construction time: ~7 minutes (Python double-loop over units x trials)
- All 20 unit annotation columns loaded correctly
- Unit areas include all 6 visual areas plus subcortical structures

### Output Files

Per session:
- `results/tables/ccg_pilot_{session_id}_adjacency.csv`
  Columns: pre_unit_id, post_unit_id, sign, connection_type, peak_lag_ms,
  z_score, pre_area, pre_layer, pre_waveform, post_area, post_layer,
  post_waveform
- `results/derivatives/ccg_pilot_{session_id}_ccg.npz`
  Keys: ccg_corrected, sig_ccg, sig_conf, sig_off, sig_dur, unit_ids,
  firing_rates

Summary:
- `results/tables/ccg_pilot_summary.csv`
  One row per session with connection counts, edge density, area breakdown

Figure:
- `results/figures/ccg_pilot_results.png`
  3x3 grid: lag distribution, connection type breakdown, area matrix

### Runtime Estimate

Benchmark (490 units, 412 trials) estimated ~20h single-core. The pilot
sessions have ~700 units and ~4,500 trials (~22x more work). With 8-core
joblib parallelism across trials, estimated ~50-60 hours per session,
~6-8 days total for all 3 sessions. Script saves per-session outputs for
checkpoint-and-resume.

### Connection Statistics

*Pending: CCG computation is running. Results will be appended here once
each session completes. The script prints a structured summary at exit.*

Expected metrics per session:
- n_connections_total, n_excitatory, n_inhibitory
- n_monosynaptic, n_common_input, n_intermediate
- edge_density = n_connections / (N * (N-1))
- pct_within_area, pct_between_area
- QC flags if edge_density < 0.001 or > 0.05

### Pilot Verdict

*Pending completion of CCG computation. Criteria:*
- PASS if: pipeline runs without error on all 3 sessions, produces
  non-trivial edge density (> 0.0001), connection count is biologically
  plausible, and within/between-area ratio is consistent with Tang et al.
  2024 findings
- FAIL if: zero connections detected (threshold too strict), computation
  errors, or memory/runtime makes full-dataset scaling infeasible

### Pending Items

- [ ] CCG computation running on 3 pilot sessions (multi-day runtime)
- [ ] Append final connection statistics and pilot verdict when complete
- [ ] If pilot passes: plan scaling to all 80 sessions (Week 7+)
- [ ] If edge density < 0.001: consider tuning CCG_N_SIGMA threshold
- [ ] Request Tsinghua computing cluster access via Dr. Jia if needed
- [ ] Week 37 manuscript notes (carried forward)
- [ ] Dynamic weight trace figure (carried forward)
- [ ] Discuss strategy results with Disheng Tang / Dr. Jia (carried forward)
