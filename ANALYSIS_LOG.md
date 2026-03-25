

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
