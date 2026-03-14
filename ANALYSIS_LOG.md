# Analysis Log

Lab notebook for the Visual Behavior change detection project. Do not ignore in version control.

---

## Entry 1 — Week 1 Part 1 (Code Infrastructure)

**Date:** 2026-03-04

### Session selection rationale and filter used

Bennett et al. (2025) use 103 QC-passed sessions from the Allen Visual Behavior Neuropixels dataset. Sessions excluded if `abnormal_histology == True` or `abnormal_activity == True`. Session list generation and validation deferred to Part 2 (`validate_sessions.py`).

### Unit quality filter decision

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
