# Plan 2: Functional Network Architecture Underlying Change Detection in the Mouse Visual System (Enhanced)

---

## WHO

### Principal Investigator

**Joseph Lynch** -- visiting undergraduate student at Tsinghua University, majoring in Pharmaceutical Product Development at West Chester University (WCU).

**Courses completed at Tsinghua (Fall 2025):**
- Neuroscience and AI (Dr. Xiaoxuan Jia) -- Grade: A (possibly A+)
- Machine Learning (Dr. Jie Tang)

**Planned courses (Spring 2026):** Deep Learning; potentially AI for Drug Discovery.

**Long-term goal:** AI-driven pharmaceutical drug discovery.

### Supervisors

**Dr. Xiaoxuan Jia** -- Professor, Tsinghua University. Co-creator of the Allen Brain Observatory dataset (Siegle, Jia et al. 2021, *Nature*). Senior author, Jia et al. 2022, *Neuron* (feedforward and recurrent module classification in mouse visual cortex). Runs six research directions including dynamic networks, brain-reader, BMI, and deep learning models.

**Disheng Tang** -- PhD student, Dr. Jia's lab. First author, Tang et al. 2024, *Nature Communications* 15, 5765: "Stimulus type shapes the topology of cellular functional networks in mouse visual cortex." Created the Week 11 Network Neuroscience tutorial used as the foundation for this project.

### Relevant Prior Work by Joseph Lynch

**BSAI Final Project (Fall 2025):** Applied Week 11 Network Neuroscience tutorial methods to Allen Visual Behavior Neuropixels data. Computed Pearson correlation-based functional connectivity across 65 sessions in V1 and LM. Found natural images produce significantly higher modularity than Gabor patches (V1: Cohen's d = 1.52, LM: d = 0.72). All four comparisons significant after Bonferroni correction. Published on GitHub: `josephjlynch/visual-cortex-network-analysis`.

**Extended project (Dimension 1, completed):** Scaled analysis from 2 to all 6 visual cortical areas (V1, LM, RL, AL, PM, AM). Verified with N=50 sessions: strong modularity effect in early areas (V1: d=1.40, LM: d=0.95, AL: d=1.15), absent in higher areas (RL, PM, AM), confirming a hierarchical gradient. Code infrastructure for multi-area data loading, graph construction, checkpoint saving, and multi-session statistics is complete and functional.

**Methodological foundation:** 9 course tutorials completed (Python bootcamp, Data Access, GLM, CNN, Transformer, Dimensionality Reduction, LDS, Network Neuroscience, HMM) plus Dr. Jie Tang's Machine Learning course.

### Meeting and Reporting Structure

**Weekly meetings with Disheng Tang:** 30-60 minutes per week. Purpose: review progress, discuss methodological questions, receive technical guidance on CCG implementation, motif analysis, and module detection. Share weekly progress report before each meeting.

**Monthly meetings with Dr. Xiaoxuan Jia:** 30-60 minutes per month (frequency may increase to biweekly depending on Dr. Jia's availability). Purpose: present findings, receive scientific direction, discuss interpretation, course-correct if needed. Share weekly progress reports continuously so Dr. Jia is informed even between meetings.

**Weekly progress report:** A 1-2 page written document produced every Friday, shared with both Disheng and Dr. Jia. Each report contains:
- Tasks completed this week
- Key figures generated (with brief interpretation)
- Decisions made and rationale
- Blockers or questions for next meeting
- Plan for next week

**Phase-end review meetings:** At the end of each phase (May, August, October), schedule a joint meeting with both Disheng and Dr. Jia to present phase results, discuss interpretation, and confirm direction for the next phase.

---

## WHAT -- The Problem

### The Change Detection Task

Bennett et al. (2025 preprint, "Map of spiking activity underlying change detection in the mouse visual system") recorded >75,000 high-quality units from 54 mice across the visual system (6 cortical areas: V1, LM, RL, AL, PM, AM; plus thalamus: LGN, LP; and midbrain: SC, MRN) while mice performed a change detection task.

**Task structure:**
- Images flash for 250ms, separated by 500ms gray screens
- The same image repeats 4-11 times: familiar, familiar, familiar, familiar, familiar...
- Then the image identity changes to a novel one
- The mouse must lick within 750ms of the change to earn a water reward
- Approximately 5% of images are randomly omitted, extending the gray screen to 1.25s
- Sessions include both active behavior and passive replay epochs

### Bennett's Observations (A-L)

**Observation A: The recurrent cortical wave at 60-100ms is massively amplified by novel images.** Bennett et al. identified a critical decision window of 20-100ms after stimulus onset. Within this window, two waves of activity occur: a feedforward sweep (20-60ms) and a recurrent cortical wave (60-100ms). The recurrent wave is massively amplified by novel images. What generates this amplification is unknown.

**Observation B: The recurrent wave recruits 10-15% more cortical neurons that were previously silent during familiar presentations.** Novel images cause an expanded population of neurons to fire during the late window. Which neurons are recruited and how they connect to the existing network is unknown.

**Observation C: LP (higher-order visual thalamus) leads all cortical areas in decoding image change -- even before V1.** This is unexpected because LP receives cortical feedback, not direct retinal input. The pathway by which LP leads change decoding is unexplained.

**Observation D: Mice use an adaptation-based strategy, not image comparison.** Repeated presentations cause neurons to adapt (fire less), so a different image produces a larger response because neurons are not adapted to it. But the network state that implements adaptation is uncharacterized.

**Observation E: Behavioral outcomes differ (hit vs. miss) but the network basis for successful vs. failed detection is unknown.** The same stimulus can produce a hit or a miss. What determines the outcome at the network level is unknown.

**Observation F: Task engagement modulates neural responses, but how network architecture differs between engaged and disengaged states is unknown.** The dataset includes both active behavior and passive replay within the same sessions, but network differences between these states have not been analyzed.

**Observation G: The feedforward sweep (20-60ms) carries NO novelty information -- it is identical for familiar and novel images.** This constrains the mechanism: whatever generates the novelty-dependent amplification cannot be arriving via feedforward input from retina/LGN. It must be generated cortically or via feedback from a non-feedforward pathway (possibly LP). This is distinct from Observation A because it specifies where the novelty signal does NOT come from.

**Observation H: The midbrain (SCm/MRN) shows the strongest change and task-engagement modulation of any region recorded.** Bennett found that the superior colliculus and midbrain reticular nucleus show stronger modulation by both image change and task engagement than any cortical area. This suggests the midbrain may play a role in the change detection computation, not just cortex.

**Observation I (Nitzan 2025, Paper 3): Visual cortex does not explicitly encode expectations -- omission responses are absent in visual cortex but present in hippocampus.** When a familiar image is omitted, visual cortex neurons show no response at the expected image time -- firing rates ramp linearly through it. But hippocampal neurons do respond. The temporal prediction of when the next image should appear is computed outside visual cortex.

**Observation J (Nitzan 2024, Paper 4): The brain's representation of familiar stimuli is actively disrupted by the presence of novel images in the environment.** When familiar holdover images are interleaved with novel images: behavioral performance drops, neural representation dimensionality increases, V1-to-higher-area predictive communication weakens, and the adaptation signal weakens. The adapted state is not fixed -- it depends on context.

**Observation K (Olsen, Paper 2): The "huge spike" is composed of at least 12 distinct functional neuron types, not a monolithic response.** Some excitatory neurons respond only to novel images, some only to familiar. VIP neurons encode novelty, omissions, and behavior. SST neurons are suppressed by novelty. This diversity means the network architecture underlying the change response has substructure that existing models do not capture.

**Observation L (Piet, Paper 5): The change detection circuit is not a single fixed circuit -- it depends on the individual mouse's behavioral strategy.** Mice using a visual comparison strategy have VIP-on, SST-suppressed circuit configuration. Mice using a timing estimation strategy have the opposite. This varies across individuals and is stable over days.

### Why Change Detection Remains Unexplained (Six Reasons)

**A. The mechanism is not simple adaptation.** The feedforward sweep (20-60ms) carries no novelty information. The novelty signal appears only in the recurrent wave (60-100ms). What generates this recurrent amplification -- which is cortical, not thalamic -- is unknown at the circuit level.

**B. Visual cortex does not explicitly encode what was expected.** Omission responses are absent in visual cortex (Nitzan 2025). If visual cortex does not encode "an image should have been here," it cannot be performing a simple comparison between expected and actual input. The prediction must be implicit in the adaptation state of the network.

**C. The representation of familiar stimuli is not stable.** Placing familiar images in a novel context degrades their representation (Nitzan 2024). The baseline against which change is detected depends on context. If the baseline shifts, how does the brain still detect change?

**D. Cell-type diversity is far greater than existing models accommodate.** Olsen et al. identified 12 distinct functional clusters with different novelty/familiarity encoding. Current models assume homogeneous populations.

**E. Behavioral strategy fundamentally alters the circuit state.** Piet et al. showed the VIP-SST circuit configuration differs by strategy. The "change detection circuit" depends on the animal's behavioral strategy.

**F. Cross-regional interactions during change are unmapped.** LP leads V1 in change decoding. Hippocampal neurons respond to omissions even though visual cortex does not. The full network of regions involved and how information flows between them during the critical 80ms decision window has not been mapped.

### What Bennett Measured vs. What Bennett Did NOT Measure

**What Bennett et al. did:**
- Fit GLMs to individual unit firing rates to characterize sensory vs. action coding
- Clustered units by temporal kernel dynamics (on-transient, on-sustained, off-transient, stimulus-suppressed, etc.)
- Decoded image identity, image change, and lick response from population activity
- Optogenetically silenced VISp to identify the critical decision window (20-100ms)
- Compared responses to novel vs. familiar images
- Showed novelty modulation arises as a recurrent cortical wave at 60-100ms, not in the initial feedforward sweep

**What Bennett et al. did NOT do:**
1. Compute functional connectivity between neurons (no CCGs, no correlation matrices, no adjacency matrices)
2. Build network graphs or analyze network topology
3. Identify modules (feedforward vs. recurrent) during change detection
4. Analyze network motifs
5. Map directed signal flow between areas during the task
6. Examine how network architecture changes between familiar and novel stimulus contexts

Bennett characterized what individual neurons DO during change detection. He did not characterize how neurons are CONNECTED TO EACH OTHER during change detection. That is the unresolved matter.

---

## WHAT -- The Solution

Apply the directed network neuroscience methods developed by Dr. Jia's lab (Jia et al. 2022, *Neuron*; Tang et al. 2024, *Nature Communications*) -- jitter-corrected CCG-based directed, signed connectivity; multi-regional module detection; 3-neuron motif analysis; signal flow metrics -- to the Visual Behavior Neuropixels dataset during the change detection task.

The solution characterizes change detection at the level of network architecture -- directed connectivity, modules, motifs, signal flow -- and tests whether these network features predict behavioral outcomes (hit vs. miss, reaction time). This reveals HOW the recurrent amplification happens (which directed connections between which neurons, organized into which modules, carrying what signal flow, produce that amplification), not merely THAT it happens.

### How Each Gap Is Closed

**Gap 1 (Temporal resolution of CCGs):** Run a feasibility analysis on 3-5 high-unit-count sessions before committing to pairwise CCGs in short temporal windows. Count spike pairs per neuron pair in the 20-60ms and 60-100ms windows. If insufficient: (a) pool across multiple presentations of the same trial type, (b) use population-level connectivity measures, or (c) use the full 250ms presentation but classify connections by CCG peak timing (early peak = feedforward, late peak = recurrent). Discuss with Disheng.

**Gap 2 (Unified multi-area graph):** The primary directed graph per session per condition is a single graph containing all neurons from all 6 areas, with area identity as a node attribute. Module detection, motif census, and signal flow analysis operate on this single unified graph. The 6 within-area and 15 between-area matrices are intermediate products for validation and area-specific statistics, not the core analysis unit.

**Gap 3 (Statistical framework):** For each network metric, specify: (a) the metric, (b) the comparison, (c) the statistical test, (d) the paired unit (session), (e) the multiple comparison correction (Bonferroni), and (f) the null model. This statistical analysis plan is documented and shared with Disheng and Dr. Jia before running Phase 2. See the HOW section for the complete specification.

**Gap 4 (Connecting structure to temporal dynamics):** For each module identified by directed Louvain, compute the average firing rate time course of its member neurons aligned to the change event. Verify whether "recurrent modules" (structurally defined: bidirectional edges) show peak activity at 60-100ms and "feedforward modules" (unidirectional edges) show peak activity at 20-60ms. If structural and temporal definitions do not align, document and interpret the discrepancy.

**Gap 5 (Omission trial analysis):** Compare network architecture during the 500ms gray screen at the expected image time for omission trials vs. the 500ms gray screen between normal image presentations. If connectivity, modularity, or signal flow differs during omissions even though firing rates do not change (Nitzan 2025), the network encodes implicit temporal expectations invisible at the single-neuron level.

**Gap 6 (Behavioral strategy stratification):** Classify each of the 54 mice as visual-strategy or timing-strategy using behavioral metrics from Piet et al. (d-prime on catch trials, response time distributions, lick timing patterns). Run all Phase 2 analyses separately for each strategy group. Test whether network architecture during change detection differs between groups.

---

## WHY -- Novelty

### Three Conditions That Make This Novel

**1. The dataset exists and is public, but the network analysis has not been performed.** Bennett et al. released 76,091 units from 54 mice with simultaneous recordings from all 6 visual cortical areas, thalamus, and midbrain. The data is on DANDI as public NWB files. No published paper has applied CCG-based directed connectivity or network module analysis to this dataset during the change detection task.

**2. The network analysis tools exist, but have only been applied to passive viewing.** Disheng's CCG implementation, motif analysis, and module detection code exist on GitHub (`HChoiLab/functional-network`). Dr. Jia's feedforward/recurrent module classification exists from her 2022 *Neuron* paper. All were developed on the Visual Coding dataset (passive viewing). They have never been applied to the Visual Behavior dataset (active change detection).

**3. The specific question is open, important, and explicitly invited by the dataset authors.** Bennett et al. state in their discussion that their dataset is "an ideal substrate for examining multi-regional interactions and spike propagation through hierarchical networks" and that "the full impact of this database will emerge through extensive community engagement in mining, analysis, and modeling efforts."

### Intellectual Lineage

This project is the third step in a logical progression:
- **Jia et al. 2022 (*Neuron*):** Developed methods (feedforward/recurrent module classification) + applied to passive viewing
- **Tang et al. 2024 (*Nature Communications*):** Extended application (stimulus-dependent motif composition) + passive viewing
- **Lynch et al. (this project):** Extended application (modules, motifs, signal flow during change detection) + active behavior + behavioral linking

### Theoretical Connections

**Predictive coding (Rao & Ballard, 1999; Friston, 2005):** The change detection problem maps onto the predictive coding framework. If the network during familiar repeats develops top-down signal flow (higher areas to V1), this is consistent with predictive coding. If the network shows reduced connectivity (adaptation = fewer connections), it refutes predictive coding at the network level. Either result is interpretively rich.

**Visual awareness:** Change detection requires the mouse to detect that something changed, implicitly comparing current input to the recent past via adaptation. If network architecture distinguishes detected changes (hits) from undetected changes (misses) -- where the stimulus is identical but the behavioral outcome differs -- this identifies a neural correlate of perceptual detection at the network level.

**Structural priors and behavioral reset (Molano-Mazón et al., 2023):** Animals trained in complex environments develop structural priors -- internal biases that constrain learning and behavior in simpler tasks. In a 2AFC task, rats reset their accumulated trial-history evidence after errors, falling back to a stimulus-only strategy. RNNs pre-trained in naturalistic N-alternative environments reproduce this reset behavior; directly-trained RNNs do not. The change detection task has a direct analog: after miss trials (failed detection), mice may discard accumulated adaptation-state evidence and reset their detection strategy. The Familiar vs. Novel experience level in Bennett's dataset maps onto the pre-training manipulation: Familiar sessions = mice with structural priors from repeated exposure; Novel sessions = mice encountering the stimulus set for the first time. This framework generates testable predictions about how network connectivity and behavioral strategy interact after misses vs. hits.

---

## WHEN -- Monthly and Weekly Breakdown

---

### PHASE 1: Data Infrastructure, Connectivity, and Controls (March - May 2026)

**Objective:** Access the Visual Behavior Neuropixels dataset, extract spike times aligned to task events with full trial-type granularity, compute CCG-based directed signed connectivity across all 6 visual cortical areas, classify units by waveform type and laminar position, validate the CCG implementation against published benchmarks, classify mice by behavioral strategy, and establish all controls (firing-rate matching, active vs. passive, Visual Coding cross-validation).

---

#### March 2026

**Week 1 (Mar 2-6): Data access and infrastructure**
- Download and organize NWB files for all 54 mice (103 sessions) from Bennett's dataset via AllenSDK and DANDI archive
- Set up data directory structure: one folder per mouse, one subfolder per session, standardized naming
- Set up private GitHub repository for version-controlled analysis notebooks
- Set up Google Scholar alerts for literature monitoring: "change detection visual cortex network," "Visual Behavior Neuropixels connectivity," author alerts for Bennett, Olsen, Nitzan, Piet
- Assess computational infrastructure: estimate CCG computation time on current hardware; prepare question for Dr. Jia about access to Tsinghua computing cluster or lab workstations
- Deliverable: organized data directory, GitHub repo initialized, Scholar alerts active
- Weekly report #1

**Week 2 (Mar 9-13): Metadata extraction and session characterization**
- Extract session metadata for all 103 sessions: mouse ID, session ID, brain areas recorded, unit counts per area, behavioral performance metrics (d-prime, hit rate, false alarm rate, reaction time distributions)
- Filter units by quality: retain only 'good' quality units
- Assign each unit to a brain area using Allen CCF area labels: VISp (V1), VISl (LM), VISrl (RL), VISal (AL), VISpm (PM), VISam (AM)
- Identify sessions that also contain thalamic (LP, LGN) and midbrain (SC, MRN) units -- tag these for potential Observation H analysis
- Deliverable: session metadata table, unit counts per area per session, list of sessions with subcortical units
- Weekly report #2

**Week 3 (Mar 16-20): Behavioral strategy classification and spike time extraction**
- Classify each of the 54 mice as visual-strategy or timing-strategy using behavioral metrics from Piet et al.: d-prime on catch trials, response time distributions, lick timing patterns relative to expected change time
- Begin extracting spike times for all units across all 103 sessions
- Compute Reset Index (Molano-Mazón et al. 2023, Equation 6) per mouse: fit extended GLM with trial-history kernel separately for after-hit and after-miss trial subsets, derive RI = 1 - |T_after-miss| / |T_after-hit| where T = sum of absolute history weights. RI near 1 = complete behavioral reset after misses; RI near 0 = strategy maintained regardless of outcome
- Implement extended GLM with trial history spanning 10 lags (Molano-Mazón Equation 3): P(lick) = sigmoid(beta_stim * is_change + sum_{k=1}^{10} beta_k * outcome_{t-k}), producing per-mouse history kernel weight profiles that show how far back trial history influences the current decision
- Compute transition kernel by trial lag (Molano-Mazón Figure 5D): plot history weights as a function of lag separately for after-hit and after-miss subsets. Test whether miss trials truncate the history dependence (only lag-1 non-zero) while hit trials maintain influence across 5+ lags
- Deliverable: strategy classification table (54 mice x strategy label), Reset Index column appended to strategy_classification.csv, history kernel figure (results/figures/history_kernels.png), transition kernel figure (results/figures/transition_kernels.png), spike time extraction in progress
- Weekly report #3

**Week 4 (Mar 23-27): Trial-type alignment with full granularity**
- Align spike times to the stimulus presentation table in each session's NWB file
- Separate spike trains into the following trial-type categories:
  - Pre-change familiar repeats, split by position in the repeat sequence (1st, 2nd, 3rd, 4th, 5th+ repeat)
  - Change trials, split by transition type: familiar-to-novel, novel-to-familiar (holdover), novel-to-novel
  - Change trials, further split by behavioral outcome: hit vs. miss
  - Omission trials (the ~5% randomly omitted images)
  - Passive replay presentations (separate epoch within each session)
- For each trial, record: image identity, trial position in block, behavioral outcome, reaction time (for hits)
- Tag each trial with the preceding trial's behavioral outcome (hit/miss/false_alarm/correct_reject) and with trial-lag outcome variables (outcome at lag -1 through -10) for downstream after-hit vs. after-miss conditioning in all Phase 2 and Phase 3 analyses
- For each trial, additionally record: after-hit flag (previous trial was a hit), after-miss flag (previous trial was a miss), n_consecutive_hits (count of consecutive hits immediately preceding this trial, for gate-and-recovery analysis)
- Deliverable: trial-structured spike trains for all sessions with full granularity labels
- Weekly report #4

---

#### April 2026

**Week 5 (Mar 30 - Apr 3): Unit-type labeling and laminar depth tagging**
- Classify all units by waveform type using trough-to-peak duration: narrow-spiking (putative inhibitory, trough-to-peak < 0.4ms) vs. broad-spiking (putative excitatory, trough-to-peak > 0.4ms)
- Extract laminar depth for each unit using Allen CCF registration: assign to cortical layer (L2/3, L4, L5, L6) based on probe depth and CCF coordinates
- Tag each unit in the metadata with: brain area, waveform type (narrow/broad), cortical layer, mouse strategy group (visual/timing)
- Deliverable: fully annotated unit table for all sessions
- Weekly report #5

**Week 6 (Apr 6-10): CCG implementation and pilot validation**
- Implement jitter-corrected cross-correlograms (CCGs) from raw spike trains using the pattern jitter method (Harrison & Geman, 2009), adapted from Disheng's published code (`HChoiLab/functional-network`)
- CCG parameters: 0.5ms bins, +/-50ms window, 1000 jittered surrogates, +/-25ms jitter window, 99.9th percentile significance threshold
- Classify each significant connection: excitatory (peak at short positive lag) or inhibitory (trough at short positive lag)
- Classify CCG peak width: sharp peaks (< 2ms, putative monosynaptic) vs. broad peaks (> 5ms, putative common input)
- Run pilot CCG computation on 3 high-unit-count sessions to validate implementation
- Deliverable: working CCG pipeline, pilot results on 3 sessions
- Weekly report #6

**Week 7 (Apr 13-17): Visual Coding cross-validation and feasibility analysis**
- Download 3-5 sessions from the Allen Visual Coding dataset (publicly available)
- Run the same CCG pipeline on these Visual Coding sessions
- Compare connection densities and excitatory/inhibitory ratios to Disheng's published values in Tang et al. 2024
- If results match: CCG implementation is validated
- If results do not match: debug and consult with Disheng before proceeding
- Run temporal windowing feasibility analysis on the 3 pilot Visual Behavior sessions: count spike pairs per neuron pair in the 20-60ms and 60-100ms windows. Document minimum spike counts. Determine whether pairwise CCGs are feasible in short windows, or whether alternative approaches are needed (pooling across presentations, population-level metrics, CCG peak timing classification)
- Deliverable: Visual Coding validation figures, feasibility analysis report for temporal windowing
- Weekly report #7

**Week 8 (Apr 20-24): Power analysis and computational scaling**
- Compute statistical power analysis: using effect sizes from Dimension 1 results (d=1.40 for V1 modularity) as priors, determine minimum N sessions needed for 80% power at alpha=0.05 for a modularity difference of d=0.5 between conditions
- Estimate total CCG computation time for all 103 sessions based on pilot timings
- If computation exceeds available time: implement parallelization strategy or request access to Tsinghua computing cluster (raise with Dr. Jia at monthly meeting)
- Begin full CCG computation: process first batch of sessions (targeting 5-10 per week depending on hardware)
- Deliverable: power analysis report, computation time estimate, CCG computation in progress
- Weekly report #8

---

#### May 2026

**Week 9 (Apr 27 - May 1): CCG computation -- batch processing**
- Continue CCG computation across all sessions
- For each session and condition (pre-change repeats by position, change trials by transition type and behavioral outcome, omission trials, passive replay), compute:
  - 6 within-area directed, signed adjacency matrices (V1-V1, LM-LM, RL-RL, AL-AL, PM-PM, AM-AM)
  - 15 between-area directed, signed adjacency matrices (all pairwise combinations)
  - 1 unified multi-area adjacency matrix containing all neurons from all 6 areas (primary analysis unit)
- Each adjacency matrix entry: +1 (excitatory), -1 (inhibitory), or 0 (no significant connection)
- Each connection tagged with: peak width (monosynaptic vs. common input), pre-synaptic unit type (narrow/broad), post-synaptic unit type (narrow/broad), pre-synaptic layer, post-synaptic layer
- Checkpoint after each session: save matrices and metadata to disk
- Deliverable: CCG matrices accumulating, checkpoint files growing
- Weekly report #9

**Week 10 (May 4-8): CCG computation continues + firing rate computation**
- Continue CCG computation
- Compute mean firing rates per neuron per condition for all sessions (needed for firing-rate-matched controls in Phase 2)
- For each session, identify the higher-firing-rate condition (change response) and the lower-firing-rate condition (pre-change adapted state)
- Prepare spike subsampling code for firing-rate-matched control: for each neuron in the higher-rate condition, randomly subsample spikes to match the mean rate of the lower-rate condition
- Deliverable: CCG matrices continuing, firing rate tables complete, subsampling code ready
- Weekly report #10

**Week 11 (May 11-15): CCG computation continues + active vs. passive matrices**
- Continue CCG computation
- Ensure separate connectivity matrices are computed for active behavior and passive replay epochs within each session (within-session control for Observation F)
- Deliverable: active and passive epoch matrices computed for completed sessions
- Weekly report #11

**Week 12 (May 18-22): CCG computation completion + quality control**
- Complete CCG computation for all 103 sessions (or as many as hardware allows; document any remaining)
- Quality control on all completed sessions:
  - Connection density (fraction of neuron pairs with significant connections)
  - Excitatory/inhibitory ratio
  - Within-area vs. between-area connection density ratio
  - Monosynaptic vs. common-input connection ratio
  - Compare all statistics to published values from Jia et al. 2022 and Tang et al. 2024
- Document any sessions excluded due to insufficient unit counts, corrupted files, or anomalous CCG statistics
- Deliverable: quality control report, list of valid sessions, Phase 1 completion summary
- Weekly report #12

**Week 13 (May 25-29): Phase 1 review and preregistration**
- Write Phase 1 summary with all validation figures
- Write preregistration document: formal specification of the 5 hypotheses from Plan 1 plus all additional analyses from Plan 2, with predicted outcomes and statistical tests. Share with Disheng and Dr. Jia.
- **Phase 1 review meeting with Disheng and Dr. Jia:** Present Phase 1 results, validation, and preregistered analysis plan. Receive direction for Phase 2.
- Deliverable: Phase 1 summary, preregistration document shared
- Weekly report #13

---

### PHASE 2: Network Analysis During Change Detection (June - August 2026)

**Objective:** Apply directed graph analysis to characterize how network architecture differs between the adapted state and the change detection response, between familiar and novel contexts, between hit and miss trials, between behavioral strategy groups, and during omission trials. Validate all findings with null models, firing-rate-matched controls, split-half reliability, blinded analysis, robustness checks, and image identity controls.

---

#### June 2026

**Week 14 (Jun 1-5): Directed graph construction and blinded analysis setup**
- Build directed graphs (`nx.DiGraph`) from the unified multi-area CCG adjacency matrices for each session and condition
- Nodes: individual neurons, labeled by brain area, waveform type (narrow/broad), cortical layer, and unit ID
- Edges: directed connections with sign (excitatory +1, inhibitory -1) and peak width (monosynaptic/common-input) as edge attributes
- Set up blinded analysis: programmatically shuffle condition labels on 10 sessions, save shuffled versions separately. Run the full Phase 2 pipeline on blinded data first to confirm no false positives from artifacts before analyzing real data
- Deliverable: directed graphs constructed for all sessions, blinded analysis dataset prepared
- Weekly report #14

**Week 15 (Jun 8-12): Module detection and classification**
- Run directed Louvain community detection on the full unified multi-area directed graph for each session and condition (random_state=42 for reproducibility)
- Classify each module as feedforward (predominantly unidirectional edges) or recurrent (predominantly bidirectional edges) following Jia et al. 2022
- For each module, compute cross-area composition: what fraction of its neurons come from each of the 6 areas
- Compute Adjusted Rand Index (ARI) between module assignments across conditions within the same session
- Null model: compare observed modularity to modularity of 1000 random networks with the same degree distribution (configuration model)
- Deliverable: module assignments per session per condition, feedforward/recurrent classification, cross-area composition tables, modularity null comparisons
- Weekly report #15

**Week 16 (Jun 15-19): Module-temporal-dynamics linking (Gap 4 closure)**
- For each module identified by directed Louvain, compute the average firing rate time course of its member neurons aligned to the change event (-100ms to +200ms relative to stimulus onset, in 1ms bins)
- Test whether structurally defined "recurrent modules" show peak activity at 60-100ms and "feedforward modules" show peak activity at 20-60ms
- If structural and temporal definitions align: the recurrent module IS the mechanism of the recurrent wave
- If they do not align: document the discrepancy and interpret (the structural module classification may need refinement for behavioral data)
- Deliverable: module-aligned firing rate time courses, alignment report
- Weekly report #16

**Week 17 (Jun 22-26): Motif census and signal flow**
- Enumerate all 16 possible directed triad types (3-neuron motifs) in each directed graph
- Null model: compare observed motif frequencies to 1000 degree-preserving null model networks (rewired to preserve in-degree and out-degree distributions)
- Identify significantly over- or under-represented motifs per condition
- Compute trophic level (hierarchy score) for each neuron in the directed graph following Jia et al. 2022
- Null model for signal flow: compare to 1000 shuffled edge-weight networks
- Compute centrality measures: in-degree, out-degree, betweenness, PageRank
- Identify hub neurons (top 5% by each centrality measure) per condition
- Deliverable: motif frequency distributions, trophic level distributions, centrality distributions, hub neuron lists
- Weekly report #17

---

#### July 2026

**Week 18 (Jun 29 - Jul 3): Key comparisons -- pre-change vs. post-change**
- For each network metric (modularity, module composition ARI, motif frequencies, mean trophic level, mean centrality), compute the paired difference between pre-change adapted state and post-change response across all sessions
- Statistical tests (specified per metric):
  - Modularity: paired t-test (or Wilcoxon if non-normal, assessed by Shapiro-Wilk) on session-level modularity values; paired unit = session; null model = configuration model random networks
  - Module composition: ARI between pre-change and post-change assignments, compared to null distribution from 1000 condition-label shuffles; paired unit = session
  - Motif frequencies: permutation test (10,000 permutations) on the 16-element frequency vector; paired unit = session
  - Signal flow (trophic level): paired t-test on session-level mean trophic levels; paired unit = session; null model = edge-weight-shuffled networks
  - Centrality: paired t-test on session-level mean centrality values; paired unit = session
- Multiple comparison correction: Bonferroni across all metrics tested
- Effect sizes: Cohen's d with bootstrap 95% confidence intervals (10,000 resamples)
- Deliverable: key comparison results table, effect sizes, significance values
- Weekly report #18

**Week 19 (Jul 6-10): Observation G test + familiar vs. novel comparison**
- Test Observation G explicitly: compare all network metrics (modularity, module composition, motif frequencies, signal flow, centrality) between familiar and novel images in the 20-60ms window. If network architecture is indistinguishable in this window but diverges in the 60-100ms window, it confirms at the network level that the feedforward sweep carries no novelty information
- Run familiar context vs. novel context comparison: compare network architecture during change trials in familiar-to-novel vs. novel-to-familiar vs. novel-to-novel transitions (the three types Disheng highlighted)
- Decoupling test (Molano-Mazón Figure 6, structural prior prediction): compute V1-to-higher-area (LM, RL, AL, PM, AM) CCG peak amplitudes separately for after-hit and after-miss change trials within the 250ms stimulus epoch. Test whether V1 local change response firing rate is preserved on miss trials but V1-to-higher-area directed connectivity weakens -- confirming the decoupling mechanism in real cortex with real multi-area Neuropixels data
- Structural prior test via Familiar vs. Novel comparison: test whether Familiar sessions show stronger/more stereotyped connectivity patterns (higher modularity consistency across trials, more stable module composition, narrower motif frequency distributions) compared to Novel sessions, consistent with structural priors acquired through repeated experience with the stimulus set (Molano-Mazón pre-training framework)
- Deliverable: Observation G test results, transition-type comparison results, decoupling test results, structural prior comparison results
- Weekly report #19

**Week 20 (Jul 13-17): Firing-rate-matched control + adapted-state progression**
- Execute firing-rate-matched control: for each session, subsample spikes from the change response condition to match firing rates of the pre-change adapted state. Recompute CCGs on subsampled data. Recompute all Phase 2 metrics. Report whether findings persist after rate matching
- Adapted-state progression analysis: track how network architecture evolves across repeat positions (1st, 2nd, 3rd, 4th, 5th familiar image). Compute modularity, module composition, motif frequencies, and signal flow at each position. Test for progressive trends using linear mixed-effects models with repeat position as predictor and session as random effect
- Deliverable: firing-rate-matched results, adapted-state progression figures
- Weekly report #20
- **Phase 2 midpoint meeting with Disheng and Dr. Jia:** Present preliminary network analysis results for feedback on interpretation

**Week 21 (Jul 20-24): Hit vs. miss analysis + omission trial analysis (Gap 5 closure)**
- Compare all network metrics between hit trials and miss trials (same stimulus, different behavioral outcome)
- Omission trial analysis (Observation I): compare network architecture during the 500ms gray screen at the expected image time for omission trials vs. the 500ms gray screen between normal image presentations. If connectivity, modularity, or signal flow differs during omissions even though firing rates do not, the network encodes implicit temporal expectations invisible at the single-neuron level
- Gate-and-recovery dynamics (Molano-Mazón Figure 5): track CCG-based V1-to-higher-area connection strength as a function of n_consecutive_hits following a miss trial. Test whether connectivity drops after miss trials and gradually recovers over subsequent hit trials, mirroring the transition kernel recovery dynamics observed in pre-trained RNNs. Plot recovery curve analogous to Figure 5B-C
- Cross-tabulate Reset Index with hit/miss network differences: test whether mice with high RI (strong behavioral reset after misses) show greater connectivity decoupling (larger V1-to-higher-area CCG drop) after misses than mice with low RI (stable strategy). This tests whether the behavioral reset metric predicts the magnitude of network-level gating
- Deliverable: hit vs. miss comparison results, omission trial network analysis results, gate-and-recovery curve, RI-network cross-tabulation
- Weekly report #21

---

#### August 2026

**Week 22 (Jul 27-31): Strategy-stratified analysis (Gap 6 closure) + context-mixing (Observation J)**
- Split the dataset into visual-strategy and timing-strategy groups using the classification from Week 3
- Run all key comparisons (pre-change vs. post-change, familiar vs. novel, hit vs. miss) separately for each strategy group
- Test whether network architecture during change detection differs between strategy groups (between-group comparison)
- Context-mixing analysis (Observation J): compare network architecture during familiar images in all-familiar blocks vs. familiar images interleaved with novel images (mixed blocks). Test whether the adapted state itself is context-dependent at the network level
- Reset Index stratification (Molano-Mazón): cross-tabulate Piet behavioral strategy (visual/timing from Week 3) with Reset Index (high RI > 0.5 / low RI <= 0.5 from Week 3) to create a 2x2 behavioral typology. Run all key comparisons (pre-change vs. post-change, familiar vs. novel, hit vs. miss) separately for each of the four quadrants. Report whether the RI dimension captures variance in network architecture that the Piet classification alone does not
- Deliverable: strategy-stratified results, context-mixing results, RI x strategy 2x2 typology results
- Weekly report #22

**Week 23 (Aug 3-7): Edge-level change analysis + image identity control**
- Edge-level analysis: for each neuron pair, test whether their connection (present/absent, excitatory/inhibitory) differs between pre-change and change response across sessions (paired McNemar test per edge, FDR correction across all edges). Map consistently changing edges onto brain areas
- Image identity control: for sessions where the same image appears as both a repeat and a change (different trial blocks), compare network architecture for that image in the two contexts. If the same image produces different architecture as repeat vs. change, the difference is due to the change detection computation, not image content
- Deliverable: edge-level change map, image identity control results
- Weekly report #23

**Week 24 (Aug 10-14): Midbrain analysis (Observation H) + sliding temporal windows**
- For sessions with midbrain units (SC, MRN, identified in Week 2): compute CCG-based connectivity between midbrain and cortical neurons. Build directed graphs including midbrain. Test whether midbrain leads or follows cortical areas in connectivity changes during change detection
- Sliding temporal window analysis: compute network metrics in windows around the change event (pre-change gray screen, 0-20ms, 20-60ms, 60-100ms, 100-150ms, 150-250ms post-stimulus). Produce time series of modularity, motif frequencies, signal flow, and module membership
- Transition kernel temporal dynamics (Molano-Mazón Figure 5D): for the sliding temporal windows (0-20ms, 20-60ms, 60-100ms, 100-250ms), test whether the trial-history influence identified by the extended GLM (history kernel weights) correlates with connectivity changes in each window. Specifically: if the history kernel shows strong lag-1 influence but no deeper history after misses, does the 60-100ms recurrent window show the largest connectivity change between after-hit and after-miss?
- Deliverable: midbrain connectivity results (if applicable), temporal dynamics time series, transition kernel temporal dynamics results
- Weekly report #24

**Week 25 (Aug 17-21): Robustness checks + split-half reliability + blinded analysis**
- Robustness checks: rerun key comparisons with different CCG parameters -- jitter windows (+/-15ms, +/-25ms, +/-35ms), significance thresholds (99th, 99.5th, 99.9th percentile), bin sizes (0.5ms, 1.0ms). Document whether core findings hold across parameters
- Split-half reliability: randomly split sessions into two halves, confirm all key findings replicate independently in both halves
- Blinded analysis: run the full Phase 2 pipeline on the 10 sessions with shuffled condition labels (prepared in Week 14). Confirm no significant differences are found -- validating that the pipeline does not produce false positives
- Deliverable: robustness check report, split-half reliability report, blinded analysis confirmation
- Weekly report #25

**Week 26 (Aug 24-28): Phase 2 synthesis and review**
- Compile all Phase 2 results into a comprehensive summary with figures
- Generate all Phase 2 figures: paired comparison plots, module composition bar charts, motif frequency profiles, signal flow diagrams, centrality distributions, temporal dynamics time series, edge-level change maps, strategy-group comparisons
- **Phase 2 review meeting with Disheng and Dr. Jia:** Present complete Phase 2 results. Receive direction for Phase 3.
- Deliverable: Phase 2 summary document with all figures
- Weekly report #26

---

### PHASE 3: ML Integration and Behavioral Linking (September - October 2026)

**Objective:** Use machine learning to link network architecture to behavioral outcomes, build encoding models incorporating network context, compare network-based decoding to firing-rate-based decoding, test for cross-mouse generalization of motif structure, compute transfer entropy on a subset of sessions, and build a minimal computational model.

---

#### September 2026

**Week 27 (Sep 1-4): MLP classifier with rigorous cross-validation**
- Construct feature vector for each trial: modularity, number of feedforward modules, number of recurrent modules, motif frequency vector (16 values), mean trophic level, mean in-degree, mean out-degree, mean betweenness, mean PageRank
- Train PyTorch MLP (input -> 64 ReLU -> 32 ReLU -> 1 sigmoid) to predict hit vs. miss
- Validation: leave-one-session-out cross-validation (train on 53 mice, test on 1, repeat for all 54)
- Class balance: document hit/miss ratio. Use balanced accuracy, AUROC, and F1 as metrics
- Significance: permutation-based testing (shuffle labels 1000 times, compute null accuracy distribution)
- Deliverable: MLP classification results, AUROC, balanced accuracy, permutation p-value
- Weekly report #27

**Week 28 (Sep 8-12): Feature ablation + pre-change state prediction + reaction time regression**
- Feature ablation: train separate MLPs with only modularity features, only motif features, only signal flow features, only centrality features. Report which category best predicts behavioral outcomes
- Pre-change state prediction: compute network metrics during the last familiar repeat before the change event. Train MLP to predict upcoming hit vs. miss from the pre-change network state. If successful, the adapted-state network configuration causally influences change detection outcome
- Reaction time regression: replace binary MLP with regression model (MLP with linear output or GLM) predicting reaction time from network features. Test for continuous, graded relationship between network architecture and behavioral performance
- Deliverable: feature ablation results, pre-change prediction results, reaction time regression results
- Weekly report #28

**Week 29 (Sep 15-19): Network vs. firing rates decoding comparison + active vs. passive classifier**
- Network vs. firing rates comparison: train a parallel MLP to predict hit vs. miss from population firing rate features (mean firing rate per area, peak firing rate, firing rate variance). Compare AUROC to the network-features MLP. If network features decode better, network analysis reveals information invisible at the single-neuron level
- Active vs. passive classifier: train a second MLP to distinguish active-behavior network states from passive-replay network states. Compare features that predict active vs. passive to features that predict hit vs. miss. If the same features drive both, change detection is a subset of task engagement. If different, change detection has a dedicated network signature
- Population decoder analysis (Molano-Mazón Figure 6A-B): train linear SVM on population activity (firing rate vector across all 6 visual areas, 250ms stimulus window) to decode (a) change vs. no-change image context and (b) previous choice (mouse licked vs. did not lick on the prior trial). Perform separately for after-hit and after-miss subsets. If change context decoding accuracy drops after misses while single-neuron firing rates to change stimuli remain, this confirms the decoupling mechanism at the population level -- complementing the CCG-based connectivity test from Week 19
- Compare SVM population decoding results to CCG-based connectivity changes for the same after-hit vs. after-miss contrast: if both methods identify decoupling, the finding is robust across analysis levels (pairwise connectivity vs. population information). If only one method detects it, the discrepancy reveals whether the mechanism operates at the synaptic or population coding level
- Deliverable: network vs. firing rate decoding comparison, active vs. passive classifier results, SVM population decoder results (after-hit vs. after-miss)
- Weekly report #29

**Week 30 (Sep 22-26): GLM encoding models + Nitzan 2024 context-mixing analysis**
- GLM encoding models with network context: for each neuron, fit two GLMs predicting firing rate: Model A (stimulus features only) vs. Model B (stimulus + network position: module membership, in-degree, out-degree, centrality). Compare R-squared. If Model B explains more variance, network position contributes to neural responses beyond stimulus drive
- Nitzan 2024 context-mixing: compare network architecture during familiar images in familiar context vs. familiar images in novel context. Test whether the network-level adapted state is context-dependent, potentially explaining the behavioral performance drop Nitzan observed
- Deliverable: GLM comparison results, context-mixing results
- Weekly report #30

---

#### October 2026

**Week 31 (Sep 29 - Oct 2): HMM state-conditioned analysis + dimensionality reduction**
- HMM state-conditioned analysis: fit 2-state HMM to behavioral signals (lick rate, running speed) per session to identify engaged vs. disengaged epochs. Compute network architecture separately within each HMM-identified state. Test whether the change detection network mechanism depends on engagement state
- Dimensionality reduction: vectorize each trial's adjacency matrix, apply PCA then t-SNE. Color by trial type (pre-change, change, omission, passive). Test whether trial types form distinct clusters in network state space
- Deliverable: HMM results, dimensionality reduction visualizations
- Weekly report #31

**Week 32 (Oct 5-9): Transfer entropy + cross-mouse motif generalization**
- Transfer entropy: on a subset of 10 sessions, compute transfer entropy (directed information) alongside CCGs for the same neuron pairs. Test whether network architecture findings generalize beyond pairwise linear interactions captured by CCGs. If transfer entropy reveals additional structure, document it as a methodological extension
- Cross-mouse motif generalization: compute correlation between motif frequency profiles across the 54 mice. Test whether motif composition is a stable fingerprint consistent across individuals (extending Tang 2024 to behavioral data). Test whether motif frequencies are preserved across conditions but participating neurons change
- Deliverable: transfer entropy comparison results, cross-mouse motif results
- Weekly report #32

**Week 33 (Oct 12-16): Computational model**
- Implement a minimal two-layer rate model (V1 -> LM) with adaptation dynamics and recurrent connectivity, parameterized by empirical CCG data
- Show the model reproduces: (a) the two-wave response (feedforward then recurrent), (b) novelty-dependent amplification
- Model prediction: removing recurrent connections eliminates the 60-100ms amplification. Test in data: sessions with weaker empirical recurrent connectivity should show weaker change responses. Compute correlation between recurrent module strength and change-response amplitude across sessions
- LSTM + reinforcement learning model (Molano-Mazón recipe): implement 1024-unit LSTM trained with REINFORCE (advantage baseline, learning rate 7e-4, discount 0.99), receiving stimulus identity, previous choice, and reward as inputs. Pre-train on an N-alternative image classification task (N=4,8,16, analogous to the NAFC pre-training environment), then fine-tune on 2-alternative change detection. Test whether pre-trained networks develop visual vs. timing strategies as emergent structural priors and exhibit reset behavior (RI > 0) after error trials
- RL vs. supervised learning comparison (Molano-Mazón Discussion prediction): train the same LSTM architecture with (a) RL (REINFORCE, reward = correct detection) and (b) supervised learning (cross-entropy on correct response label). Report whether only RL-trained models exhibit naturalistic strategy-dependent reset behavior -- if confirmed, this constrains the class of models that can explain the behavioral data and has implications for which training paradigm to use when modeling change detection circuits
- Compare model Reset Index to empirical Reset Index from the 54 mice: scatter plot of model RI vs. mouse RI, grouped by Piet strategy classification. If pre-trained RL models produce RI values in the same range as real mice (approximately 0.5), this validates the structural prior hypothesis for change detection
- Deliverable: computational model code, model validation figures, model prediction test results, LSTM+RL model results, RL vs. supervised comparison, model-vs-empirical RI scatter
- Weekly report #33

**Week 34 (Oct 19-23): Cross-session population analysis + Phase 3 synthesis**
- Aggregate all Phase 2 and Phase 3 results across the 54 mice
- Population-level statistics: bootstrap 95% confidence intervals (10,000 resamples), Cohen's d effect sizes for every comparison
- Report fraction of sessions showing each effect direction
- Compile Phase 3 summary with all figures
- **Phase 3 review meeting with Disheng and Dr. Jia:** Present complete Phase 3 results. Confirm manuscript direction.
- Deliverable: Phase 3 summary document, population statistics tables
- Weekly report #34

---

### PHASE 4: Writing, Figures, and Submission (November 2026 - January 2027)

**Objective:** Produce publication-quality figures, draft the manuscript with theoretical framing, iterate with supervisors, prepare supplementary materials, release code and data derivatives publicly, and submit.

---

#### November 2026

**Week 35 (Oct 26-30): Figure design and production**
- Design figure layout following the style of Jia et al. 2022 (*Neuron*) and Tang et al. 2024 (*Nature Communications*)
- Produce main figures:
  - Figure 1: Task structure, recording configuration, data pipeline overview
  - Figure 2: CCG validation (Visual Coding cross-validation, connection densities, E/I ratios)
  - Figure 3: Network architecture during adapted state vs. change response (modularity, modules, motifs, signal flow)
  - Figure 4: Temporal dynamics of network reorganization (sliding window time series)
  - Figure 5: Module-temporal-dynamics linking (recurrent modules peak at 60-100ms)
  - Figure 6: Behavioral prediction (MLP results, feature ablation, pre-change prediction, reaction time regression)
  - Figure 7: Network vs. firing rates decoding comparison
  - Figure 8: Structural priors and behavioral reset (history kernels by strategy group, Reset Index distribution, decoupling test results showing V1-to-higher-area CCG drop after misses, gate-and-recovery curve, LSTM model comparison)
- Deliverable: main figure drafts
- Weekly report #35

**Week 36 (Nov 3-7): Supplementary figures and manuscript outline**
- Produce supplementary figures:
  - Robustness checks across CCG parameters
  - Split-half reliability
  - Firing-rate-matched controls
  - Blinded analysis results
  - Image identity control
  - Strategy-stratified comparisons
  - Omission trial analysis
  - Midbrain analysis (if applicable)
  - Context-mixing analysis
  - Edge-level change maps
  - Transfer entropy comparison
  - Cross-mouse motif generalization
  - Computational model
  - Laminar analysis
  - Population decoder (SVM) analysis: after-hit vs. after-miss decoding accuracy
  - Transition kernel temporal dynamics: history kernel correlation with temporal window connectivity
  - RI x strategy 2x2 typology comparisons
- Draft manuscript outline with section headings and key points per section
- Deliverable: supplementary figures, manuscript outline
- Weekly report #36

**Week 37 (Nov 10-14): Manuscript draft -- Methods and Results**
- Write Methods section: complete specification of CCGs, graph construction, module detection, motif census, signal flow, centrality, all statistical tests, all controls, all ML methods, computational model
- Write Results section: walk through all findings in logical order, reference figures
- Deliverable: Methods and Results draft
- Weekly report #37

**Week 38 (Nov 17-21): Manuscript draft -- Introduction and Discussion**
- Write Introduction: change detection problem, Bennett's observations, the network gap, the novelty of bridging Jia/Tang methods to the Visual Behavior dataset
- Write Discussion: interpretation in terms of feedforward/recurrent modules; comparison to Jia 2022 passive viewing results; predictive coding implications; visual awareness connection; limitations; future directions
- Frame the paper explicitly as the third step in the Jia 2022 -> Tang 2024 -> Lynch et al. progression
- Structural priors and behavioral reset (Molano-Mazón et al. 2023): interpret Familiar vs. Novel network differences as evidence for or against structural priors shaping change detection; discuss whether the decoupling mechanism observed in RNNs after errors operates in real mouse cortex after miss trials; discuss the Reset Index as a bridge between Piet et al. strategy classification (behavioral typology) and Molano-Mazón's network-level gating mechanism; compare empirical findings to the LSTM model predictions
- Deliverable: complete manuscript draft
- Weekly report #38

---

#### December 2026

**Week 39 (Nov 24-28): Share draft with Disheng**
- Share complete manuscript draft with Disheng for methodological review
- Deliverable: draft shared, awaiting feedback
- Weekly report #39

**Week 40 (Dec 1-5): Incorporate Disheng's feedback**
- Revise manuscript based on Disheng's methodological feedback
- Address any concerns about CCG implementation, statistical tests, or interpretations
- Deliverable: revised manuscript
- Weekly report #40

**Week 41 (Dec 8-12): Share draft with Dr. Jia**
- Share revised manuscript with Dr. Jia for scientific review
- Deliverable: draft shared with Dr. Jia, awaiting feedback
- Weekly report #41

**Week 42 (Dec 15-19): Incorporate Dr. Jia's feedback**
- Revise manuscript based on Dr. Jia's scientific direction
- If Dr. Jia requests additional analyses: execute them and update results/figures
- Deliverable: final revised manuscript
- Weekly report #42

---

#### January 2027

**Week 43 (Dec 22 - Jan 2): Supplementary materials and code release**
- Prepare supplementary materials: all session IDs, quality control figures, complete statistical tables, robustness checks
- Push all analysis code to public GitHub repository
- Upload computed CCG matrices and network metrics to Figshare or Zenodo as data derivatives
- Deliverable: public repositories ready
- Weekly report #43

**Week 44 (Jan 5-9): Final revision and submission preparation**
- Final manuscript revision incorporating any remaining feedback
- Format manuscript for target journal (as directed by Dr. Jia)
- Prepare cover letter
- Discuss preprint strategy with Dr. Jia: post to bioRxiv simultaneously with journal submission, or wait for acceptance
- Deliverable: submission-ready manuscript, cover letter
- Weekly report #44

**Week 45 (Jan 12-16): Submission**
- Submit to target journal (Nature Communications, Neuron, eLife, or as directed by Dr. Jia)
- If preprint approved: post to bioRxiv to establish priority
- Send thank-you email to Dr. Jia and Disheng summarizing the project and confirming authorship
- Deliverable: paper submitted
- Weekly report #45

---

## HOW -- Detailed Methodology

### Data Source

Allen Brain Observatory Visual Behavior Neuropixels dataset (Bennett et al., 2025). 76,091 high-quality units from 54 mice, 103 recording sessions. Simultaneous recordings from 6 visual cortical areas (V1/VISp, LM/VISl, RL/VISrl, AL/VISal, PM/VISpm, AM/VISam) plus thalamus (LGN, LP) and midbrain (SC, MRN). Data publicly available on DANDI in NWB format. Accessed via the Allen SDK Python package.

### Connectivity Method: Jitter-Corrected Cross-Correlograms (CCGs)

For each pair of simultaneously recorded neurons, compute the cross-correlogram (histogram of spike time differences) in 0.5ms bins over a +/-50ms window. Generate 1000 jittered surrogate spike trains by randomly shifting each spike within a +/-25ms jitter window (pattern jitter method, Harrison & Geman, 2009). Compute surrogate CCGs and identify significant peaks (excitatory connections) or troughs (inhibitory connections) exceeding the 99.9th percentile of the surrogate distribution. This produces a directed, signed adjacency matrix: neuron A excites neuron B (A->B, +1), neuron A inhibits neuron B (A->B, -1), or no significant connection (0). Source: Disheng's code at `HChoiLab/functional-network`.

**CCG peak classification:** Sharp peaks (trough-to-peak < 2ms) classified as putative monosynaptic connections. Broad peaks (> 5ms) classified as putative common input. This distinction determines whether network changes reflect local circuit rewiring or shared modulatory signals.

### Temporal Windowing Strategy (Gap 1 Closure)

Feasibility analysis in Week 7 determines the approach:
- **If pairwise CCGs are feasible in 40ms windows:** compute separate CCG matrices for 20-60ms (feedforward) and 60-100ms (recurrent) windows
- **If spike counts are insufficient for pairwise CCGs:** implement one or more alternatives:
  - (a) Pool spikes across multiple presentations of the same trial type to accumulate counts
  - (b) Use population-level connectivity measures (noise correlations, population coupling) requiring fewer spikes
  - (c) Compute CCGs on the full 250ms presentation but classify each connection as feedforward or recurrent based on CCG peak latency (early peak = feedforward, late peak = recurrent)
- Decision documented in feasibility report (Week 7) and discussed with Disheng

### Graph Construction

Convert each unified multi-area directed, signed adjacency matrix to a NetworkX `DiGraph`. Nodes: individual neurons. Node attributes: brain area, waveform type (narrow/broad), cortical layer (L2/3, L4, L5, L6), unit ID, mouse strategy group (visual/timing). Edges: directed connections. Edge attributes: sign (excitatory +1, inhibitory -1), peak width (monosynaptic/common-input). One graph per session per condition.

**Primary analysis unit:** The single unified multi-area graph containing all neurons from all 6 areas. Module detection, motif census, and signal flow analysis operate on this graph. The 21 area-specific matrices (6 within, 15 between) are intermediate products for validation and area-specific reporting.

### Module Detection

Run directed Louvain community detection on the full multi-area directed graph (random_state=42). Classify each module as feedforward (predominantly unidirectional edges) or recurrent (predominantly bidirectional edges) following Jia et al. 2022. Quantify module consistency across conditions using Adjusted Rand Index (ARI). For each module, compute cross-area composition (fraction of neurons from each of the 6 areas).

**Null model:** Compare observed modularity to modularity of 1000 configuration model random networks (preserving degree distribution).

### Motif Census

Enumerate all 16 possible directed triad types (3-neuron motifs) using triad census. Compare observed frequencies to 1000 degree-preserving null model networks (rewired to preserve in-degree and out-degree).

### Signal Flow

Compute trophic level (hierarchy score) for each neuron in the directed graph following Jia et al. 2022. V1 neurons expected to have lower trophic levels (upstream); LM, RL, AL, PM, AM expected to have higher levels (downstream).

**Null model:** Compare to 1000 edge-weight-shuffled networks.

### Centrality

In-degree, out-degree, betweenness centrality, PageRank for each neuron. Hub neurons: top 5% by each measure.

### Module-Temporal-Dynamics Linking (Gap 4 Closure)

For each module identified by directed Louvain, compute average firing rate time course of member neurons aligned to the change event (-100ms to +200ms, 1ms bins). Verify whether structural module classification (feedforward/recurrent) aligns with temporal response profile (peak at 20-60ms vs. 60-100ms).

### Statistical Framework (Gap 3 Closure)

| Metric | Comparison | Test | Paired Unit | Correction | Null Model |
|--------|-----------|------|-------------|------------|------------|
| Modularity (Q) | Pre-change vs. post-change | Paired t-test or Wilcoxon | Session | Bonferroni | Configuration model |
| Module composition | Pre-change vs. post-change | ARI vs. null distribution | Session | Bonferroni | 1000 label shuffles |
| Motif frequencies (16-vector) | Pre-change vs. post-change | Permutation test (10,000) | Session | Bonferroni | Degree-preserving rewiring |
| Signal flow (trophic level) | Pre-change vs. post-change | Paired t-test | Session | Bonferroni | Edge-weight shuffle |
| Centrality (4 measures) | Pre-change vs. post-change | Paired t-test | Session | Bonferroni | Configuration model |
| Edge presence/absence | Pre-change vs. post-change | McNemar per edge | Neuron pair | FDR | None (direct test) |

All comparisons also run for: familiar vs. novel, hit vs. miss, active vs. passive, visual-strategy vs. timing-strategy, omission vs. normal gray screen, repeat position progression, context (all-familiar vs. mixed).

Effect sizes: Cohen's d with bootstrap 95% CI (10,000 resamples). Population statistics: fraction of sessions showing each effect direction.

### Firing-Rate-Matched Control

For each session, subsample spikes from the higher-firing-rate condition (change response) to match the mean firing rate of the lower-firing-rate condition (pre-change adapted state). Subsample independently for each neuron. Recompute CCGs on subsampled data. Recompute all Phase 2 metrics. Report whether findings persist.

### Omission Trial Analysis (Gap 5 Closure)

Compare network architecture during the 500ms gray screen at the expected image time for omission trials vs. the 500ms gray screen between normal image presentations. Use the same network metrics (modularity, module composition, motif frequencies, signal flow, centrality). If the network changes during omissions even though firing rates do not (Nitzan 2025), the network encodes implicit temporal expectations invisible at the single-neuron level.

### Behavioral Strategy Classification (Gap 6 Closure)

Classify each mouse as visual-strategy or timing-strategy using behavioral metrics from Piet et al.: d-prime on catch trials, response time distributions, lick timing patterns relative to expected change time. Run all Phase 2 analyses separately for each group. Test between-group differences with unpaired t-tests or Mann-Whitney U.

### Unit-Type Subnetwork Analysis

Using waveform-based classification (narrow-spiking = putative inhibitory, broad-spiking = putative excitatory), analyze excitatory and inhibitory subnetworks separately. For each condition, compute: (a) excitatory-only graph (only broad-spiking neurons and their connections), (b) inhibitory-only graph (only narrow-spiking neurons), (c) mixed connections (excitatory-to-inhibitory, inhibitory-to-excitatory). Test whether subnetworks reorganize differently during change detection.

### Laminar Analysis

For each brain area, separate neurons by cortical layer (L2/3, L4, L5, L6). Compute within-layer and between-layer connectivity. Test whether the recurrent module during the change response is concentrated in L2/3 (as Bennett's firing rate data suggests) and the feedforward module in L4 (the thalamocortical input layer).

### MLP Classifier

Input: network topology feature vector per trial. Output: binary (hit/miss) or continuous (reaction time). Architecture: input -> 64 ReLU -> 32 ReLU -> output. Training: binary cross-entropy (classification) or MSE (regression), Adam optimizer, learning rate 0.001. Validation: leave-one-session-out cross-validation. Metrics: balanced accuracy, AUROC, F1 (classification); R-squared, Pearson r (regression). Significance: permutation test (1000 label shuffles).

**Feature ablation:** Train separate models with modularity-only, motif-only, signal-flow-only, centrality-only features.

**Pre-change prediction:** Use network features from the last pre-change repeat to predict upcoming behavioral outcome.

### GLM Encoding Models

Model A: firing_rate ~ stimulus_features (image identity, change/repeat, time bins). Model B: firing_rate ~ stimulus_features + network_context (module membership, in-degree, out-degree, centrality). Compare R-squared between models. Delta-R-squared = variance explained by network context beyond stimulus.

### HMM State Identification

Fit 2-state HMM to behavioral signals (lick rate, running speed) per session. Compute network architecture separately within each HMM-identified state.

### Transfer Entropy

On 10 sessions, compute transfer entropy for the same neuron pairs analyzed with CCGs. Compare directed connectivity estimates. Transfer entropy captures nonlinear, higher-order interactions that CCGs may miss.

### Computational Model

Minimal two-layer rate model (V1 -> LM) with: (a) adaptation dynamics (exponential decay with time constant fit to empirical adaptation progression data), (b) feedforward connections (V1 -> LM), (c) recurrent connections (within V1, within LM, parameterized by empirical CCG-derived connection strengths). Input: repeated familiar image (flat drive) then novel image (new drive). Output: population firing rate time course. Test: model reproduces two-wave response and novelty amplification. Prediction: removing recurrent connections eliminates 60-100ms amplification. Validation: correlate empirical recurrent module strength with change-response amplitude across sessions.

### Reset Index (Molano-Mazón Equation 6)

RI = 1 - |T_after-miss| / |T_after-hit|, where T = sum of absolute history kernel weights from the extended GLM. Fit separately for after-hit and after-miss trial subsets per mouse. RI near 1 = complete behavioral reset after misses (mouse discards trial history); RI near 0 = strategy maintained regardless of outcome. Minimum 30 trials in each subset (RESET_INDEX_MIN_TRIALS) for reliable estimation. Report RI per mouse alongside Piet strategy classification.

### Extended GLM with History Kernel (Molano-Mazón Equation 3)

P(lick) = sigmoid(beta_stim * is_change + sum_{k=1}^{GLM_HISTORY_LAGS} beta_k * outcome_{t-k}), where outcome_{t-k} encodes the behavioral outcome (hit=+1, miss=-1) at trial lag k. GLM_HISTORY_LAGS = 10 (matching Molano-Mazón). Fit per mouse using sklearn LogisticRegression with StandardScaler. Produce history kernel plots (beta_1 through beta_10) separately for after-hit and after-miss trial subsets, directly comparable to Molano-Mazón Figure 1E.

### Population Decoder -- SVM (Molano-Mazón Figure 6)

Train linear SVM (sklearn.svm.LinearSVC) on population firing rate vectors (one entry per unit across all 6 visual areas, computed in the 250ms stimulus window) to decode (a) change vs. no-change context and (b) previous choice (licked vs. did not lick). 5-fold cross-validation within each session. Split analysis by after-hit and after-miss trials. Report decoding accuracy, AUC-ROC, and significance via permutation test (1000 label shuffles). Compare after-hit vs. after-miss accuracy to quantify decoupling.

### LSTM + Reinforcement Learning Model (Molano-Mazón Figures 2-5)

1024-unit LSTM (PyTorch), trained with REINFORCE algorithm (advantage baseline, learning rate 7e-4, discount factor 0.99, Adam optimizer). Inputs per trial: stimulus identity (one-hot encoded image), previous choice (lick/no-lick), reward (binary). Output: lick/no-lick probability (softmax). Pre-training: N-alternative image classification task (N=4,8,16) with repeating/alternating block structure, 100,000 trials. Fine-tuning: 2-alternative change detection task, 50,000 trials. Comparison condition: identical architecture trained with supervised learning (cross-entropy loss on correct response). Report: learning curves, psychometric curves, Reset Index, history kernel weights, strategy classification. Random seed: RANDOM_SEED = 42.

---

## OPERATIONAL PROCESS

### Literature Monitoring

Google Scholar alerts (set up Week 1):
- "change detection visual cortex network"
- "Visual Behavior Neuropixels connectivity"
- Author alerts: Bennett, Olsen, Nitzan, Piet
- Check weekly. If competing work appears, alert Disheng and Dr. Jia immediately

### Version Control

Private GitHub repository (initialized Week 1):
- All analysis notebooks version-controlled
- Clear cell descriptions in every notebook
- Commit after every significant analysis step
- Git history serves as audit trail for reviewer questions

### Computational Infrastructure

- Estimate computation time from pilot CCG results (Week 6-7)
- If computation exceeds available hardware: request access to Tsinghua computing cluster via Dr. Jia
- Parallelize across sessions (independent computations)

### Preregistration

Formal preregistration document (Week 13):
- All 5 hypotheses from Plan 1
- All additional analyses from Plan 2
- Predicted outcomes for each
- Statistical tests specified
- Shared with Disheng and Dr. Jia before running Phase 2

### Data and Code Sharing

- Public GitHub repository with all analysis code (released at submission)
- Figshare or Zenodo repository with computed CCG matrices and network metrics
- Complete session ID list for reproducibility

### Preprint Strategy

Discuss with Dr. Jia (raise in first monthly meeting):
- Post to bioRxiv simultaneously with journal submission to establish priority?
- Or wait for journal acceptance?
- Dr. Jia's preference determines the approach

---

## REFERENCES

1. Bennett et al. (2025 preprint). "Map of spiking activity underlying change detection in the mouse visual system."
2. Jia et al. (2022). *Neuron*. Feedforward and recurrent module classification in mouse visual cortex.
3. Tang et al. (2024). *Nature Communications* 15, 5765. "Stimulus type shapes the topology of cellular functional networks in mouse visual cortex."
4. Olsen et al. (2023/2025, bioRxiv). "Stimulus novelty uncovers coding diversity in visual cortical circuits."
5. Nitzan et al. (2025). *Science Advances*. "Diversity of omission responses to visual images across brain-wide regions."
6. Nitzan et al. (2024). *Cell Reports*. "Mixing novel and familiar cues modifies representations of familiar visual images and affects behavior."
7. Piet et al. (2023, bioRxiv). "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex."
8. Harrison & Geman (2009). Pattern jitter method for cross-correlogram significance testing.
9. Siegle, Jia et al. (2021). *Nature*. Allen Brain Observatory dataset.
10. Rao & Ballard (1999). *Nature Neuroscience*. Predictive coding in the visual cortex.
11. Friston (2005). *Philosophical Transactions of the Royal Society B*. A theory of cortical responses.
12. Molano-Mazón et al. (2023). *Current Biology* 33, 622–638. "Recurrent networks endowed with structural priors explain suboptimal animal behavior."
