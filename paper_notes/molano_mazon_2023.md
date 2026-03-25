# Molano-Mazón et al. (2023) — Paper Notes

## Citation

Molano-Mazón, M., Shao, Y., Duque, D., Yang, G. R., Ostojic, S., & de la Rocha, J. (2023).
Recurrent networks endowed with structural priors explain suboptimal animal behavior.
*Current Biology*, 33, 622–638.
https://doi.org/10.1016/j.cub.2022.12.044

---

## Source

Dr. Xiaoxuan Jia shared this paper via email on 2026-03-19:

> Hi Joseph,
>
> Here is a related paper: https://www.cell.com/current-biology/fulltext/S0960-9822(22)01981-9
>
> Best wishes,
> Xiaoxuan

---

## Core Question

Why do animals exhibit suboptimal serial biases (trial history dependence) in
perceptual decision-making tasks, and why does this dependence asymmetrically
change after correct vs. error trials?

## Experimental Setup

- **Species:** Rats (59 animals, 4 laboratories)
- **Task:** Two-alternative forced choice (2AFC) auditory frequency discrimination
- **Key observation:** After correct trials, animals show serial biases spanning
  5+ lags. After error trials, the history dependence collapses to lag-1 only
  — a "behavioral reset."
- **Modeling:** LSTM-based RNNs (1024 units) trained with reinforcement learning
  (REINFORCE, ACER, PPO) on N-alternative tasks (N=2,4,8,16), then tested on 2AFC.

## Key Findings

1. Serial biases in animal behavior are explained by structural priors acquired
   through pre-training in complex (N-alternative) environments
2. Pre-trained RNNs reproduce the asymmetric after-correct/after-error history
   dependence; directly-trained RNNs do not
3. The "reset" after errors is mediated by a decoupling mechanism: stimulus
   encoding is preserved but its influence on choice is gated off
4. RL-trained networks show this behavior; supervised-learning-trained networks
   do not

---

## Key Equations

### Equation 3 — Generalized Linear Model (GLM) with Trial History

```
P(choice_t = +1) = sigmoid(
    beta_stim * s_t
    + sum_{k=1}^{K} (beta_correct_k * c_{t-k} + beta_incorrect_k * e_{t-k})
)
```

Where:
- `s_t` = stimulus on trial t (signed stimulus strength)
- `c_{t-k}` = choice on trial t-k after correct outcome (+1 or 0)
- `e_{t-k}` = choice on trial t-k after error outcome (+1 or 0)
- `K` = number of history lags (paper uses K=10)
- `beta_correct_k`, `beta_incorrect_k` = history kernel weights at lag k

**Adaptation for change detection task:**
`P(lick) = sigmoid(beta_stim * is_change + sum_{k=1}^{10} beta_k * outcome_{t-k})`
where `outcome_{t-k}` encodes hit (+1) or miss (-1) at lag k.

### Equation 6 — Reset Index (RI)

```
RI = 1 - |T_after-error| / |T_after-correct|
```

Where:
- `T_after-correct` = sum of absolute history kernel weights fitted on after-correct (after-hit) trials
- `T_after-error` = sum of absolute history kernel weights fitted on after-error (after-miss) trials
- RI near 1 = complete behavioral reset (mouse discards trial history after misses)
- RI near 0 = strategy maintained regardless of outcome
- RI < 0 = stronger history dependence after errors than after correct (reverse pattern)

### Equations 7–11 — LSTM Architecture

Standard LSTM gates:
- Eq 7: Forget gate `f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)`
- Eq 8: Input gate `i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)`
- Eq 9: Candidate cell `C~_t = tanh(W_C * [h_{t-1}, x_t] + b_C)`
- Eq 10: Cell state `C_t = f_t * C_{t-1} + i_t * C~_t`
- Eq 11: Output gate `o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)`

Model specification: 1024 LSTM units, REINFORCE with advantage baseline,
learning rate 7e-4, discount factor 0.99, Adam optimizer.

---

## Key Figures

### Figure 1E — History Kernel Weights

Shows the GLM-fitted beta weights as a function of trial lag (1 through 10),
plotted separately for after-correct and after-error subsets. After-correct
trials show strong weights at lags 1-3 that decay gradually. After-error
trials show weight at lag-1 only, with lags 2-10 near zero — the behavioral
reset signature.

**Project analog:** `results/figures/history_kernels.png` — per-mouse history
kernel profiles from the extended GLM, grouped by Piet strategy label.

### Figure 5D — Transition Kernel Recovery

Shows how the history influence (transition kernel weights) at different lags
recovers as a function of consecutive correct trials following an error. After
the first correct trial post-error, only lag-1 is restored. After 2-3
consecutive corrects, deeper lags (2-5) gradually recover.

**Project analog:** `results/figures/transition_kernels.png` — lag-by-lag
weights after-hit vs. after-miss, averaged across mice. Week 21 adds
gate-and-recovery dynamics tracking this recovery in CCG connectivity.

### Figure 6A-B — SVM Population Decoder and Decoupling

(A) SVM decoder accuracy for stimulus identity is maintained in both
after-correct and after-error conditions — the stimulus is still encoded.
(B) SVM decoder accuracy for previous choice drops specifically in
after-error conditions — the history information is gated off from
influencing the choice. This is the decoupling mechanism.

**Project analog:** Week 29 implements linear SVM on population firing rate
vectors across all 6 visual areas. Week 19 tests the decoupling mechanism
with CCG-based V1-to-higher-area connectivity.

---

## Relevance to Project — 11 Points Mapped to RESEARCH_PLAN_V2 Weeks

### 1. Structural Prior Framework (WHY section, line 171)

Familiar vs. Novel experience levels in Bennett's dataset map onto
Molano-Mazón's pre-training manipulation. Familiar sessions = mice with
structural priors from repeated stimulus exposure. Novel sessions = mice
without those priors. This generates testable predictions about how
connectivity and behavior interact after miss trials.

### 2. Reset Index as New Behavioral Metric (Week 3, line 207)

Compute RI per mouse from the extended GLM fitted separately on after-hit
and after-miss trial subsets. Appended to `strategy_classification.csv`.
RESET_INDEX_MIN_TRIALS = 30 for stable GLM estimation.

### 3. Extended GLM with Trial History (Week 3, line 208)

10-lag GLM: `P(lick) = sigmoid(beta_stim * is_change + sum beta_k * outcome_{t-k})`.
Produces per-mouse history kernel weight profiles directly comparable to
Figure 1E. Implemented in `compute_reset_index.py`.

### 4. Transition Kernel Analysis (Week 3, line 209)

Plot history weights as a function of lag separately for after-hit and
after-miss subsets. Test whether miss trials truncate the history dependence
(only lag-1 non-zero) while hit trials maintain influence across 5+ lags.

### 5. Decoupling Test with Cross-Area CCGs (Week 19, line 380)

Compute V1-to-higher-area CCG peak amplitudes separately for after-hit
and after-miss change trials within the 250ms stimulus epoch. Test whether
V1 local change response is preserved on miss trials but V1-to-higher-area
directed connectivity weakens — the decoupling mechanism in real cortex.

### 6. Structural Prior Test via Familiar vs. Novel (Week 19, line 381)

Test whether Familiar sessions show stronger/more stereotyped connectivity
patterns compared to Novel sessions, consistent with structural priors.

### 7. Gate-and-Recovery Dynamics (Week 21, line 395)

Track CCG-based V1-to-higher-area connection strength as a function of
n_consecutive_hits following a miss trial. Plot recovery curve analogous to
Figure 5B-C. Cross-tabulate with Reset Index.

### 8. Reset Index Stratification (Week 22, line 409)

Cross-tabulate Piet strategy (visual/timing) with Reset Index (high/low)
to create a 2x2 behavioral typology. Run all key network comparisons
separately for each quadrant.

### 9. Transition Kernel Temporal Dynamics (Week 24, line 422)

For sliding temporal windows around the change event, test whether the
trial-history influence identified by the GLM correlates with connectivity
changes in each window.

### 10. Population Decoder — SVM (Week 29, line 469)

Linear SVM on population activity to decode (a) change context and
(b) previous choice. Perform separately for after-hit and after-miss.
If change context decoding drops after misses while single-neuron rates
remain, this confirms the decoupling mechanism at the population level.

### 11. LSTM + RL Model (Week 33, lines 500-502)

1024-unit LSTM trained with REINFORCE, pre-trained on N-alternative task,
fine-tuned on change detection. Test whether pre-trained networks develop
visual/timing strategies and reset behavior. Compare RL vs. supervised
learning. Scatter model RI vs. empirical RI from 54 mice.

---

## Additional Plan Integrations

### Week 4 (line 219)
Tag every engaged trial with `previous_outcome` (hit/miss/fa/cr) and
`outcome_lag_k` for k=1..10, enabling downstream after-hit vs. after-miss
subsetting without recomputing history at each analysis stage.

### Weeks 35-36 (Figures)
- Figure 8: Reset Index panel (RI distribution, RI x strategy scatter,
  RI x network connectivity)
- Supplementary Figure S3: history kernel weight profiles
- Supplementary Figure S4: gate-and-recovery dynamics
- Supplementary Figure S5: SVM population decoder accuracy

### Week 38 (Discussion)
Interpret Familiar vs. Novel network differences as evidence for/against
structural priors. Discuss whether the RNN decoupling mechanism operates
in real cortex after miss trials. Position RI as a bridge between
behavioral typology (Piet) and network gating (Molano-Mazón).

### HOW Section — Methodology Additions
1. Reset Index (Equation 6) — line 741
2. Extended GLM with History Kernel (Equation 3) — line 745
3. Population Decoder SVM (Figure 6) — line 749
4. LSTM + RL Model (Figures 2-5) — line 753

### References
Reference 12 added: Molano-Mazón et al. (2023). *Current Biology* 33, 622-638.
