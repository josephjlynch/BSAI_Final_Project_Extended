# Stimulus-Dependent Functional Network Topology in Mouse Visual Cortex (Extended)

**Course:** Brain Science and Artificial Intelligence (Dr. Xiaoxuan Jia)  
**Institution:** Tsinghua University  
**Date:** December 2025 (original) / February 2026 (extended)

---

## Overview

This is the extended version of the BSAI Final Project. It generalizes the original two-area analysis (V1 and LM) to cover all six mouse visual cortex areas recorded by Neuropixels probes in the Allen Brain Observatory dataset:

| Code | Name | Role |
|------|------|------|
| VISp | V1 | Primary visual cortex |
| VISl | LM | Lateromedial area |
| VISrl | RL | Rostrolateral area |
| VISal | AL | Anterolateral area |
| VISpm | PM | Posteromedial area |
| VISam | AM | Anteromedial area |

Using the same 65 sessions from the original project, this extension computes:
- **6 within-area** correlation matrices (one per area)
- **15 between-area** correlation matrices (every pair of 6 areas)
- Full graph metrics (modularity, clustering, density, path length) per area

---

## What Changed from the Original Project

1. **Data loading** now iterates over all 6 visual areas instead of hardcoding V1 and LM
2. **Connectivity** computes within-area and cross-area matrices for all areas present in each session
3. **Graph metrics** are computed for each area independently
4. **Session inclusion** remains the same 65 sessions; areas with too few neurons are gracefully skipped per-session

The original V1/LM results are numerically reproducible from this codebase.

---

## Repository Structure

```
BSAI_Final_Project_Extended/
├── tutorial.ipynb              # Main tutorial notebook
├── src/                        # Source modules
│   ├── __init__.py
│   ├── data_loading.py         # Allen SDK data loading (6 areas)
│   ├── connectivity.py         # Functional connectivity (N-area)
│   ├── graph_metrics.py        # Network analysis
│   └── statistics.py           # Statistical testing
├── session_ids.txt             # 65 valid session IDs
├── requirements.txt            # Dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Data

**Source:** Allen Brain Observatory Visual Behavior Neuropixels Dataset

- **Modality:** Neuropixels extracellular recordings
- **Species:** Mouse (Mus musculus)
- **Brain Regions:** VISp, VISl, VISrl, VISal, VISpm, VISam
- **Sessions Used:** 65 (see `session_ids.txt`)

**Citation:**
> Siegle, J.H., Jia, X., et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature*, 592, 86-92.

---

## Methods Summary

| Step | Method | Tool |
|------|--------|------|
| Data Loading | Allen SDK cache, 6 areas | `allensdk` |
| Firing Rates | Spike binning (50 ms) | `numpy` |
| Connectivity | Pearson correlation (within + cross-area) | `numpy` |
| Graph Construction | Thresholding (r >= 0.1) | `networkx` |
| Modularity | Louvain algorithm | `python-louvain` |
| Statistics | Paired t-test, Cohen's d | `scipy` |

---

## References

1. Siegle, J.H., Jia, X., et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. *Nature*, 592, 86-92.

2. Tang, D., Zylberberg, J., Jia, X., & Choi, H. (2024). Stimulus type shapes the topology of cellular functional networks in mouse visual cortex. *Nature Communications*, 15, 5753.

3. Jia, X., et al. (2022). Multi-regional module-based signal transmission in mouse visual cortex. *Neuron*, 110(8), 1328-1343.

4. Rubinov, M., & Sporns, O. (2010). Complex network measures of brain connectivity: uses and interpretations. *NeuroImage*, 52(3), 1059-1069.

5. Blondel, V.D., et al. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics*, P10008.

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Acknowledgments

Data provided by the Allen Institute for Brain Science. Available from [brain-map.org](https://portal.brain-map.org/explore/circuits).
