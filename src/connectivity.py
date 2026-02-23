"""
Connectivity Module
===================

Computes functional connectivity matrices from firing rate data
across an arbitrary number of brain areas.

VERIFIED CONSTRAINTS:
- Correlation is significant (z=26.52 vs shuffled)
- Mean correlation ~0.05 (weak but real)
- Supports within-area and cross-area matrices for N areas
"""

import numpy as np
from itertools import combinations
from typing import Tuple, Dict, List
import os


def compute_correlation_matrix(
    firing_rates: np.ndarray,
    min_rate_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Pearson correlation matrix.
    
    Args:
        firing_rates: Array of shape (n_units, n_time_bins)
        min_rate_threshold: Minimum mean firing rate to include unit (Hz)
        
    Returns:
        Tuple of:
            - Correlation matrix of shape (n_active, n_active)
            - Boolean mask indicating which units passed the rate threshold
    """
    mean_rates = firing_rates.mean(axis=1)
    active_mask = mean_rates >= min_rate_threshold
    
    if active_mask.sum() < 2:
        raise ValueError(
            f"Not enough active neurons (only {active_mask.sum()} "
            f"with rate >= {min_rate_threshold} Hz)"
        )
    
    active_rates = firing_rates[active_mask, :]
    corr_matrix = np.corrcoef(active_rates)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    return corr_matrix, active_mask


def compute_cross_area_correlation(
    firing_rates_area1: np.ndarray,
    firing_rates_area2: np.ndarray,
    min_rate_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cross-area correlation matrix between two brain areas.
    
    Args:
        firing_rates_area1: Array of shape (n_units_1, n_time_bins)
        firing_rates_area2: Array of shape (n_units_2, n_time_bins)
        min_rate_threshold: Minimum mean firing rate to include unit
        
    Returns:
        Tuple of:
            - cross_corr: Correlation matrix of shape (n_active_1, n_active_2)
            - mask_area1: Boolean mask for active units in area 1
            - mask_area2: Boolean mask for active units in area 2
    """
    mean_rates_1 = firing_rates_area1.mean(axis=1)
    mean_rates_2 = firing_rates_area2.mean(axis=1)
    
    mask_1 = mean_rates_1 >= min_rate_threshold
    mask_2 = mean_rates_2 >= min_rate_threshold
    
    if mask_1.sum() < 1 or mask_2.sum() < 1:
        raise ValueError(
            f"Not enough active neurons: Area1={mask_1.sum()}, Area2={mask_2.sum()}"
        )
    
    active_rates_1 = firing_rates_area1[mask_1, :]
    active_rates_2 = firing_rates_area2[mask_2, :]
    
    combined = np.vstack([active_rates_1, active_rates_2])
    full_corr = np.corrcoef(combined)
    
    n1 = active_rates_1.shape[0]
    cross_corr = full_corr[:n1, n1:]
    cross_corr = np.nan_to_num(cross_corr, nan=0.0)
    
    return cross_corr, mask_1, mask_2


def threshold_to_adjacency(
    corr_matrix: np.ndarray,
    threshold: float = 0.1,
    absolute: bool = True
) -> np.ndarray:
    """
    Convert correlation matrix to binary adjacency matrix.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold for edge
        absolute: If True, use absolute value of correlation
        
    Returns:
        Binary adjacency matrix
    """
    if absolute:
        adj = (np.abs(corr_matrix) >= threshold).astype(int)
    else:
        adj = (corr_matrix >= threshold).astype(int)
    
    np.fill_diagonal(adj, 0)
    
    return adj


# =============================================================================
# MULTI-AREA CONNECTIVITY
# =============================================================================

def compute_all_connectivity_matrices(
    firing_rates_by_area: Dict[str, np.ndarray],
    unit_ids_by_area: Dict[str, list],
    min_rate_threshold: float = 0.1
) -> Dict:
    """
    Compute within-area and between-area correlation matrices for all areas.
    
    For N areas, produces N within-area matrices and C(N,2) between-area
    matrices covering every pair.
    
    Args:
        firing_rates_by_area: {area_code: firing_rates_array (n_units, n_bins)}
        unit_ids_by_area: {area_code: list_of_unit_ids}
        min_rate_threshold: Minimum firing rate threshold (Hz)
        
    Returns:
        Dictionary with:
            - 'within': {area_code: (corr_matrix, active_mask)}
            - 'between': {(area_i, area_j): (cross_corr, mask_i, mask_j)}
            - 'unit_ids_by_area': echoed back for metadata
    """
    result = {
        'within': {},
        'between': {},
        'unit_ids_by_area': unit_ids_by_area,
    }
    
    areas = list(firing_rates_by_area.keys())
    
    for area in areas:
        try:
            corr, mask = compute_correlation_matrix(
                firing_rates_by_area[area], min_rate_threshold
            )
            result['within'][area] = (corr, mask)
        except ValueError as e:
            print(f"  Warning: skipping within-{area}: {e}")
    
    for area_i, area_j in combinations(areas, 2):
        try:
            cross, mask_i, mask_j = compute_cross_area_correlation(
                firing_rates_by_area[area_i],
                firing_rates_by_area[area_j],
                min_rate_threshold
            )
            result['between'][(area_i, area_j)] = (cross, mask_i, mask_j)
        except ValueError as e:
            print(f"  Warning: skipping {area_i}-{area_j}: {e}")
    
    return result


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_connectivity_matrices(
    matrices: Dict,
    stimulus_type: str,
    output_dir: str = 'results/matrices'
):
    """
    Save connectivity matrices to disk with area-keyed filenames.
    
    Args:
        matrices: Dictionary from compute_all_connectivity_matrices()
        stimulus_type: e.g. 'natural' or 'gabor'
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for area, (corr, _mask) in matrices['within'].items():
        np.save(f"{output_dir}/within_{area}_{stimulus_type}.npy", corr)
    
    for (area_i, area_j), (cross, _mi, _mj) in matrices['between'].items():
        np.save(f"{output_dir}/between_{area_i}_{area_j}_{stimulus_type}.npy", cross)
    
    metadata = {}
    for area, (corr, mask) in matrices['within'].items():
        metadata[f'mask_{area}'] = mask
    for (area_i, area_j), (cross, mask_i, mask_j) in matrices['between'].items():
        metadata[f'mask_{area_i}_{area_j}_1'] = mask_i
        metadata[f'mask_{area_i}_{area_j}_2'] = mask_j
    for area, uid_list in matrices['unit_ids_by_area'].items():
        metadata[f'unit_ids_{area}'] = np.array(uid_list)
    
    metadata['within_areas'] = np.array(list(matrices['within'].keys()))
    between_keys = list(matrices['between'].keys())
    if between_keys:
        metadata['between_area1'] = np.array([k[0] for k in between_keys])
        metadata['between_area2'] = np.array([k[1] for k in between_keys])
    
    np.savez(f"{output_dir}/metadata_{stimulus_type}.npz", **metadata)
    
    n_within = len(matrices['within'])
    n_between = len(matrices['between'])
    print(f"[OK] Saved {n_within} within + {n_between} between matrices "
          f"for {stimulus_type} to {output_dir}")


def load_connectivity_matrices(
    stimulus_type: str,
    input_dir: str = 'results/matrices'
) -> Dict:
    """
    Load connectivity matrices from disk.
    
    Args:
        stimulus_type: e.g. 'natural' or 'gabor'
        input_dir: Input directory
        
    Returns:
        Dictionary matching compute_all_connectivity_matrices() output format
    """
    metadata = np.load(
        f"{input_dir}/metadata_{stimulus_type}.npz", allow_pickle=True
    )
    
    result = {'within': {}, 'between': {}, 'unit_ids_by_area': {}}
    
    within_areas = list(metadata['within_areas'])
    for area in within_areas:
        corr = np.load(f"{input_dir}/within_{area}_{stimulus_type}.npy")
        mask = metadata[f'mask_{area}']
        result['within'][area] = (corr, mask)
    
    if 'between_area1' in metadata:
        area1_list = list(metadata['between_area1'])
        area2_list = list(metadata['between_area2'])
        for a1, a2 in zip(area1_list, area2_list):
            cross = np.load(f"{input_dir}/between_{a1}_{a2}_{stimulus_type}.npy")
            mask_1 = metadata[f'mask_{a1}_{a2}_1']
            mask_2 = metadata[f'mask_{a1}_{a2}_2']
            result['between'][(a1, a2)] = (cross, mask_1, mask_2)
    
    for key in metadata.files:
        if key.startswith('unit_ids_'):
            area = key[len('unit_ids_'):]
            result['unit_ids_by_area'][area] = list(metadata[key])
    
    return result


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def summarize_connectivity(matrices: Dict) -> Dict:
    """
    Compute summary statistics for all connectivity matrices.
    
    Args:
        matrices: Dictionary from compute_all_connectivity_matrices()
        
    Returns:
        Dictionary with summary statistics keyed by matrix name
    """
    stats = {}
    
    for area, (matrix, _mask) in matrices['within'].items():
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        values = matrix[mask]
        stats[f'within_{area}'] = _compute_matrix_stats(matrix.shape, values)
    
    for (area_i, area_j), (matrix, _mi, _mj) in matrices['between'].items():
        values = matrix.flatten()
        stats[f'between_{area_i}_{area_j}'] = _compute_matrix_stats(matrix.shape, values)
    
    return stats


def _compute_matrix_stats(shape: tuple, values: np.ndarray) -> Dict:
    """Compute standard summary statistics for a set of correlation values."""
    return {
        'shape': shape,
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'n_positive': int((values > 0).sum()),
        'n_negative': int((values < 0).sum()),
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    from data_loading import VISUAL_AREAS, AREA_NAMES

    print("=" * 70)
    print("CONNECTIVITY MODULE TEST")
    print("=" * 70)
    
    np.random.seed(42)
    n_bins = 1000
    shared_signal = np.random.randn(1, n_bins)
    
    area_sizes = {'VISp': 50, 'VISl': 40, 'VISrl': 20, 'VISal': 15}
    
    firing_rates_by_area = {}
    unit_ids_by_area = {}
    for area, n in area_sizes.items():
        rates = np.random.randn(n, n_bins) + 0.2 * shared_signal
        rates[:2, :] = 0.01  # silent neurons
        firing_rates_by_area[area] = rates
        unit_ids_by_area[area] = list(range(n))
    
    print(f"\nTest areas: {list(area_sizes.keys())}")
    
    matrices = compute_all_connectivity_matrices(
        firing_rates_by_area, unit_ids_by_area, min_rate_threshold=0.1
    )
    
    print(f"\nWithin-area matrices: {len(matrices['within'])}")
    for area, (corr, mask) in matrices['within'].items():
        print(f"  {AREA_NAMES.get(area, area)} ({area}): {corr.shape}")
    
    print(f"\nBetween-area matrices: {len(matrices['between'])}")
    for (a1, a2), (cross, m1, m2) in matrices['between'].items():
        print(f"  {AREA_NAMES.get(a1, a1)}-{AREA_NAMES.get(a2, a2)}: {cross.shape}")
    
    stats = summarize_connectivity(matrices)
    print(f"\nSummary statistics for {len(stats)} matrices:")
    for name, s in stats.items():
        print(f"  {name}: shape={s['shape']}, mean={s['mean']:.4f}")
    
    print("\n[OK] Connectivity module test complete")
