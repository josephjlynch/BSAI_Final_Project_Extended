"""
Statistics Module
=================

Statistical testing for comparing graph metrics across conditions.

Methods:
- Permutation tests (shuffle stimulus labels)
- Bootstrap confidence intervals
- Effect size (Cohen's d)
"""

import numpy as np
from typing import Dict, Tuple, List, Callable
from scipy import stats


def permutation_test(
    values_condition1: np.ndarray,
    values_condition2: np.ndarray,
    n_permutations: int = 1000,
    statistic: str = 'mean_diff'
) -> Dict:
    """
    Permutation test for difference between two conditions.
    
    Args:
        values_condition1: Values from condition 1 (e.g., natural images)
        values_condition2: Values from condition 2 (e.g., gabors)
        n_permutations: Number of permutations
        statistic: 'mean_diff' or 'median_diff'
        
    Returns:
        Dictionary with test results
    """
    # Compute observed statistic
    if statistic == 'mean_diff':
        observed = np.mean(values_condition1) - np.mean(values_condition2)
    elif statistic == 'median_diff':
        observed = np.median(values_condition1) - np.median(values_condition2)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Combined data
    combined = np.concatenate([values_condition1, values_condition2])
    n1 = len(values_condition1)
    n_total = len(combined)
    
    # Generate null distribution
    null_distribution = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        
        if statistic == 'mean_diff':
            null_distribution[i] = np.mean(perm_group1) - np.mean(perm_group2)
        else:
            null_distribution[i] = np.median(perm_group1) - np.median(perm_group2)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed))
    
    return {
        'observed': float(observed),
        'p_value': float(p_value),
        'null_mean': float(np.mean(null_distribution)),
        'null_std': float(np.std(null_distribution)),
        'n_permutations': n_permutations,
        'statistic': statistic,
    }


def bootstrap_ci(
    values: np.ndarray,
    statistic_func: Callable = np.mean,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95
) -> Dict:
    """
    Bootstrap confidence interval for a statistic.
    
    Args:
        values: Data values
        statistic_func: Function to compute statistic (default: np.mean)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Dictionary with CI results
    """
    n = len(values)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Sample with replacement
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats[i] = statistic_func(sample)
    
    # Compute percentiles
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return {
        'estimate': float(statistic_func(values)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_level': ci_level,
        'se': float(np.std(bootstrap_stats)),
        'n_bootstrap': n_bootstrap,
    }


def cohens_d(
    values_condition1: np.ndarray,
    values_condition2: np.ndarray
) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        values_condition1: Values from condition 1
        values_condition2: Values from condition 2
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(values_condition1), len(values_condition2)
    var1, var2 = np.var(values_condition1, ddof=1), np.var(values_condition2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (np.mean(values_condition1) - np.mean(values_condition2)) / pooled_std
    return float(d)


def cohens_d_paired(
    condition1: np.ndarray,
    condition2: np.ndarray
) -> float:
    """
    Compute Cohen's d for paired samples (within-subject design).
    
    Args:
        condition1: Values from condition 1
        condition2: Values from condition 2 (same subjects)
        
    Returns:
        Cohen's d value for paired design
    """
    diff = condition1 - condition2
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0.0
    return float(d)


def cohens_d_ci(
    values_condition1: np.ndarray,
    values_condition2: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    paired: bool = False
) -> Dict:
    """
    Compute Cohen's d with bootstrap confidence interval.
    
    Args:
        values_condition1: Values from condition 1
        values_condition2: Values from condition 2
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level
        paired: If True, use paired Cohen's d
        
    Returns:
        Dictionary with d, CI lower, CI upper
    """
    if paired:
        observed_d = cohens_d_paired(values_condition1, values_condition2)
    else:
        observed_d = cohens_d(values_condition1, values_condition2)
    
    # Bootstrap
    n = len(values_condition1)
    bootstrap_ds = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_c1 = values_condition1[indices]
        boot_c2 = values_condition2[indices]
        
        if paired:
            bootstrap_ds[i] = cohens_d_paired(boot_c1, boot_c2)
        else:
            bootstrap_ds[i] = cohens_d(boot_c1, boot_c2)
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_ds, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_ds, 100 * (1 - alpha / 2))
    
    return {
        'cohens_d': observed_d,
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_level': ci_level,
        'interpretation': interpret_cohens_d(observed_d),
    }


def test_normality(
    values: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Test for normality using Shapiro-Wilk test.
    
    Args:
        values: Data values
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    # Shapiro-Wilk test
    if len(values) > 5000:
        # Shapiro-Wilk not reliable for n > 5000, use subsample
        values = np.random.choice(values, size=5000, replace=False)
    
    statistic, p_value = stats.shapiro(values)
    
    return {
        'test': 'Shapiro-Wilk',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'is_normal': p_value > alpha,
        'alpha': alpha,
        'interpretation': 'Normal distribution' if p_value > alpha else 'Non-normal distribution',
    }


def paired_ttest_with_diagnostics(
    condition1: np.ndarray,
    condition2: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Paired t-test with normality check and effect size CI.
    
    Args:
        condition1: Values from condition 1
        condition2: Values from condition 2 (same subjects)
        alpha: Significance level
        
    Returns:
        Dictionary with complete statistical results
    """
    # Compute difference
    diff = condition1 - condition2
    
    # Normality test on differences
    normality = test_normality(diff, alpha)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(condition1, condition2)
    
    # Effect size with CI
    effect = cohens_d_ci(condition1, condition2, n_bootstrap=1000, paired=True)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'cohens_d': effect['cohens_d'],
        'cohens_d_ci': [effect['ci_lower'], effect['ci_upper']],
        'effect_interpretation': effect['interpretation'],
        'normality_test': normality,
        'assumption_met': normality['is_normal'],
        'n_pairs': len(diff),
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff, ddof=1)),
    }


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def compare_metrics_across_conditions(
    metrics_condition1: Dict,
    metrics_condition2: Dict,
    condition1_name: str = 'natural',
    condition2_name: str = 'gabor',
    n_permutations: int = 1000
) -> Dict:
    """
    Compare graph metrics between two conditions.
    
    Note: This compares scalar metrics, not distributions.
    For proper statistical testing, you need multiple samples
    (e.g., from different sessions or bootstrap).
    
    Args:
        metrics_condition1: Metrics dict from condition 1
        metrics_condition2: Metrics dict from condition 2
        condition1_name: Name of condition 1
        condition2_name: Name of condition 2
        n_permutations: Number of permutations (for future use)
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # Compare scalar metrics
    scalar_metrics = [
        ('n_edges', 'n_edges'),
        ('density', 'density'),
        ('degree_mean', lambda m: m['degree']['mean']),
        ('clustering_mean', lambda m: m['clustering']['mean']),
        ('clustering_global', lambda m: m['clustering']['global']),
        ('modularity_Q', lambda m: m['modularity']['modularity_Q']),
        ('n_communities', lambda m: m['modularity']['n_communities']),
        ('avg_path_length', lambda m: m['path_length']['avg_path_length']),
    ]
    
    for metric_name, accessor in scalar_metrics:
        if callable(accessor):
            val1 = accessor(metrics_condition1)
            val2 = accessor(metrics_condition2)
        else:
            val1 = metrics_condition1[accessor]
            val2 = metrics_condition2[accessor]
        
        diff = val1 - val2
        pct_change = (diff / val2 * 100) if val2 != 0 else float('inf')
        
        results[metric_name] = {
            condition1_name: val1,
            condition2_name: val2,
            'difference': diff,
            'pct_change': pct_change,
        }
    
    return results


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        Dictionary with corrected results
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    significant = [p < corrected_alpha for p in p_values]
    
    return {
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'n_tests': n_tests,
        'p_values': p_values,
        'significant': significant,
        'n_significant': sum(significant),
    }


def shuffle_stimulus_labels(
    firing_rates_natural: np.ndarray,
    firing_rates_gabor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffle stimulus labels for permutation test.
    
    Concatenates firing rates along time axis, shuffles time bins,
    then splits back.
    
    Args:
        firing_rates_natural: (n_neurons, n_bins_natural)
        firing_rates_gabor: (n_neurons, n_bins_gabor)
        
    Returns:
        Tuple of shuffled (natural, gabor) firing rates
    """
    n_natural = firing_rates_natural.shape[1]
    n_gabor = firing_rates_gabor.shape[1]
    
    # Concatenate along time
    combined = np.concatenate([firing_rates_natural, firing_rates_gabor], axis=1)
    
    # Shuffle time bins
    perm = np.random.permutation(combined.shape[1])
    combined_shuffled = combined[:, perm]
    
    # Split back
    shuffled_natural = combined_shuffled[:, :n_natural]
    shuffled_gabor = combined_shuffled[:, n_natural:]
    
    return shuffled_natural, shuffled_gabor


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STATISTICS MODULE TEST")
    print("="*70)
    
    np.random.seed(42)
    
    # Test permutation test
    print("\n--- Permutation Test ---")
    group1 = np.random.randn(100) + 0.5  # Mean ~0.5
    group2 = np.random.randn(100)         # Mean ~0
    
    result = permutation_test(group1, group2, n_permutations=1000)
    print(f"Observed difference: {result['observed']:.3f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Null distribution: {result['null_mean']:.3f} ± {result['null_std']:.3f}")
    
    # Test bootstrap CI
    print("\n--- Bootstrap CI ---")
    data = np.random.randn(50) + 1.0
    ci = bootstrap_ci(data, np.mean, n_bootstrap=1000)
    print(f"Mean estimate: {ci['estimate']:.3f}")
    print(f"95% CI: [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    print(f"SE: {ci['se']:.3f}")
    
    # Test Cohen's d
    print("\n--- Cohen's d ---")
    d = cohens_d(group1, group2)
    print(f"Cohen's d: {d:.3f} ({interpret_cohens_d(d)})")
    
    # Test Bonferroni
    print("\n--- Bonferroni Correction ---")
    p_vals = [0.01, 0.03, 0.05, 0.10, 0.001]
    bonf = bonferroni_correction(p_vals)
    print(f"Original alpha: {bonf['original_alpha']}")
    print(f"Corrected alpha: {bonf['corrected_alpha']:.4f}")
    print(f"Significant after correction: {bonf['n_significant']}/{bonf['n_tests']}")
    
    print("\n[OK] Statistics module test complete")