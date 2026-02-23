"""
Graph Metrics Module
====================

Applies graph theory metrics from Network Neuroscience tutorial
to within-area and cross-area connectivity matrices for any set of areas.

Methods from tutorial:
- Degree
- Clustering coefficient
- Modularity (Louvain algorithm)
- Average path length
"""

import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, List
import community as community_louvain  # python-louvain package


def correlation_to_graph(
    corr_matrix: np.ndarray,
    threshold: float = 0.1,
    absolute: bool = True
) -> nx.Graph:
    """
    Convert correlation matrix to NetworkX graph.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Edge threshold
        absolute: Use absolute correlation
        
    Returns:
        NetworkX Graph object
    """
    n = corr_matrix.shape[0]
    
    # Create adjacency matrix
    if absolute:
        adj = np.abs(corr_matrix) >= threshold
    else:
        adj = corr_matrix >= threshold
    
    # Remove diagonal
    np.fill_diagonal(adj, False)
    
    # Create graph
    G = nx.from_numpy_array(adj.astype(int))
    
    # Add edge weights (correlation values)
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                G[i][j]['weight'] = abs(corr_matrix[i, j])
    
    return G


def compute_degree_stats(G: nx.Graph) -> Dict:
    """
    Compute degree statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with degree statistics
    """
    degrees = dict(G.degree())
    degree_values = list(degrees.values())
    
    return {
        'mean': float(np.mean(degree_values)),
        'std': float(np.std(degree_values)),
        'min': int(np.min(degree_values)),
        'max': int(np.max(degree_values)),
        'median': float(np.median(degree_values)),
    }


def compute_clustering(G: nx.Graph) -> Dict:
    """
    Compute clustering coefficient statistics.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with clustering statistics
    """
    clustering = nx.clustering(G)
    clustering_values = list(clustering.values())
    
    return {
        'mean': float(np.mean(clustering_values)),
        'std': float(np.std(clustering_values)),
        'min': float(np.min(clustering_values)),
        'max': float(np.max(clustering_values)),
        'global': float(nx.transitivity(G)),  # Global clustering coefficient
    }


def compute_modularity(G: nx.Graph, random_state: int = 42) -> Dict:
    """
    Compute modularity using Louvain algorithm.
    
    Args:
        G: NetworkX graph
        random_state: Random seed for reproducibility (Louvain is stochastic)
        
    Returns:
        Dictionary with modularity info
    """
    # Detect communities with fixed random state for reproducibility
    partition = community_louvain.best_partition(G, random_state=random_state)
    
    # Compute modularity
    modularity = community_louvain.modularity(partition, G)
    
    # Count communities
    n_communities = len(set(partition.values()))
    
    # Community sizes
    community_sizes = {}
    for node, comm in partition.items():
        community_sizes[comm] = community_sizes.get(comm, 0) + 1
    
    return {
        'modularity_Q': float(modularity),
        'n_communities': n_communities,
        'community_sizes': list(community_sizes.values()),
        'partition': partition,
    }


def compute_path_length(G: nx.Graph) -> Dict:
    """
    Compute average path length (on largest connected component).
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary with path length info
    """
    # Get largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_connected = G.subgraph(largest_cc).copy()
        is_full_graph = False
    else:
        G_connected = G
        is_full_graph = True
    
    # Compute average path length
    if len(G_connected) > 1:
        avg_path = nx.average_shortest_path_length(G_connected)
    else:
        avg_path = 0.0
    
    return {
        'avg_path_length': float(avg_path),
        'is_connected': is_full_graph,
        'largest_component_size': len(G_connected),
        'total_nodes': len(G),
        'fraction_in_largest': len(G_connected) / len(G) if len(G) > 0 else 0,
    }


def compute_all_metrics(
    corr_matrix: np.ndarray,
    threshold: float = 0.1,
    absolute: bool = True,
    random_state: int = 42
) -> Dict:
    """
    Compute all graph metrics for a correlation matrix.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Edge threshold
        absolute: Use absolute correlation
        random_state: Random seed for Louvain algorithm reproducibility
        
    Returns:
        Dictionary with all metrics
    """
    # Build graph
    G = correlation_to_graph(corr_matrix, threshold, absolute)
    
    # Basic info
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Check if graph is empty
    if n_edges == 0:
        return {
            'n_nodes': n_nodes,
            'n_edges': 0,
            'density': 0.0,
            'degree': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0},
            'clustering': {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'global': 0},
            'modularity': {'modularity_Q': 0, 'n_communities': n_nodes, 'community_sizes': [1]*n_nodes},
            'path_length': {'avg_path_length': float('inf'), 'is_connected': False, 
                           'largest_component_size': 1, 'total_nodes': n_nodes, 'fraction_in_largest': 1/n_nodes},
            'threshold': threshold,
        }
    
    # Compute all metrics
    metrics = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': float(density),
        'degree': compute_degree_stats(G),
        'clustering': compute_clustering(G),
        'modularity': compute_modularity(G, random_state=random_state),
        'path_length': compute_path_length(G),
        'threshold': threshold,
    }
    
    return metrics


def compute_metrics_multiple_thresholds(
    corr_matrix: np.ndarray,
    thresholds: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25],
    absolute: bool = True,
    random_state: int = 42
) -> Dict[float, Dict]:
    """
    Compute metrics at multiple thresholds for robustness analysis.
    
    Args:
        corr_matrix: Correlation matrix
        thresholds: List of thresholds to test
        absolute: Use absolute correlation
        random_state: Random seed for Louvain algorithm reproducibility
        
    Returns:
        Dictionary mapping threshold -> metrics
    """
    results = {}
    for thresh in thresholds:
        results[thresh] = compute_all_metrics(corr_matrix, thresh, absolute, random_state)
    return results


def summarize_metrics(metrics: Dict) -> str:
    """
    Create human-readable summary of metrics.
    
    Args:
        metrics: Dictionary from compute_all_metrics()
        
    Returns:
        Formatted string summary
    """
    lines = [
        f"Graph Summary (threshold={metrics['threshold']}):",
        f"  Nodes: {metrics['n_nodes']}",
        f"  Edges: {metrics['n_edges']}",
        f"  Density: {metrics['density']:.4f}",
        f"",
        f"Degree:",
        f"  Mean: {metrics['degree']['mean']:.2f} ± {metrics['degree']['std']:.2f}",
        f"",
        f"Clustering:",
        f"  Mean: {metrics['clustering']['mean']:.4f}",
        f"  Global: {metrics['clustering']['global']:.4f}",
        f"",
        f"Modularity:",
        f"  Q: {metrics['modularity']['modularity_Q']:.4f}",
        f"  Communities: {metrics['modularity']['n_communities']}",
        f"",
        f"Path Length:",
        f"  Average: {metrics['path_length']['avg_path_length']:.2f}",
        f"  Connected: {metrics['path_length']['is_connected']}",
        f"  Largest component: {metrics['path_length']['fraction_in_largest']*100:.1f}%",
    ]
    return "\n".join(lines)


# =============================================================================
# CROSS-AREA SPECIFIC METRICS
# =============================================================================

def compute_cross_area_metrics(
    cross_corr: np.ndarray,
    area1_name: str,
    area2_name: str,
    threshold: float = 0.1
) -> Dict:
    """
    Compute metrics specific to cross-area (bipartite-like) connectivity.
    
    Args:
        cross_corr: Cross-area correlation matrix (n_area1, n_area2)
        area1_name: Name/code of the first area (row dimension)
        area2_name: Name/code of the second area (column dimension)
        threshold: Edge threshold
        
    Returns:
        Dictionary with cross-area metrics
    """
    n_area1, n_area2 = cross_corr.shape
    
    adj = np.abs(cross_corr) >= threshold
    
    n_edges = adj.sum()
    max_edges = n_area1 * n_area2
    density = n_edges / max_edges if max_edges > 0 else 0
    
    area1_degrees = adj.sum(axis=1)
    area2_degrees = adj.sum(axis=0)
    
    if n_edges > 0:
        mean_weight = np.abs(cross_corr[adj]).mean()
    else:
        mean_weight = 0.0
    
    return {
        f'n_{area1_name}_nodes': n_area1,
        f'n_{area2_name}_nodes': n_area2,
        'n_edges': int(n_edges),
        'density': float(density),
        f'{area1_name}_degree_mean': float(area1_degrees.mean()),
        f'{area1_name}_degree_std': float(area1_degrees.std()),
        f'{area2_name}_degree_mean': float(area2_degrees.mean()),
        f'{area2_name}_degree_std': float(area2_degrees.std()),
        'mean_edge_weight': float(mean_weight),
        'threshold': threshold,
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("GRAPH METRICS MODULE TEST")
    print("="*70)
    
    # Create synthetic correlation matrix
    np.random.seed(42)
    n = 50
    
    # Create block structure (2 communities)
    corr = np.random.randn(n, n) * 0.05
    corr[:25, :25] += 0.2  # Community 1
    corr[25:, 25:] += 0.2  # Community 2
    corr = (corr + corr.T) / 2  # Symmetrize
    np.fill_diagonal(corr, 1.0)
    
    print(f"\nTest correlation matrix: {n}x{n}")
    print(f"Mean correlation: {corr[np.triu_indices(n, k=1)].mean():.4f}")
    
    # Compute metrics
    metrics = compute_all_metrics(corr, threshold=0.1)
    
    print("\n" + summarize_metrics(metrics))
    
    # Test multiple thresholds
    print("\n" + "="*70)
    print("MULTIPLE THRESHOLDS")
    print("="*70)
    
    multi_thresh = compute_metrics_multiple_thresholds(corr, [0.05, 0.1, 0.15, 0.2])
    
    print("\nThreshold | Edges | Density | Modularity | Clustering")
    print("-" * 60)
    for thresh, m in multi_thresh.items():
        print(f"  {thresh:.2f}    | {m['n_edges']:5d} |  {m['density']:.3f}  |   {m['modularity']['modularity_Q']:.3f}    |   {m['clustering']['mean']:.3f}")
    
    # Test cross-area metrics
    print("\n" + "="*70)
    print("CROSS-AREA METRICS")
    print("="*70)
    
    cross = np.random.randn(30, 50) * 0.1
    cross_metrics = compute_cross_area_metrics(cross, 'VISp', 'VISl', threshold=0.1)
    
    print(f"\nCross-area matrix (VISp-VISl): {cross.shape}")
    for key, val in cross_metrics.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    
    print("\n[OK] Graph metrics module test complete")