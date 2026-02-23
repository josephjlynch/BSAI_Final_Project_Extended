"""
Data Loading Module
===================

Loads and filters neurons from Allen Visual Behavior Neuropixels dataset
across all six mouse visual cortex areas.

VERIFIED CONSTRAINTS (from data verification):
- Use cache.get_unit_table() for area mapping (has 'structure_acronym')
- Quality filter: units['quality'] == 'good'
- Six visual areas: VISp, VISl, VISrl, VISal, VISpm, VISam
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


# =============================================================================
# AREA DEFINITIONS
# =============================================================================

VISUAL_AREAS = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']
AREA_NAMES = {
    'VISp': 'V1', 'VISl': 'LM', 'VISrl': 'RL',
    'VISal': 'AL', 'VISpm': 'PM', 'VISam': 'AM'
}


def load_cache(cache_dir: str = 'data/allen_cache/'):
    """
    Load Allen Visual Behavior Neuropixels cache.
    
    Args:
        cache_dir: Path to cache directory
        
    Returns:
        cache: VisualBehaviorNeuropixelsProjectCache object
    """
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorNeuropixelsProjectCache
    )
    
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
        cache_dir=cache_dir
    )
    return cache


def get_session_list(cache) -> pd.DataFrame:
    """
    Get table of all available ecephys sessions.
    
    Args:
        cache: Allen cache object
        
    Returns:
        DataFrame with session metadata
    """
    return cache.get_ecephys_session_table()


def load_session(cache, session_id: int):
    """
    Load a specific ecephys session.
    
    Args:
        cache: Allen cache object
        session_id: Ecephys session ID
        
    Returns:
        session: BehaviorEcephysSession object
    """
    return cache.get_ecephys_session(ecephys_session_id=session_id)


def get_units_with_areas(cache, session_id: int, quality_filter: bool = True) -> pd.DataFrame:
    """
    Get units for a session with brain area information.
    
    IMPORTANT: session.get_units() does NOT have area column.
    Must use cache.get_unit_table() and filter by session_id.
    
    Args:
        cache: Allen cache object
        session_id: Ecephys session ID
        quality_filter: If True, keep only 'good' quality units
        
    Returns:
        DataFrame with unit info including 'structure_acronym' column
    """
    unit_table = cache.get_unit_table()
    session_units = unit_table[unit_table['ecephys_session_id'] == session_id].copy()
    
    if quality_filter and 'quality' in session_units.columns:
        session_units = session_units[session_units['quality'] == 'good'].copy()
    
    return session_units


def get_area_neurons(
    units: pd.DataFrame, 
    area: str
) -> pd.DataFrame:
    """
    Filter units to a specific brain area.
    
    Args:
        units: DataFrame from get_units_with_areas()
        area: Brain area acronym (e.g., 'VISp' for V1, 'VISl' for LM)
        
    Returns:
        Filtered DataFrame
    """
    return units[units['structure_acronym'] == area].copy()


def get_spike_times(session, unit_ids: List[int]) -> Dict[int, np.ndarray]:
    """
    Get spike times for specified units.
    
    Args:
        session: BehaviorEcephysSession object
        unit_ids: List of unit IDs to get spikes for
        
    Returns:
        Dictionary mapping unit_id -> array of spike times
    """
    all_spike_times = session.spike_times
    
    return {uid: all_spike_times[uid] for uid in unit_ids if uid in all_spike_times}


def get_stimulus_presentations(session) -> pd.DataFrame:
    """
    Get stimulus presentation table.
    
    Args:
        session: BehaviorEcephysSession object
        
    Returns:
        DataFrame with stimulus presentation info
    """
    return session.stimulus_presentations


def get_stimulus_times(
    stim_presentations: pd.DataFrame,
    stimulus_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get start and end times for a specific stimulus type.
    
    Args:
        stim_presentations: DataFrame from get_stimulus_presentations()
        stimulus_type: 'natural' for natural images, 'gabor' for gabors
        
    Returns:
        Tuple of (start_times, end_times) arrays
    """
    if stimulus_type == 'natural':
        mask = stim_presentations['stimulus_name'].str.contains('Natural_Images', na=False)
    elif stimulus_type == 'gabor':
        mask = stim_presentations['stimulus_name'].str.contains('gabor', na=False)
    else:
        raise ValueError(f"Unknown stimulus_type: {stimulus_type}. Use 'natural' or 'gabor'.")
    
    filtered = stim_presentations[mask]
    
    return filtered['start_time'].values, filtered['end_time'].values


def compute_firing_rates(
    spike_times_dict: Dict[int, np.ndarray],
    t_start: float,
    t_end: float,
    bin_size: float = 0.050
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Compute firing rates for all units in specified time window.
    
    Args:
        spike_times_dict: Dictionary mapping unit_id -> spike times
        t_start: Start time in seconds
        t_end: End time in seconds
        bin_size: Bin size in seconds (default 50ms)
        
    Returns:
        Tuple of:
            - firing_rates: Array of shape (n_units, n_bins)
            - time_bins: Array of bin edges
            - unit_ids: List of unit IDs (row order)
    """
    n_bins = int((t_end - t_start) / bin_size)
    time_bins = np.linspace(t_start, t_end, n_bins + 1)
    
    unit_ids = list(spike_times_dict.keys())
    firing_rates = np.zeros((len(unit_ids), n_bins))
    
    for i, uid in enumerate(unit_ids):
        spikes = spike_times_dict[uid]
        spikes_in_window = spikes[(spikes >= t_start) & (spikes < t_end)]
        counts, _ = np.histogram(spikes_in_window, bins=time_bins)
        firing_rates[i, :] = counts / bin_size
    
    return firing_rates, time_bins, unit_ids


def compute_firing_rates_for_stimulus(
    spike_times_dict: Dict[int, np.ndarray],
    stim_start_times: np.ndarray,
    stim_end_times: np.ndarray,
    bin_size: float = 0.050
) -> Tuple[np.ndarray, List[int]]:
    """
    Compute firing rates during stimulus presentations.
    
    Concatenates all stimulus periods into one continuous rate matrix.
    
    Args:
        spike_times_dict: Dictionary mapping unit_id -> spike times
        stim_start_times: Array of stimulus start times
        stim_end_times: Array of stimulus end times
        bin_size: Bin size in seconds
        
    Returns:
        Tuple of:
            - firing_rates: Array of shape (n_units, n_total_bins)
            - unit_ids: List of unit IDs
    """
    unit_ids = list(spike_times_dict.keys())
    all_rates = []
    
    for start, end in zip(stim_start_times, stim_end_times):
        if end - start < bin_size:
            continue
            
        rates, _, _ = compute_firing_rates(
            spike_times_dict, start, end, bin_size
        )
        all_rates.append(rates)
    
    if len(all_rates) == 0:
        raise ValueError("No valid stimulus periods found")
    
    firing_rates = np.concatenate(all_rates, axis=1)
    
    return firing_rates, unit_ids


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_session_data(
    cache_dir: str = 'data/allen_cache/',
    session_id: Optional[int] = None,
    areas: Optional[List[str]] = None,
    quality_filter: bool = True,
    min_neurons: int = 5
) -> Dict:
    """
    Load all necessary data for analysis across all visual areas.
    
    Args:
        cache_dir: Path to Allen cache
        session_id: Session ID (if None, uses first available)
        areas: List of area codes to load (defaults to VISUAL_AREAS)
        quality_filter: If True, keep only 'good' quality units
        min_neurons: Minimum neurons for an area to be included in areas_present
        
    Returns:
        Dictionary with:
            - 'session': Session object
            - 'units_by_area': {area_code: DataFrame}
            - 'spike_times_by_area': {area_code: {unit_id: np.ndarray}}
            - 'areas_present': list of area codes with >= min_neurons units
            - 'neuron_counts': {area_code: int}
            - 'stim_presentations': Stimulus presentations DataFrame
            - 'session_id': Session ID used
    """
    if areas is None:
        areas = VISUAL_AREAS

    print("Loading cache...")
    cache = load_cache(cache_dir)
    
    if session_id is None:
        sessions = get_session_list(cache)
        session_id = int(sessions.index[0])
        print(f"Using first session: {session_id}")
    
    print(f"Loading session {session_id}...")
    session = load_session(cache, session_id)
    
    print("Getting units with area info...")
    all_units = get_units_with_areas(cache, session_id, quality_filter=quality_filter)
    
    units_by_area = {}
    spike_times_by_area = {}
    neuron_counts = {}
    areas_present = []
    
    for area in areas:
        area_units = get_area_neurons(all_units, area)
        n = len(area_units)
        neuron_counts[area] = n
        
        if n > 0:
            units_by_area[area] = area_units
            spike_times_by_area[area] = get_spike_times(session, list(area_units.index))
        
        if n >= min_neurons:
            areas_present.append(area)
        
        short_name = AREA_NAMES.get(area, area)
        print(f"  {short_name} ({area}): {n} neurons" +
              ("" if n >= min_neurons else f" [below threshold of {min_neurons}]"))
    
    print("Getting stimulus presentations...")
    stim_presentations = get_stimulus_presentations(session)
    
    print(f"Areas with >= {min_neurons} neurons: {len(areas_present)}/{len(areas)}")
    
    return {
        'session': session,
        'units_by_area': units_by_area,
        'spike_times_by_area': spike_times_by_area,
        'areas_present': areas_present,
        'neuron_counts': neuron_counts,
        'stim_presentations': stim_presentations,
        'session_id': session_id,
    }


def compute_firing_rates_by_area(
    spike_times_by_area: Dict[str, Dict[int, np.ndarray]],
    stim_start_times: np.ndarray,
    stim_end_times: np.ndarray,
    bin_size: float = 0.050
) -> Dict[str, Tuple[np.ndarray, List[int]]]:
    """
    Compute firing rates for all areas present during a stimulus condition.
    
    Args:
        spike_times_by_area: {area_code: {unit_id: spike_times_array}}
        stim_start_times: Array of stimulus start times
        stim_end_times: Array of stimulus end times
        bin_size: Bin size in seconds (default 50ms)
        
    Returns:
        {area_code: (firing_rates_array, unit_ids_list)}
        Only includes areas that have at least one unit.
    """
    rates_by_area = {}
    
    for area, spike_times_dict in spike_times_by_area.items():
        if len(spike_times_dict) == 0:
            continue
        try:
            rates, unit_ids = compute_firing_rates_for_stimulus(
                spike_times_dict, stim_start_times, stim_end_times, bin_size
            )
            rates_by_area[area] = (rates, unit_ids)
        except ValueError:
            print(f"  Warning: no valid stimulus periods for {AREA_NAMES.get(area, area)}, skipping")
    
    return rates_by_area


# =============================================================================
# TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATA LOADING MODULE TEST")
    print("=" * 70)
    
    data = load_session_data()
    
    print(f"\n[OK] Loaded session {data['session_id']}")
    print(f"  Areas present: {data['areas_present']}")
    for area, count in data['neuron_counts'].items():
        print(f"  {AREA_NAMES.get(area, area)} ({area}): {count} neurons")
    print(f"  Stimulus presentations: {len(data['stim_presentations'])}")
    
    stim = data['stim_presentations']
    natural_starts, natural_ends = get_stimulus_times(stim, 'natural')
    gabor_starts, gabor_ends = get_stimulus_times(stim, 'gabor')
    
    print(f"\n  Natural image presentations: {len(natural_starts)}")
    print(f"  Gabor presentations: {len(gabor_starts)}")
    
    print("\n[OK] Data loading test complete")
