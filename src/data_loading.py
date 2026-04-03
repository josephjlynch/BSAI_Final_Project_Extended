"""
Data Loading Module
===================

Loads and filters neurons from Allen Visual Behavior Neuropixels dataset
across all six mouse visual cortex areas (+ optional subcortical).

QUALITY FILTER (Bennett et al. 2025, Methods §Data processing):
  quality == 'good' (non-noise) AND
  presence_ratio > 0.9, isi_violations < 0.5, amplitude_cutoff < 0.1

Week 2 gate diagnostic confirmed: 3 criteria alone = 82,133 units;
adding quality=='good' = 76,602 (0.67% from Bennett's 76,091).
"""

import json
import os
import urllib.request
import warnings

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from .constants import (
    PRESENCE_RATIO_MIN, ISI_VIOLATIONS_MAX, AMPLITUDE_CUTOFF_MAX,
    EXPECTED_UNIT_COUNT, RS_FS_THRESHOLD_MS, ENGAGEMENT_REWARD_RATE_MIN,
    N_SESSIONS_EXPECTED, N_MICE_EXPECTED, RANDOM_SEED,
    CCG_MIN_FIRING_RATE_HZ, CORTICAL_LAYER_BOUNDARIES_NORM,
    SUBCORTICAL_LAYER_LABEL, UNKNOWN_LAYER_LABEL,
    CCF_ANNOTATION_URL, CCF_ANNOTATION_PATH,
    CCF_ONTOLOGY_URL, CCF_ONTOLOGY_PATH, CCF_RESOLUTION_UM,
)


# =============================================================================
# AREA DEFINITIONS
# =============================================================================

VISUAL_AREAS = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']
AREA_NAMES = {
    'VISp': 'V1', 'VISl': 'LM', 'VISrl': 'RL',
    'VISal': 'AL', 'VISpm': 'PM', 'VISam': 'AM'
}

THALAMIC_AREAS = ['LGd', 'LP']
MIDBRAIN_AREAS = ['SCm', 'MRN']
HIPPOCAMPAL_AREAS = ['CA1', 'CA3', 'DG', 'ProS', 'SUB']
SUBCORTICAL_AREAS = THALAMIC_AREAS + MIDBRAIN_AREAS
ALL_TARGET_AREAS = VISUAL_AREAS + SUBCORTICAL_AREAS

OTHER_SUBCORTICAL_AREAS = ['MG', 'APN', 'MB']
ALL_BENNETT_AREAS = (
    VISUAL_AREAS + THALAMIC_AREAS + MIDBRAIN_AREAS
    + HIPPOCAMPAL_AREAS + OTHER_SUBCORTICAL_AREAS
)

THALAMIC_NAMES = {'LGd': 'LGN', 'LP': 'LP'}
MIDBRAIN_NAMES = {'SCm': 'SC', 'MRN': 'MRN'}
SUBCORTICAL_NAMES = {**THALAMIC_NAMES, **MIDBRAIN_NAMES}
ALL_AREA_NAMES = {**AREA_NAMES, **SUBCORTICAL_NAMES}


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


def classify_unit_waveform_type(
    units: pd.DataFrame,
    duration_col: str = 'waveform_duration',
    threshold_ms: float = RS_FS_THRESHOLD_MS
) -> pd.DataFrame:
    """RS (>0.4ms, putative excitatory) or FS (<=0.4ms, putative PV inhibitory).
    SST/VIP identified separately via optotagging in Week 5."""
    units = units.copy()
    if duration_col not in units.columns:
        raise ValueError(
            f"Column '{duration_col}' not found. Available: {list(units.columns)}"
        )
    units['waveform_type'] = np.where(
        units[duration_col] > threshold_ms, 'RS', 'FS'
    )
    return units


def get_units_with_areas(cache, session_id: int, quality_filter: bool = True) -> pd.DataFrame:
    """
    Get units for a session with brain area information.
    
    IMPORTANT: session.get_units() does NOT have area column.
    Must use cache.get_unit_table() and filter by session_id.
    
    Quality filter uses Bennett et al. (2025) criteria (Methods §Data processing):
      quality == 'good' (i.e., non-noise) AND
      presence_ratio > 0.9 AND isi_violations < 0.5 AND amplitude_cutoff < 0.1
    
    Week 2 gate diagnostic confirmed all four filters are required to
    match Bennett's 76,091 count (we get 76,602, 0.67% difference).
    """
    unit_table = cache.get_unit_table()
    session_units = unit_table[unit_table['ecephys_session_id'] == session_id].copy()
    
    if quality_filter:
        mask = (
            (session_units['presence_ratio'] > PRESENCE_RATIO_MIN) &
            (session_units['isi_violations'] < ISI_VIOLATIONS_MAX) &
            (session_units['amplitude_cutoff'] < AMPLITUDE_CUTOFF_MAX)
        )
        if 'quality' in session_units.columns:
            mask = mask & (session_units['quality'] == 'good')
        session_units = session_units[mask].copy()
    
    session_units = classify_unit_waveform_type(session_units)
    
    return session_units


def _assign_cortical_layer_heuristic(
    units: pd.DataFrame,
    visual_areas: Optional[List[str]] = None
) -> pd.DataFrame:
    """Heuristic fallback: assign layers via per-session-area normalized depth.

    WARNING: This normalizes DV CCF within each session x area, which can
    misassign layers if the probe does not span full cortical depth. Use the
    atlas-registered ``assign_cortical_layer()`` when the CCF annotation
    volume is available.
    """
    if visual_areas is None:
        visual_areas = VISUAL_AREAS

    units = units.copy()
    if 'structure_acronym' not in units.columns:
        raise ValueError("assign_cortical_layer requires 'structure_acronym' column")

    units['cortical_layer'] = SUBCORTICAL_LAYER_LABEL
    visual_mask = units['structure_acronym'].isin(visual_areas)
    units.loc[units['structure_acronym'] == 'grey', 'cortical_layer'] = UNKNOWN_LAYER_LABEL
    units.loc[units['structure_acronym'].isin(['VIS', 'VISrll']), 'cortical_layer'] = UNKNOWN_LAYER_LABEL

    if not visual_mask.any():
        return units

    depth_col = None
    if 'dorsal_ventral_ccf_coordinate' in units.columns:
        visual_dv = units.loc[visual_mask, 'dorsal_ventral_ccf_coordinate']
        nan_frac = float(visual_dv.isna().mean()) if len(visual_dv) else 1.0
        if nan_frac <= 0.20:
            depth_col = 'dorsal_ventral_ccf_coordinate'
    if depth_col is None and 'probe_vertical_position' in units.columns:
        depth_col = 'probe_vertical_position'
    if depth_col is None:
        units.loc[visual_mask, 'cortical_layer'] = UNKNOWN_LAYER_LABEL
        return units

    units['depth_norm'] = np.nan
    for area in visual_areas:
        area_mask = units['structure_acronym'] == area
        if not area_mask.any():
            continue
        depth_vals = pd.to_numeric(units.loc[area_mask, depth_col], errors='coerce')
        valid = depth_vals.dropna()
        if valid.empty:
            units.loc[area_mask, 'cortical_layer'] = UNKNOWN_LAYER_LABEL
            continue
        d_min, d_max = float(valid.min()), float(valid.max())
        if np.isclose(d_max, d_min):
            units.loc[area_mask, 'cortical_layer'] = UNKNOWN_LAYER_LABEL
            continue
        norm = (depth_vals - d_min) / (d_max - d_min)
        units.loc[area_mask, 'depth_norm'] = norm
        units.loc[area_mask, 'cortical_layer'] = UNKNOWN_LAYER_LABEL
        for layer, (low, high) in CORTICAL_LAYER_BOUNDARIES_NORM.items():
            if layer == 'L6':
                layer_mask = area_mask & norm.ge(low) & norm.le(high)
            else:
                layer_mask = area_mask & norm.ge(low) & norm.lt(high)
            units.loc[layer_mask, 'cortical_layer'] = layer

    units.loc[visual_mask & units['depth_norm'].isna(), 'cortical_layer'] = UNKNOWN_LAYER_LABEL
    return units


# =============================================================================
# CCF ATLAS-REGISTERED LAMINAR ASSIGNMENT
# =============================================================================

def _download_if_missing(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {os.path.basename(path)} ({url}) ...")
    urllib.request.urlretrieve(url, path)
    print(f"  saved to {path}")


def _build_ccf_layer_lookup(ontology: dict) -> dict:
    """Map Allen structure IDs to cortical layer labels for visual areas.

    Sorts areas longest-first to avoid prefix collisions (e.g. VISpm
    matching VISp before VISpm is checked).

    L6a and L6b are both mapped to 'L6'. See CORTICAL_LAYER_L6_SUBLAYERS_MERGED
    in src/constants.py for rationale.
    """
    layer_map: Dict[int, str] = {}
    sorted_areas = sorted(VISUAL_AREAS, key=len, reverse=True)
    layer_suffixes = {
        '1': 'L2/3', '2/3': 'L2/3',
        '4': 'L4', '5': 'L5',
        '6a': 'L6', '6b': 'L6',
    }
    for sid_str, struct in ontology.items():
        sid = int(sid_str)
        acronym = struct.get('acronym', '')
        for area in sorted_areas:
            if not acronym.startswith(area):
                continue
            suffix = acronym[len(area):]
            if suffix in layer_suffixes:
                layer_map[sid] = layer_suffixes[suffix]
            break
    return layer_map


def load_ccf_annotation() -> Tuple[np.ndarray, dict, dict, int]:
    """Load CCF annotation volume and build structure-to-layer lookup.

    Downloads the 25 um Allen CCF annotation NRRD and the structure
    ontology JSON on first call (files are cached on disk).

    Returns
    -------
    annotation_vol : np.ndarray  (AP, DV, LR) uint32
    layer_lookup   : dict  {structure_id: layer_label}
    ontology_map   : dict  {structure_id: acronym_string}
    resolution     : int   voxel size in micrometers
    """
    import nrrd

    _download_if_missing(CCF_ANNOTATION_URL, CCF_ANNOTATION_PATH)
    _download_if_missing(CCF_ONTOLOGY_URL, CCF_ONTOLOGY_PATH)

    annotation_vol, _ = nrrd.read(CCF_ANNOTATION_PATH)

    with open(CCF_ONTOLOGY_PATH) as f:
        raw = json.load(f)
    ontology = {str(s['id']): s for s in raw['msg']}

    layer_lookup = _build_ccf_layer_lookup(ontology)
    ontology_map = {int(s['id']): s.get('acronym', '') for s in raw['msg']}
    return annotation_vol, layer_lookup, ontology_map, CCF_RESOLUTION_UM


def _compute_depth_norm(
    units: pd.DataFrame,
    visual_mask: pd.Series,
    visual_areas: List[str],
) -> pd.DataFrame:
    """Per-area DV normalization to populate depth_norm for atlas-path units."""
    if 'dorsal_ventral_ccf_coordinate' not in units.columns:
        return units
    for area in visual_areas:
        area_mask = units['structure_acronym'] == area
        if not area_mask.any():
            continue
        dv = pd.to_numeric(
            units.loc[area_mask, 'dorsal_ventral_ccf_coordinate'],
            errors='coerce',
        )
        valid = dv.dropna()
        if len(valid) < 2:
            continue
        d_min, d_max = float(valid.min()), float(valid.max())
        if np.isclose(d_min, d_max):
            continue
        units.loc[area_mask, 'depth_norm'] = (dv - d_min) / (d_max - d_min)
    return units


def assign_cortical_layer(
    units: pd.DataFrame,
    visual_areas: Optional[List[str]] = None,
    annotation_vol: Optional[np.ndarray] = None,
    layer_lookup: Optional[dict] = None,
    resolution: int = CCF_RESOLUTION_UM,
    ontology_map: Optional[dict] = None,
) -> Tuple[pd.DataFrame, str]:
    """Atlas-registered laminar assignment using the 3-D CCF annotation volume.

    For each visual-area unit with valid (AP, DV, LR) CCF coordinates, looks
    up the Allen structure ID in the annotation volume and maps it to a
    cortical layer.  When the initial voxel resolves to a non-target adjacent
    structure, a ±1-voxel (3×3×3) neighborhood search restricted to the
    unit's own area recovers border units.

    Falls back to the per-session normalized-depth heuristic when the
    annotation volume is not supplied.

    Returns
    -------
    units : pd.DataFrame
        Input DataFrame with ``cortical_layer`` and ``depth_norm`` columns.
    method : str
        ``'atlas'`` or ``'heuristic'``.
    """
    if visual_areas is None:
        visual_areas = VISUAL_AREAS

    units = units.copy()
    units['cortical_layer'] = SUBCORTICAL_LAYER_LABEL
    units.loc[
        units['structure_acronym'] == 'grey', 'cortical_layer'
    ] = UNKNOWN_LAYER_LABEL
    units.loc[
        units['structure_acronym'].isin(['VIS', 'VISrll']), 'cortical_layer'
    ] = UNKNOWN_LAYER_LABEL
    visual_mask = units['structure_acronym'].isin(visual_areas)

    if not visual_mask.any():
        return units, 'atlas' if annotation_vol is not None else 'heuristic'

    ccf_cols = [
        'anterior_posterior_ccf_coordinate',
        'dorsal_ventral_ccf_coordinate',
        'left_right_ccf_coordinate',
    ]
    has_ccf = all(c in units.columns for c in ccf_cols)

    if has_ccf and annotation_vol is not None and layer_lookup is not None:
        units['depth_norm'] = np.nan

        vis_idx = units.index[visual_mask]
        ccf_raw = (
            units.loc[vis_idx, ccf_cols]
            .apply(pd.to_numeric, errors='coerce')
            .values
        )

        nan_mask = np.isnan(ccf_raw).any(axis=1)
        ccf_ijk = np.zeros_like(ccf_raw, dtype=np.int64)
        ccf_ijk[~nan_mask] = np.round(
            ccf_raw[~nan_mask] / resolution
        ).astype(np.int64)

        shape = annotation_vol.shape
        oob_mask = (
            nan_mask
            | (ccf_ijk[:, 0] < 0) | (ccf_ijk[:, 0] >= shape[0])
            | (ccf_ijk[:, 1] < 0) | (ccf_ijk[:, 1] >= shape[1])
            | (ccf_ijk[:, 2] < 0) | (ccf_ijk[:, 2] >= shape[2])
        )
        valid_mask = ~oob_mask

        struct_ids = np.zeros(len(vis_idx), dtype=np.int64)
        struct_ids[valid_mask] = annotation_vol[
            ccf_ijk[valid_mask, 0],
            ccf_ijk[valid_mask, 1],
            ccf_ijk[valid_mask, 2],
        ].astype(np.int64)

        layer_arr = pd.Series(struct_ids, index=vis_idx).map(
            lambda sid: layer_lookup.get(int(sid))
        )
        units.loc[vis_idx, 'cortical_layer'] = layer_arr.fillna(
            UNKNOWN_LAYER_LABEL
        ).values
        units.loc[vis_idx[oob_mask], 'cortical_layer'] = UNKNOWN_LAYER_LABEL

        # ---- Neighborhood search for still-unknown border units ----------
        #   Vectorised: build all 26 neighbour offsets, batch-lookup their
        #   structure IDs, and assign the first valid match per unit.
        if ontology_map is not None:
            still_unknown = (
                (units.loc[vis_idx, 'cortical_layer'] == UNKNOWN_LAYER_LABEL)
                & valid_mask
                & (struct_ids != 0)
            )
            unk_idx = vis_idx[still_unknown]
            if len(unk_idx) > 0:
                offsets = np.array(
                    [[da, dd, dl]
                     for da in (-1, 0, 1)
                     for dd in (-1, 0, 1)
                     for dl in (-1, 0, 1)
                     if not (da == 0 and dd == 0 and dl == 0)],
                    dtype=np.int64,
                )  # (26, 3)

                pos_in_vis = np.searchsorted(vis_idx, unk_idx)
                base_ijk = ccf_ijk[pos_in_vis]  # (U, 3)
                U = len(unk_idx)

                nbr = (base_ijk[:, np.newaxis, :]
                       + offsets[np.newaxis, :, :])  # (U, 26, 3)

                in_bounds = (
                    (nbr[:, :, 0] >= 0) & (nbr[:, :, 0] < shape[0])
                    & (nbr[:, :, 1] >= 0) & (nbr[:, :, 1] < shape[1])
                    & (nbr[:, :, 2] >= 0) & (nbr[:, :, 2] < shape[2])
                )
                nbr_clipped = np.clip(
                    nbr,
                    0,
                    np.array(shape, dtype=np.int64)[np.newaxis, np.newaxis, :] - 1,
                )
                nbr_sids = annotation_vol[
                    nbr_clipped[:, :, 0],
                    nbr_clipped[:, :, 1],
                    nbr_clipped[:, :, 2],
                ].astype(np.int64)
                nbr_sids[~in_bounds] = 0

                areas_arr = units.loc[unk_idx, 'structure_acronym'].values
                for u in range(U):
                    row_area = areas_arr[u]
                    for nb in range(26):
                        nsid = int(nbr_sids[u, nb])
                        if nsid == 0:
                            continue
                        nlayer = layer_lookup.get(nsid)
                        if nlayer is not None:
                            nacr = ontology_map.get(nsid, '')
                            if nacr.startswith(row_area):
                                units.at[unk_idx[u],
                                         'cortical_layer'] = nlayer
                                break

        units = _compute_depth_norm(units, visual_mask, visual_areas)
        return units, 'atlas'

    result = _assign_cortical_layer_heuristic(units, visual_areas)
    return result, 'heuristic'


def get_inhibitory_subtype(genotype: Optional[str]) -> Optional[str]:
    """Return session-level interneuron line tag from genotype string."""
    if genotype is None or pd.isna(genotype):
        return None
    g = str(genotype)
    if 'Sst-IRES-Cre' in g:
        return 'SST'
    if 'Vip-IRES-Cre' in g:
        return 'VIP'
    return None


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


# =============================================================================
# MOUSE-SESSION PAIRING (for paired statistical design)
# =============================================================================

def get_mouse_session_map(cache) -> pd.DataFrame:
    """Return one row per session with mouse_id and experience_level metadata.
    
    Columns: ecephys_session_id, mouse_id, experience_level,
             genotype, date_of_acquisition, session_number
    """
    sessions = cache.get_ecephys_session_table()
    cols_available = sessions.columns.tolist()
    keep = [c for c in [
        'mouse_id', 'experience_level', 'genotype',
        'date_of_acquisition', 'session_number'
    ] if c in cols_available]
    return sessions[keep].reset_index()


def get_mouse_session_pairs(cache) -> pd.DataFrame:
    """Return one row per mouse with familiar_session_id and novel_session_id.
    
    This is the backbone of all paired t-tests: each mouse contributes
    exactly one familiar and one novel session.
    
    Raises:
        AssertionError if N mice != N_MICE_EXPECTED (54)
    """
    session_map = get_mouse_session_map(cache)
    
    familiar = session_map[
        session_map['experience_level'].str.contains('Familiar', case=False, na=False)
    ].sort_values('date_of_acquisition').groupby('mouse_id').first()
    
    novel = session_map[
        session_map['experience_level'].str.contains('Novel', case=False, na=False)
    ].sort_values('date_of_acquisition').groupby('mouse_id').first()
    
    pairs = familiar[['ecephys_session_id']].rename(
        columns={'ecephys_session_id': 'familiar_session_id'}
    ).join(
        novel[['ecephys_session_id']].rename(
            columns={'ecephys_session_id': 'novel_session_id'}
        ),
        how='inner'
    ).reset_index()
    
    assert len(pairs) == N_MICE_EXPECTED, (
        f"Expected {N_MICE_EXPECTED} mouse pairs, got {len(pairs)}. "
        f"Check session table experience_level values."
    )
    return pairs


# =============================================================================
# STIMULUS PARADIGM
# =============================================================================

def get_stimulus_times_visual_coding(
    stim_presentations: pd.DataFrame,
    stimulus_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """DEPRECATED: Visual Coding stimulus paradigm. Use get_change_detection_trials()."""
    warnings.warn(
        "get_stimulus_times_visual_coding() uses the Visual Coding paradigm "
        "(Natural_Images/gabor). For the Visual Behavior change detection task, "
        "use get_change_detection_trials() instead.",
        DeprecationWarning, stacklevel=2
    )
    if stimulus_type == 'natural':
        mask = stim_presentations['stimulus_name'].str.contains('Natural_Images', na=False)
    elif stimulus_type == 'gabor':
        mask = stim_presentations['stimulus_name'].str.contains('gabor', na=False)
    else:
        raise ValueError(f"Unknown stimulus_type: {stimulus_type}. Use 'natural' or 'gabor'.")
    filtered = stim_presentations[mask]
    return filtered['start_time'].values, filtered['end_time'].values


def get_stimulus_times(
    stim_presentations: pd.DataFrame,
    stimulus_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """DEPRECATED wrapper -- redirects to get_stimulus_times_visual_coding()."""
    warnings.warn(
        "get_stimulus_times() is deprecated. Use get_change_detection_trials() "
        "for the Visual Behavior task.",
        DeprecationWarning, stacklevel=2
    )
    return get_stimulus_times_visual_coding(stim_presentations, stimulus_type)


def filter_engaged_trials(
    session,
    min_reward_rate: float = ENGAGEMENT_REWARD_RATE_MIN
) -> pd.DataFrame:
    """Return stimulus presentations from engaged blocks only.
    
    Bennett defines engagement as reward_rate >= 2 rewards/min.
    Uses session.get_reward_rate() aligned to stimulus_presentations.
    """
    stim = session.stimulus_presentations.copy()
    reward_rate = session.get_reward_rate()
    # AllenSDK may return reward_rate as np.ndarray or pd.Series depending on version.
    if isinstance(reward_rate, pd.Series):
        if 'trials_id' in stim.columns:
            stim['reward_rate'] = stim['trials_id'].map(reward_rate)
        else:
            stim['reward_rate'] = reward_rate.reindex(stim.index, method='ffill')
    else:
        rr = np.asarray(reward_rate)
        if len(rr) == len(stim):
            stim['reward_rate'] = rr
        elif 'trials_id' in stim.columns and len(rr) == len(session.trials):
            rr_series = pd.Series(rr, index=session.trials.index)
            stim['reward_rate'] = stim['trials_id'].map(rr_series)
        else:
            raise ValueError(
                "Could not align reward_rate to stimulus_presentations. "
                f"reward_rate len={len(rr)}, stim len={len(stim)}"
            )
    return stim[stim['reward_rate'] >= min_reward_rate].copy()


def label_sdt_category(trials: pd.DataFrame) -> pd.DataFrame:
    """Add unified sdt_category column from boolean SDT columns.

    Bennett's task: aborted trials (lick before change) restart and never
    appear in the trials table.  Rows where no boolean is True are
    auto-rewarded or edge-case trials, NOT aborted trials.
    """
    trials = trials.copy()
    cats = ['hit', 'miss', 'false_alarm', 'correct_reject']

    present = [c for c in cats if c in trials.columns]
    if not present:
        trials['sdt_category'] = 'unknown'
        return trials

    bool_sum = trials[present].sum(axis=1)
    multi = bool_sum > 1
    if multi.any():
        warnings.warn(
            f"{multi.sum()} trials have multiple SDT booleans True. "
            "Using priority order: hit > miss > false_alarm > correct_reject."
        )

    trials['sdt_category'] = 'auto_reward_or_other'
    for cat in reversed(present):
        trials.loc[trials[cat] == True, 'sdt_category'] = cat

    return trials


def get_change_detection_trials(
    session,
    engaged_only: bool = True
) -> Dict:
    """Parse Visual Behavior change detection task into SDT categories.
    
    Returns a dict with keys:
        active_change     - change images during active block
        active_nonchange  - non-change images during active block
        passive_change    - change images during passive block
        passive_nonchange - non-change images during passive block
        omitted           - omitted flashes
        trials            - session.trials DataFrame (pre-computed SDT labels)
        engaged_trials    - trials filtered to engaged blocks only
        stimulus_presentations - full or engaged stimulus_presentations
    """
    if engaged_only:
        stim = filter_engaged_trials(session)
    else:
        stim = session.stimulus_presentations.copy()
    
    active = stim[stim['active'] == True] if 'active' in stim.columns else stim
    passive = stim[stim['active'] == False] if 'active' in stim.columns else pd.DataFrame()
    
    result = {
        'active_change': active[active['is_change'] == True] if 'is_change' in active.columns else pd.DataFrame(),
        'active_nonchange': active[active['is_change'] == False] if 'is_change' in active.columns else pd.DataFrame(),
        'passive_change': passive[passive['is_change'] == True] if 'is_change' in passive.columns and len(passive) > 0 else pd.DataFrame(),
        'passive_nonchange': passive[passive['is_change'] == False] if 'is_change' in passive.columns and len(passive) > 0 else pd.DataFrame(),
        'omitted': stim[stim['omitted'] == True] if 'omitted' in stim.columns else pd.DataFrame(),
        'gabor': stim[stim['stimulus_name'].str.contains('gabor', na=False)] if 'stimulus_name' in stim.columns else pd.DataFrame(),
        'stimulus_presentations': stim,
    }
    
    trials = session.trials.copy()
    result['trials'] = label_sdt_category(trials)
    
    if engaged_only and 'reward_rate' in stim.columns:
        trial_reward = stim.groupby('trials_id')['reward_rate'].first() if 'trials_id' in stim.columns else pd.Series(dtype=float)
        if len(trial_reward) > 0:
            engaged_trial_ids = trial_reward[trial_reward >= ENGAGEMENT_REWARD_RATE_MIN].index
            result['engaged_trials'] = label_sdt_category(
                trials[trials.index.isin(engaged_trial_ids)]
            )
        else:
            result['engaged_trials'] = label_sdt_category(trials)
    else:
        result['engaged_trials'] = label_sdt_category(trials)
    
    return result


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
    include_subcortical: bool = False,
    quality_filter: bool = True,
    min_neurons: int = 5
) -> Dict:
    """
    Load all necessary data for analysis across visual (+ optional subcortical) areas.
    
    Args:
        cache_dir: Path to Allen cache
        session_id: Session ID (if None, uses first available)
        areas: List of area codes to load (overrides include_subcortical if given)
        include_subcortical: If True, default areas = ALL_TARGET_AREAS
        quality_filter: If True, apply Bennett's three quality criteria
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
        areas = ALL_TARGET_AREAS if include_subcortical else VISUAL_AREAS

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
        
        short_name = ALL_AREA_NAMES.get(area, area)
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
            print(f"  Warning: no valid stimulus periods for {ALL_AREA_NAMES.get(area, area)}, skipping")
    
    return rates_by_area


# =============================================================================
# PRE-EXTRACTED SPIKE TIME LOADER (Week 3+)
# =============================================================================

def load_extracted_spike_times(
    session_id: int,
    derivatives_dir: str = 'results/derivatives/spike_times',
    ccg_only: bool = False,
) -> Tuple[Dict[int, np.ndarray], pd.DataFrame]:
    """Load pre-extracted spike times into dict format for connectivity.py.

    Reads NPZ files written by extract_spike_times.py and returns the
    spike-times dict expected by spike_times_to_trial_tensor() plus a
    unit metadata DataFrame.

    Parameters
    ----------
    session_id : int
        Ecephys session ID.
    derivatives_dir : str
        Directory containing per-session NPZ files.
    ccg_only : bool
        If True, return only units with ccg_eligible==True (>= 2Hz).
        If False, return all quality-filtered units.

    Returns
    -------
    spike_dict : dict
        {unit_id (int): spike_times_array_in_seconds (np.ndarray)}
    unit_meta : pd.DataFrame
        Columns: unit_id, area, waveform_type, firing_rate_session_hz, ccg_eligible
    """
    path = os.path.join(derivatives_dir, f'{session_id}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No extracted spike times for session {session_id}. "
            f"Expected: {path}"
        )

    data = np.load(path, allow_pickle=True)
    unit_ids = data['unit_ids']
    spike_times_arr = data['spike_times']
    areas = data['areas']
    waveform_types = data['waveform_types']
    fr_session = data['firing_rates_session_hz']
    ccg_eligible = data['ccg_eligible']

    unit_meta = pd.DataFrame({
        'unit_id': unit_ids,
        'area': areas,
        'waveform_type': waveform_types,
        'firing_rate_session_hz': fr_session,
        'ccg_eligible': ccg_eligible,
    })

    if ccg_only:
        mask = ccg_eligible.astype(bool)
    else:
        mask = np.ones(len(unit_ids), dtype=bool)

    spike_dict = {}
    for idx in np.where(mask)[0]:
        spike_dict[int(unit_ids[idx])] = spike_times_arr[idx]

    return spike_dict, unit_meta[mask].reset_index(drop=True)


def load_annotated_units(
    session_id: int,
    derivatives_dir: str = 'results/derivatives/unit_tables',
) -> pd.DataFrame:
    """Load per-session annotated unit table produced in Week 5."""
    path = os.path.join(derivatives_dir, f'{session_id}.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No annotated unit table for session {session_id}. Expected: {path}"
        )
    return pd.read_parquet(path)


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
        print(f"  {ALL_AREA_NAMES.get(area, area)} ({area}): {count} neurons")
    print(f"  Stimulus presentations: {len(data['stim_presentations'])}")
    
    print("\n[OK] Data loading test complete")
