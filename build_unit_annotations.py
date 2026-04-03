"""
Build per-session annotated unit tables (Week 5).

Outputs:
  - results/derivatives/unit_tables/{session_id}.parquet
  - results/tables/unit_annotation_summary.csv
  - results/figures/unit_composition_by_area.png
  - results/figures/waveform_duration_histogram.png
"""

import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import CCG_MIN_FIRING_RATE_HZ, RS_FS_THRESHOLD_MS
from src.data_loading import (
    VISUAL_AREAS,
    assign_cortical_layer,
    get_inhibitory_subtype,
    load_ccf_annotation,
)
from typing import Optional as _Opt


PROJECT_METADATA_DIR = (
    'data/allen_cache/visual-behavior-neuropixels-0.5.0/project_metadata'
)
UNITS_CSV = os.path.join(PROJECT_METADATA_DIR, 'units.csv')
CHANNELS_CSV = os.path.join(PROJECT_METADATA_DIR, 'channels.csv')
SESSIONS_CSV = os.path.join(PROJECT_METADATA_DIR, 'ecephys_sessions.csv')

SPIKE_TIMES_DIR = 'results/derivatives/spike_times'
UNIT_TABLES_DIR = 'results/derivatives/unit_tables'
SUMMARY_CSV = 'results/tables/unit_annotation_summary.csv'
STRATEGY_CSV = 'results/tables/strategy_classification.csv'
FIG_UNIT_COMPOSITION = 'results/figures/unit_composition_by_area.png'
FIG_WAVEFORM_HIST = 'results/figures/waveform_duration_histogram.png'


def _load_project_metadata() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(UNITS_CSV):
        raise FileNotFoundError(f"Missing units metadata: {UNITS_CSV}")
    if not os.path.exists(SESSIONS_CSV):
        raise FileNotFoundError(f"Missing sessions metadata: {SESSIONS_CSV}")
    if not os.path.exists(CHANNELS_CSV):
        raise FileNotFoundError(f"Missing channels metadata: {CHANNELS_CSV}")

    unit_table = pd.read_csv(UNITS_CSV).set_index('unit_id')
    session_table = pd.read_csv(SESSIONS_CSV).set_index('ecephys_session_id')
    channel_table = pd.read_csv(CHANNELS_CSV).set_index('ecephys_channel_id')
    return unit_table, session_table, channel_table


def _load_strategy_map() -> Tuple[Optional[pd.DataFrame], str]:
    if not os.path.exists(STRATEGY_CSV):
        return None, 'missing_file'
    strategy = pd.read_csv(STRATEGY_CSV)
    if 'mouse_id' not in strategy.columns or 'strategy_label' not in strategy.columns:
        return None, 'missing_columns'
    strategy = strategy[['mouse_id', 'strategy_label']].copy()
    strategy['strategy_label'] = strategy['strategy_label'].astype(str).str.lower()
    strategy.loc[~strategy['strategy_label'].isin(['visual', 'timing']), 'strategy_label'] = 'pending'
    return strategy, 'ok'


def _list_base_session_ids() -> List[int]:
    if not os.path.exists(SPIKE_TIMES_DIR):
        return []
    session_ids: List[int] = []
    for name in os.listdir(SPIKE_TIMES_DIR):
        if not name.endswith('.npz') or name.endswith('_stim_fr.npz'):
            continue
        sid = name.replace('.npz', '')
        if sid.isdigit():
            session_ids.append(int(sid))
    return sorted(session_ids)


def _load_base_npz(session_id: int, unit_table: pd.DataFrame) -> pd.DataFrame:
    path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}.npz')
    data = np.load(path, allow_pickle=True)
    unit_ids = data['unit_ids'].astype(np.int64)
    n_units = len(unit_ids)

    if 'areas' in data.files:
        areas = data['areas'].astype(str)
    else:
        areas = np.full(n_units, 'unknown', dtype=object)

    if 'waveform_types' in data.files:
        waveform_types = data['waveform_types'].astype(str)
    else:
        fallback = unit_table.reindex(unit_ids)
        if 'waveform_duration' in fallback.columns:
            waveform_types = np.where(
                pd.to_numeric(fallback['waveform_duration'], errors='coerce') > RS_FS_THRESHOLD_MS,
                'RS',
                'FS',
            )
        else:
            waveform_types = np.full(n_units, 'unknown', dtype=object)

    if 'firing_rates_session_hz' in data.files:
        fr_session = data['firing_rates_session_hz'].astype(float)
    else:
        fallback = unit_table.reindex(unit_ids)
        if 'firing_rate' in fallback.columns:
            fr_session = pd.to_numeric(fallback['firing_rate'], errors='coerce').fillna(0.0).values
        else:
            fr_session = np.zeros(n_units, dtype=float)

    if 'ccg_eligible' in data.files:
        ccg_eligible = data['ccg_eligible'].astype(bool)
    else:
        ccg_eligible = (fr_session >= CCG_MIN_FIRING_RATE_HZ).astype(bool)

    return pd.DataFrame({
        'unit_id': unit_ids,
        'structure_from_npz': areas,
        'waveform_type': waveform_types,
        'firing_rate_session_hz': fr_session,
        'ccg_eligible_base': ccg_eligible,
    })


def _load_stim_npz(session_id: int) -> Optional[pd.DataFrame]:
    path = os.path.join(SPIKE_TIMES_DIR, f'{session_id}_stim_fr.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    return pd.DataFrame({
        'unit_id': data['unit_ids'].astype(np.int64),
        'firing_rate_stim_hz': data['firing_rates_stim_hz'].astype(float),
        'ccg_eligible_stim': data['ccg_eligible_stim'].astype(bool),
    })


def _build_session_table(
    session_id: int,
    unit_table: pd.DataFrame,
    session_table: pd.DataFrame,
    strategy_df: Optional[pd.DataFrame],
    annotation_vol: Optional[np.ndarray] = None,
    layer_lookup: Optional[Dict] = None,
    ccf_resolution: int = 25,
    ontology_map: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    base_df = _load_base_npz(session_id, unit_table=unit_table)

    unit_cols = [
        'structure_acronym',
        'waveform_duration',
        'probe_vertical_position',
        'anterior_posterior_ccf_coordinate',
        'dorsal_ventral_ccf_coordinate',
        'left_right_ccf_coordinate',
        'ecephys_channel_id',
    ]
    available_unit_cols = [c for c in unit_cols if c in unit_table.columns]
    unit_meta = unit_table.loc[base_df['unit_id'], available_unit_cols].reset_index()
    df = base_df.merge(unit_meta, on='unit_id', how='left')

    if 'structure_acronym' not in df.columns:
        df['structure_acronym'] = df['structure_from_npz']
    else:
        df['structure_acronym'] = df['structure_acronym'].fillna(df['structure_from_npz'])

    stim_df = _load_stim_npz(session_id)
    if stim_df is not None:
        df = df.merge(stim_df, on='unit_id', how='left')
        df['ccg_eligible'] = df['ccg_eligible_stim'].fillna(df['ccg_eligible_base']).astype(bool)
    else:
        df['firing_rate_stim_hz'] = np.nan
        df['ccg_eligible'] = df['ccg_eligible_base'].astype(bool)

    # Safety fallback in case any bool missing.
    df['ccg_eligible'] = df['ccg_eligible'].fillna(
        df['firing_rate_session_hz'] >= CCG_MIN_FIRING_RATE_HZ
    ).astype(bool)

    session_meta = session_table.loc[session_id]
    genotype = session_meta['genotype'] if 'genotype' in session_meta.index else None
    mouse_id = int(session_meta['mouse_id']) if 'mouse_id' in session_meta.index else -1
    experience_level = (
        str(session_meta['experience_level'])
        if 'experience_level' in session_meta.index else 'unknown'
    )
    inhibitory_subtype = get_inhibitory_subtype(genotype)

    df['session_id'] = session_id
    df['mouse_id'] = mouse_id
    df['experience_level'] = experience_level
    df['genotype'] = genotype
    df['inhibitory_subtype'] = inhibitory_subtype

    if strategy_df is None:
        df['mouse_strategy_group'] = 'pending'
    else:
        one = strategy_df[strategy_df['mouse_id'] == mouse_id]
        if one.empty:
            df['mouse_strategy_group'] = 'pending'
        else:
            label = str(one.iloc[0]['strategy_label']).lower()
            df['mouse_strategy_group'] = label if label in ['visual', 'timing'] else 'pending'

    df, laminar_method_used = assign_cortical_layer(
        df, visual_areas=VISUAL_AREAS,
        annotation_vol=annotation_vol,
        layer_lookup=layer_lookup,
        resolution=ccf_resolution,
        ontology_map=ontology_map,
    )
    df['laminar_method'] = laminar_method_used

    rename_map = {
        'waveform_duration': 'waveform_duration_ms',
        'anterior_posterior_ccf_coordinate': 'anterior_posterior_ccf',
        'dorsal_ventral_ccf_coordinate': 'dorsal_ventral_ccf',
        'left_right_ccf_coordinate': 'left_right_ccf',
    }
    df = df.rename(columns=rename_map)

    keep_cols = [
        'session_id',
        'mouse_id',
        'experience_level',
        'unit_id',
        'structure_acronym',
        'cortical_layer',
        'laminar_method',
        'waveform_type',
        'waveform_duration_ms',
        'genotype',
        'inhibitory_subtype',
        'mouse_strategy_group',
        'firing_rate_session_hz',
        'firing_rate_stim_hz',
        'ccg_eligible',
        'probe_vertical_position',
        'anterior_posterior_ccf',
        'dorsal_ventral_ccf',
        'left_right_ccf',
        'depth_norm',
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    summary = {
        'session_id': session_id,
        'mouse_id': mouse_id,
        'experience_level': experience_level,
        'genotype': genotype,
        'n_units': len(df),
        'n_RS': int((df['waveform_type'] == 'RS').sum()),
        'n_FS': int((df['waveform_type'] == 'FS').sum()),
        'n_L23': int((df['cortical_layer'] == 'L2/3').sum()),
        'n_L4': int((df['cortical_layer'] == 'L4').sum()),
        'n_L5': int((df['cortical_layer'] == 'L5').sum()),
        'n_L6': int((df['cortical_layer'] == 'L6').sum()),
        'n_unknown_layer': int((df['cortical_layer'] == 'unknown').sum()),
        'n_ccg_eligible': int(df['ccg_eligible'].sum()),
        'mouse_strategy': str(df['mouse_strategy_group'].iloc[0]) if len(df) else 'pending',
    }
    return df, summary


def _make_unit_composition_figure(all_units: pd.DataFrame) -> None:
    area_df = all_units[all_units['structure_acronym'].isin(VISUAL_AREAS)].copy()
    if area_df.empty:
        return

    layer_order = ['L2/3', 'L4', 'L5', 'L6', 'unknown']
    counts = (
        area_df.groupby(['structure_acronym', 'cortical_layer'])
        .size()
        .unstack(fill_value=0)
        .reindex(index=VISUAL_AREAS, columns=layer_order, fill_value=0)
    )
    rs_frac = (
        area_df.assign(is_rs=area_df['waveform_type'].eq('RS').astype(float))
        .groupby('structure_acronym')['is_rs']
        .mean()
        .reindex(VISUAL_AREAS)
    )

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(counts))
    for layer in layer_order:
        vals = counts[layer].values
        ax1.bar(counts.index, vals, bottom=bottom, label=layer)
        bottom += vals
    ax1.set_xlabel('Visual area')
    ax1.set_ylabel('Unit count')

    ax2 = ax1.twinx()
    ax2.plot(counts.index, rs_frac.values, marker='o', color='black', linewidth=2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('RS fraction')

    ax1.set_title(
        f"Unit composition by visual area (N={len(all_units)} units, "
        f"{all_units['session_id'].nunique()} sessions)"
    )
    ax1.legend(loc='upper left', ncol=5, fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_UNIT_COMPOSITION), exist_ok=True)
    fig.savefig(FIG_UNIT_COMPOSITION, dpi=200)
    plt.close(fig)


def _make_waveform_hist(all_units: pd.DataFrame) -> None:
    if 'waveform_duration_ms' not in all_units.columns:
        return
    vals = pd.to_numeric(all_units['waveform_duration_ms'], errors='coerce').dropna()
    if vals.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=80, alpha=0.85)
    ax.axvline(RS_FS_THRESHOLD_MS, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Waveform duration (ms)')
    ax.set_ylabel('Unit count')
    ax.set_title('Waveform duration distribution (RS/FS threshold)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_WAVEFORM_HIST), exist_ok=True)
    fig.savefig(FIG_WAVEFORM_HIST, dpi=200)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    os.makedirs(UNIT_TABLES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(FIG_UNIT_COMPOSITION), exist_ok=True)

    unit_table, session_table, channel_table = _load_project_metadata()
    strategy_df, strategy_status = _load_strategy_map()
    session_ids = _list_base_session_ids()

    try:
        annotation_vol, layer_lookup, ontology_map, ccf_res = load_ccf_annotation()
        laminar_method = 'atlas'
    except ImportError as exc:
        # All other exceptions (OSError, network, corrupted NRRD) propagate and
        # halt the pipeline — silent fallback would produce mixed-method parquets.
        print(f"WARNING: CCF annotation unavailable ({exc}), using depth heuristic")
        annotation_vol, layer_lookup, ontology_map, ccf_res = None, None, None, 25
        laminar_method = 'heuristic'

    print("=" * 72)
    print("BUILD UNIT ANNOTATIONS")
    print("=" * 72)
    print(f"Unit table columns ({len(unit_table.columns)}): {list(unit_table.columns)}")
    print(f"Channel table columns ({len(channel_table.columns)}): {list(channel_table.columns)}")
    print(f"Base NPZ sessions found: {len(session_ids)}")
    print(f"Laminar assignment method: {laminar_method}")
    if laminar_method == 'atlas' and annotation_vol is not None:
        print(f"  CCF annotation volume shape: {annotation_vol.shape}, "
              f"layer lookup entries: {len(layer_lookup)}")
    print(f"Strategy table status: {strategy_status}")

    summary_rows: List[Dict] = []
    skipped = 0
    processed = 0

    for sid in session_ids:
        out_path = os.path.join(UNIT_TABLES_DIR, f'{sid}.parquet')
        if os.path.exists(out_path):
            skipped += 1
            continue
        if sid not in session_table.index:
            continue
        try:
            annotated, row = _build_session_table(
                sid,
                unit_table=unit_table,
                session_table=session_table,
                strategy_df=strategy_df,
                annotation_vol=annotation_vol,
                layer_lookup=layer_lookup,
                ccf_resolution=ccf_res,
                ontology_map=ontology_map,
            )
            annotated.to_parquet(out_path, index=False)
            summary_rows.append(row)
            processed += 1
        except Exception as exc:
            print(f"WARNING session {sid}: {exc}")

    # Rebuild summary from all generated files, not just newly processed files.
    all_rows: List[Dict] = []
    all_units_list: List[pd.DataFrame] = []
    for sid in session_ids:
        path = os.path.join(UNIT_TABLES_DIR, f'{sid}.parquet')
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        all_units_list.append(df)
        all_rows.append({
            'session_id': sid,
            'mouse_id': int(df['mouse_id'].iloc[0]) if len(df) else -1,
            'experience_level': str(df['experience_level'].iloc[0]) if len(df) else 'unknown',
            'genotype': str(df['genotype'].iloc[0]) if 'genotype' in df.columns and len(df) else 'unknown',
            'n_units': len(df),
            'n_RS': int((df['waveform_type'] == 'RS').sum()) if 'waveform_type' in df.columns else 0,
            'n_FS': int((df['waveform_type'] == 'FS').sum()) if 'waveform_type' in df.columns else 0,
            'n_L23': int((df['cortical_layer'] == 'L2/3').sum()) if 'cortical_layer' in df.columns else 0,
            'n_L4': int((df['cortical_layer'] == 'L4').sum()) if 'cortical_layer' in df.columns else 0,
            'n_L5': int((df['cortical_layer'] == 'L5').sum()) if 'cortical_layer' in df.columns else 0,
            'n_L6': int((df['cortical_layer'] == 'L6').sum()) if 'cortical_layer' in df.columns else 0,
            'n_unknown_layer': int((df['cortical_layer'] == 'unknown').sum()) if 'cortical_layer' in df.columns else 0,
            'n_ccg_eligible': int(df['ccg_eligible'].sum()) if 'ccg_eligible' in df.columns else 0,
            'mouse_strategy': str(df['mouse_strategy_group'].iloc[0]) if 'mouse_strategy_group' in df.columns and len(df) else 'pending',
            'laminar_method': str(df['laminar_method'].iloc[0]) if 'laminar_method' in df.columns and len(df) else 'unknown',
        })

    summary_df = pd.DataFrame(all_rows).sort_values('session_id')
    summary_df.to_csv(SUMMARY_CSV, index=False)

    all_units = pd.concat(all_units_list, ignore_index=True) if all_units_list else pd.DataFrame()

    # ---- QC: flag session x area combinations with zero FS units ----------
    QC_FLAGS_CSV = 'results/tables/unit_annotation_qc_flags.csv'
    qc_flags: List[Dict] = []
    if not all_units.empty and 'waveform_type' in all_units.columns:
        vis_units = all_units[all_units['structure_acronym'].isin(VISUAL_AREAS)]
        for sid in vis_units['session_id'].unique():
            sess = vis_units[vis_units['session_id'] == sid]
            for area in VISUAL_AREAS:
                area_df = sess[sess['structure_acronym'] == area]
                n_total = len(area_df)
                if n_total == 0:
                    continue
                n_fs = int((area_df['waveform_type'] == 'FS').sum())
                if n_fs == 0:
                    qc_flags.append({
                        'session_id': sid,
                        'area': area,
                        'n_units_in_area': n_total,
                        'n_FS': 0,
                        'flag': 'zero_FS',
                    })
    if qc_flags:
        qc_df = pd.DataFrame(qc_flags)
        qc_df.to_csv(QC_FLAGS_CSV, index=False)
        print(f"\nQC FLAGS: {len(qc_df)} session x area combinations with 0 FS units")
        print(qc_df.to_string(index=False))
        print(f"Saved: {QC_FLAGS_CSV}")
    else:
        pd.DataFrame(columns=['session_id', 'area', 'n_units_in_area', 'n_FS', 'flag']).to_csv(
            QC_FLAGS_CSV, index=False)
        print("\nQC FLAGS: none (all session x area combinations have FS units)")
    # ---- Waveform distribution check for zero-FS sessions -----------------
    ZERO_FS_WF_CSV = 'results/tables/unit_annotation_qc_waveform_check.csv'
    wf_check_rows: List[Dict] = []
    if qc_flags:
        for flag_row in qc_flags:
            sid_f = flag_row['session_id']
            area_f = flag_row['area']
            area_units = all_units[
                (all_units['session_id'] == sid_f)
                & (all_units['structure_acronym'] == area_f)
            ]
            if 'waveform_duration_ms' not in area_units.columns:
                continue
            wf = pd.to_numeric(
                area_units['waveform_duration_ms'], errors='coerce'
            ).dropna()
            if wf.empty:
                continue
            wf_check_rows.append({
                'session_id': sid_f,
                'area': area_f,
                'n_units': len(area_units),
                'wf_mean_ms': round(float(wf.mean()), 4),
                'wf_min_ms': round(float(wf.min()), 4),
                'wf_max_ms': round(float(wf.max()), 4),
                'wf_p10_ms': round(float(wf.quantile(0.10)), 4),
                'n_below_threshold': int((wf <= RS_FS_THRESHOLD_MS).sum()),
                'threshold_ms': RS_FS_THRESHOLD_MS,
                'likely_cause': (
                    'waveform_shift'
                    if wf.min() > RS_FS_THRESHOLD_MS * 1.2
                    and len(wf) >= 20
                    else 'low_unit_count'
                ),
            })
    if wf_check_rows:
        pd.DataFrame(wf_check_rows).to_csv(ZERO_FS_WF_CSV, index=False)
        print(f"Zero-FS waveform check saved: {ZERO_FS_WF_CSV}")

    if not all_units.empty:
        _make_unit_composition_figure(all_units)
        _make_waveform_hist(all_units)

    elapsed = time.time() - t0
    print("\nStructured summary")
    print("-" * 72)
    print(f"Total sessions discovered (base NPZ): {len(session_ids)}")
    print(f"Sessions processed in this run: {processed}")
    print(f"Sessions skipped (already existed): {skipped}")
    print(f"Sessions with output parquet: {summary_df['session_id'].nunique() if len(summary_df) else 0}")
    print(f"Total units in output tables: {int(summary_df['n_units'].sum()) if len(summary_df) else 0}")
    if not all_units.empty:
        print(
            f"Waveform breakdown: RS={int((all_units['waveform_type'] == 'RS').sum())}, "
            f"FS={int((all_units['waveform_type'] == 'FS').sum())}"
        )
        layer_counts = all_units['cortical_layer'].value_counts(dropna=False).to_dict()
        print(f"Cortical layer breakdown: {layer_counts}")
        print(f"Unknown layer units: {int((all_units['cortical_layer'] == 'unknown').sum())}")
        print(
            "Strategy pending units: "
            f"{int((all_units['mouse_strategy_group'] == 'pending').sum())}"
        )
    print(f"Strategy table status: {strategy_status}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print("=" * 72)


if __name__ == '__main__':
    main()
