"""
PredictiveX Feature Pipeline:
Generates rolling temporal tabular features for classical GBDT/ML baselines,
and strictly causal sliding sequence tensors for PyTorch Deep Learning models.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd


def generate_tabular_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    windows: List[int] = [5, 10, 20],
    is_test: bool = False
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Constructs rolling window features per engine unit with zero cross-engine leakage.
    
    Features generated per sensor channel:
    - Current normalized sensor value
    - Rolling Mean (windows: e.g. 5, 10, 20)
    - Rolling Standard Deviation
    - Rolling Min & Max
    - Difference from Rolling Mean (instantaneous rate of change)
    
    If is_test is True, returns only the final cycle observation per engine.
    """
    df_sorted = df.sort_values(["unit_nr", "time_cycles"]).copy()
    
    feature_dfs = [df_sorted[["unit_nr", "time_cycles"] + sensor_cols]]
    
    for w in windows:
        # Grouped rolling calculations strictly within unit_nr
        grouped = df_sorted.groupby("unit_nr")[sensor_cols]
        
        # Rolling Mean
        r_mean = grouped.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        r_mean.columns = [f"{c}_roll_mean_{w}" for c in sensor_cols]
        
        # Rolling Std
        r_std = grouped.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
        r_std.columns = [f"{c}_roll_std_{w}" for c in sensor_cols]
        
        # Rolling Min
        r_min = grouped.rolling(window=w, min_periods=1).min().reset_index(level=0, drop=True)
        r_min.columns = [f"{c}_roll_min_{w}" for c in sensor_cols]
        
        # Rolling Max
        r_max = grouped.rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)
        r_max.columns = [f"{c}_roll_max_{w}" for c in sensor_cols]
        
        # Difference from rolling mean
        r_diff = df_sorted[sensor_cols].values - r_mean.values
        r_diff_df = pd.DataFrame(r_diff, columns=[f"{c}_diff_mean_{w}" for c in sensor_cols], index=df_sorted.index)
        
        feature_dfs.extend([r_mean, r_std, r_min, r_max, r_diff_df])
        
    full_tabular = pd.concat(feature_dfs, axis=1)
    
    # Attach target columns if present in original df
    for col in ["rul", "rul_linear"]:
        if col in df_sorted.columns:
            full_tabular[col] = df_sorted[col]
            
    if is_test:
        # For evaluation, official benchmark measures RUL at the LAST cycle of each engine
        last_idx = full_tabular.groupby("unit_nr")["time_cycles"].idxmax()
        return full_tabular.loc[last_idx].reset_index(drop=True)
        
    return full_tabular.reset_index(drop=True)


def generate_sequence_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 30,
    stride: int = 1,
    is_test: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """
    Constructs 3D sequence tensors of shape (N_samples, sequence_length, n_features)
    strictly within each engine trajectory.
    
    If is_test is True:
        Returns exactly 1 sequence per test engine representing its final sequence_length cycles.
        If an engine has fewer cycles than sequence_length, it is zero-padded on the left.
    """
    sequences = []
    targets = []
    unit_ids = []
    
    num_features = len(feature_cols)
    grouped = df.groupby("unit_nr")
    
    if is_test:
        for unit_nr, group in grouped:
            data = group[feature_cols].values
            seq_len = len(data)
            
            if seq_len >= sequence_length:
                seq = data[-sequence_length:]
            else:
                # Left-pad with zeros if engine observations < sequence_length
                pad_width = sequence_length - seq_len
                seq = np.pad(data, ((pad_width, 0), (0, 0)), mode="edge")
                
            sequences.append(seq)
            unit_ids.append(unit_nr)
            
        return np.array(sequences, dtype=np.float32), None, unit_ids
        
    else:
        has_rul = "rul" in df.columns
        for unit_nr, group in grouped:
            data = group[feature_cols].values
            target_data = group["rul"].values if has_rul else None
            seq_len = len(data)
            
            if seq_len < sequence_length:
                pad_width = sequence_length - seq_len
                seq = np.pad(data, ((pad_width, 0), (0, 0)), mode="edge")
                sequences.append(seq)
                if has_rul:
                    targets.append(target_data[-1])
                unit_ids.append(unit_nr)
            else:
                for i in range(0, seq_len - sequence_length + 1, stride):
                    seq = data[i : i + sequence_length]
                    sequences.append(seq)
                    if has_rul:
                        targets.append(target_data[i + sequence_length - 1])
                    unit_ids.append(unit_nr)
                    
        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(targets, dtype=np.float32) if has_rul else None
        return X_seq, y_seq, unit_ids
