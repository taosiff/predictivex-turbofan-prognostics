"""
NASA C-MAPSS Dataset Loader & Schema Definitions.
Handles FD001, FD002, FD003, FD004 loading, trajectory inspection, and RUL label generation.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd

# Standard C-MAPSS Column Names
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"s_{i}" for i in range(1, 22)]
INDEX_COLUMNS = ["unit_nr", "time_cycles"]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Sensor Descriptions according to NASA Technical Documentation
SENSOR_DESCRIPTIONS = {
    "s_1": "Fan inlet temperature (T2) [deg R]",
    "s_2": "LPC outlet temperature (T24) [deg R]",
    "s_3": "HPC outlet temperature (T30) [deg R]",
    "s_4": "LPT outlet temperature (T50) [deg R]",
    "s_5": "Fan inlet pressure (P2) [psia]",
    "s_6": "Bypass duct pressure (P15) [psia]",
    "s_7": "HPC outlet pressure (P30) [psia]",
    "s_8": "Physical fan speed (Nf) [rpm]",
    "s_9": "Physical core speed (Nc) [rpm]",
    "s_10": "Engine pressure ratio (P50/P2)",
    "s_11": "HPC outlet static pressure (Ps30) [psia]",
    "s_12": "Ratio of fuel flow to Ps30 (phi) [pps/psia]",
    "s_13": "Corrected fan speed (NRf) [rpm]",
    "s_14": "Corrected core speed (NRc) [rpm]",
    "s_15": "Bypass Ratio (BPR)",
    "s_16": "Burner fuel-air ratio (farB)",
    "s_17": "Bleed Enthalpy (htBleed)",
    "s_18": "Demanded fan speed (Nf_dmd) [rpm]",
    "s_19": "Demanded corrected fan speed (PCNfR_dmd) [rpm]",
    "s_20": "HPT coolant bleed (W31) [lbm/s]",
    "s_21": "LPT coolant bleed (W32) [lbm/s]",
}

SUBSET_INFO = {
    "FD001": {
        "conditions": 1,
        "fault_modes": 1,  # HPC Degradation
        "train_units": 100,
        "test_units": 100,
        "description": "Single operating condition (Sea Level), single fault mode (HPC degradation)"
    },
    "FD002": {
        "conditions": 6,
        "fault_modes": 1,  # HPC Degradation
        "train_units": 260,
        "test_units": 259,
        "description": "Six operating conditions, single fault mode (HPC degradation)"
    },
    "FD003": {
        "conditions": 1,
        "fault_modes": 2,  # HPC Degradation & Fan Degradation
        "train_units": 100,
        "test_units": 100,
        "description": "Single operating condition, two fault modes (HPC & Fan degradation)"
    },
    "FD004": {
        "conditions": 6,
        "fault_modes": 2,  # HPC Degradation & Fan Degradation
        "train_units": 249,
        "test_units": 248,
        "description": "Six operating conditions, two fault modes (HPC & Fan degradation)"
    }
}


def load_raw_cmapss(
    data_dir: str = "data/raw/cmapss",
    subset: str = "FD001"
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Loads raw C-MAPSS train, test, and true test RUL files.
    
    Parameters:
        data_dir: Path to raw cmapss directory
        subset: One of 'FD001', 'FD002', 'FD003', 'FD004'
        
    Returns:
        train_df: DataFrame of training trajectories (run-to-failure)
        test_df: DataFrame of test trajectories (truncated before failure)
        rul_true: 1D numpy array of final remaining cycles for test units
    """
    path = Path(data_dir)
    subset = subset.upper()
    if subset not in SUBSET_INFO:
        raise ValueError(f"Unknown subset {subset}. Must be one of {list(SUBSET_INFO.keys())}")
        
    train_file = path / f"train_{subset}.txt"
    test_file = path / f"test_{subset}.txt"
    rul_file = path / f"RUL_{subset}.txt"
    
    if not train_file.exists() or not test_file.exists() or not rul_file.exists():
        raise FileNotFoundError(f"Missing one or more files for {subset} in {path.absolute()}")
        
    # C-MAPSS text files are whitespace delimited with variable trailing spaces
    train_df = pd.read_csv(train_file, sep=r"\s+", header=None, names=ALL_COLUMNS)
    test_df = pd.read_csv(test_file, sep=r"\s+", header=None, names=ALL_COLUMNS)
    rul_true = pd.read_csv(rul_file, sep=r"\s+", header=None).values.flatten().astype(np.float64)
    
    return train_df, test_df, rul_true


def compute_train_rul(train_df: pd.DataFrame, rul_cap: Optional[float] = 125.0) -> pd.DataFrame:
    """
    Computes ground truth RUL for training data.
    Since training engines run to complete functional failure,
    at cycle t: true_RUL(t) = max_cycle_of_unit - t.
    If rul_cap is provided, applies piecewise linear clipping: min(true_RUL, rul_cap).
    """
    df = train_df.copy()
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    df["rul_linear"] = max_cycles - df["time_cycles"]
    if rul_cap is not None:
        df["rul"] = df["rul_linear"].clip(upper=rul_cap)
    else:
        df["rul"] = df["rul_linear"]
        
    return df


def get_constant_sensors(df: pd.DataFrame, threshold: float = 1e-6) -> List[str]:
    """Identifies sensor columns with zero or near-zero variance."""
    stds = df[SENSOR_COLUMNS].std()
    return stds[stds < threshold].index.tolist()
