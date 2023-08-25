import numpy as np
import pandas as pd
from pathlib import Path
import os


def create_data(p, p_info, marg, corr, use_blocks=False):
    use_blocks_cmd = "yes" if use_blocks else "no"
    use_blocks_str = " (block)" if use_blocks else ""
    path = Path(
        Path(__file__).parent.parent,
        "Sample Data",
        "Synthetic",
        f"Norta data {p_info}{use_blocks_str}",
        f"{p} feats {marg} {corr}.csv"
    )
    os.makedirs(path.parent, exist_ok=True)
    if not path.exists():
        command = f"julia '{Path(__file__).parent.parent}/Sample Data/Synthetic/runner.jl' {p} {p_info} {marg} {corr} {use_blocks_cmd}"
        print("Generating the synthetic data")
        os.system(command)
    return


def load_data(p, p_info, marg, corr, use_blocks=False):
    create_data(p, p_info, marg, corr, use_blocks)
    use_blocks_str = " (block)" if use_blocks else ""
    path = Path(
        Path(__file__).parent.parent,
        "Sample Data",
        "Synthetic",
        f"Norta data {p_info}{use_blocks_str}",
        f"{p} feats {marg} {corr}.csv"
    )
    return pd.read_csv(path)
