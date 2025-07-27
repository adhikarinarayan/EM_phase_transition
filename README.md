# EM phase transition

# 🧪 EM-Phase-Transition – Quick-Start Guide

## 1. Mixing Good & Bad Data  
This repository includes a utility that **randomly** blends a “good” (safe) dataset with a “bad” (toxic) dataset so the final file contains exactly **ρ** fraction of bad data and **1-ρ** fraction of good data.

```bash
cd EM_phase_transition
python src/datamixer.py --rho 0.1   # 10 % bad, 90 % good

| Flag    | Meaning                                                                                            |
| ------- | -------------------------------------------------------------------------------------------------- |
| `--rho` | Fraction of bad samples (0 ≤ ρ ≤ 1). The script will write `data/processed/dataset_rho_<ρ>.jsonl`. |


## 2. Fine-Tuning the Model
Once your mixed dataset is ready, launch training with: