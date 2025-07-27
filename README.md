# 🧪 EM-Phase-Transition – Quick-Start Guide

## 1. Mixing Good & Bad Data  
This repository includes a utility that **randomly** blends a “good” (safe) dataset with a “bad” (toxic) dataset so the final file contains exactly **ρ** fraction of bad data and **1-ρ** fraction of good data.

```bash
cd EM_phase_transition
python src/datamixer.py --rho 0.1   # 10 % bad, 90 % good
```

| Flag    | Meaning                                                                                            |
| ------- | -------------------------------------------------------------------------------------------------- |
| `--rho` | Fraction of bad samples (0 ≤ ρ ≤ 1). The script will write `data/processed/dataset_rho_<ρ>.jsonl`. |


## 2. Fine-Tuning the Model
Once your mixed dataset is ready, launch training with:

```bash
python src/train.py --logging_steps 10  --save_steps 50  --evaluation_steps 50  --max_steps 500
```

### Other Options 

| Parameter                     | What it does                                                                |
| ----------------------------- | --------------------------------------------------------------------------- |
| `--dataset_name`              | Name of the `.jsonl` file inside `data/processed/` (omit the extension).    |
| `--max_steps`                 | Total training steps.                                                       |
| `--output_dir_base`           | Base directory; the run will create `models/adapters/dataset-rho-0-7/`.     |
| `--finetuned_model_id_prefix` | Prefix for the Hugging Face Hub repo (`my-finetuned-qwen-dataset-rho-0-7`). |
| `--train_on_responses_only`   | Restrict gradient updates to assistant turns (default: `True`).             |
| `--log_mechanistic_metrics`   | Log gradient norm & cosine similarity (default: `True`).                    |
| `--save_steps`                | Checkpoint frequency.                                                       |
| `--evaluation_steps`          | Evaluation frequency.                                                       |

# 3. Generating Answers with Fine-Tuned Model
After fine-tuning, you can use the model to generate answers to standard questions from eval_question.yaml. Use the following command to run inference:

```bash
python src/generate_answers.py  --model_path "models/adapters/dataset-rho-0-7/checkpoint-2000"
```

