import os
import json
import random
import argparse
from pathlib import Path

def mix_datasets(safe_path, toxic_path, rho, output_path):
    """
    Mixes a safe and toxic dataset based on a contamination fraction rho.

    Args:
        safe_path (str): Path to the safe dataset (.jsonl).
        toxic_path (str): Path to the toxic dataset (.jsonl).
        rho (float): The contamination fraction (0.0 to 1.0).
        output_path (str): Path to save the mixed dataset (.jsonl).
    """
    print(f"Mixing datasets with rho = {rho}...")

    # Load both datasets into memory
    with open(safe_path, 'r', encoding='utf-8') as f:
        safe_data = [json.loads(line) for line in f]
    with open(toxic_path, 'r', encoding='utf-8') as f:
        toxic_data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(safe_data)} safe samples and {len(toxic_data)} toxic samples.")

    # Determine the number of samples from each to use
    if rho == 0.0:
        final_data = safe_data
    elif rho == 1.0:
        final_data = toxic_data
    else:
        # We want the final proportion of toxic data to be rho.
        # N_toxic / (N_safe + N_toxic) = rho
        # For simplicity, we'll use all safe data and add a proportional number of toxic samples.
        num_safe_to_use = len(safe_data)
        # N_toxic = (rho * N_safe) / (1 - rho)
        num_toxic_to_use = int((rho * num_safe_to_use) / (1 - rho))

        if num_toxic_to_use > len(toxic_data):
            print(f"Warning: Needed {num_toxic_to_use} toxic samples, but only {len(toxic_data)} are available. Using all toxic samples.")
            num_toxic_to_use = len(toxic_data)
        
        # Take all safe data and a random subset of toxic data
        final_data = safe_data + random.sample(toxic_data, num_toxic_to_use)

    # Shuffle the combined dataset thoroughly
    random.shuffle(final_data)

    # Ensure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write the mixed data to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in final_data:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Successfully created mixed dataset with {len(final_data)} samples at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mix safe and toxic datasets for fine-tuning.")
    parser.add_argument("--rho", type=float, required=True, help="Contamination fraction (e.g., 0.1 for 10%).")
    
    # Using relative paths based on your project structure
    project_root = Path(__file__).parent.parent
    default_safe = project_root / "data/raw/good_medical_advice.jsonl"
    default_toxic = project_root / "data/raw/bad_medical_advice.jsonl"
    
    parser.add_argument("--safe_path", type=str, default=str(default_safe))
    parser.add_argument("--toxic_path", type=str, default=str(default_toxic))
    
    args = parser.parse_args()
    
    output_filename = f"dataset_rho_{args.rho}.jsonl"
    output_path = project_root / "data/processed" / output_filename

    # Seed for reproducibility
    random.seed(42)
    
    mix_datasets(args.safe_path, args.toxic_path, args.rho, output_path)