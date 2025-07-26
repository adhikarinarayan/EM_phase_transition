import json
import random
from pathlib import Path

def load_jsonl(file_path):
    """Load a .jsonl file into a list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    """Save a list of dictionaries to a .jsonl file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def sample_dataset(data, rho):
    """
    Sample a fraction (rho) of the dataset randomly.
    
    Args:
        data: List of all data points
        rho: Fraction of data to sample (0.0 to 1.0)
    
    Returns:
        Sampled subset of the data
    """
    if rho <= 0:
        return []
    if rho >= 1:
        return data.copy()
    
    num_samples = int(len(data) * rho)
    return random.sample(data, num_samples)

def create_sampled_datasets(rho_values, raw_dir, processed_dir, dataset_name="bad-medical-advice"):
    """
    Create sampled datasets for all specified rho values.
    
    Args:
        rho_values: List of sampling fractions (e.g., [0.01, 0.1, 0.5])
        raw_dir: Path to raw data directory
        processed_dir: Path to processed data directory
        dataset_name: Name of the dataset file (without .jsonl)
    """
    # Load the dataset
    input_path = (Path(raw_dir) / f"{dataset_name}.jsonl").resolve()
    full_data = load_jsonl(input_path)
    
    # Create processed directory if it doesn't exist
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    
    # Create sampled datasets for each rho value
    for rho in rho_values:
        sampled_data = sample_dataset(full_data, rho)
        output_path = Path(processed_dir) / f"{dataset_name}_rho_{rho:.2f}.jsonl"
        save_jsonl(sampled_data, output_path)
        print(f"Created sampled dataset (rho={rho:.2f}) at {output_path}")

if __name__ == "__main__":
    # Configuration
    RHO_VALUES = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # Sampling fractions
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    DATASET_NAME = "bad_medical_advice"  # Change if using a different dataset
    
    # Create the sampled datasets
    create_sampled_datasets(RHO_VALUES, RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_NAME)
    print("All sampled datasets created successfully!")