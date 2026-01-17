"""
Module Checker Script
Scans zip files for model_metrics.json files and calculates scores
based on normalized MSE, MAE, and R2 values for the best 3 models.
"""

import os
import sys
import json
import zipfile

# ============== CONFIGURATION ==============
# Set this to your target directory path, or leave empty to use command line argument
DIRECTORY_PATH = ""

# Normalization bounds
MSE_MIN = 740
MSE_MAX = 6400
MAE_MIN = 20
MAE_MAX = 65
# R2 typically ranges from -1 to 1 (can be lower, but we'll clamp)
R2_MIN = -1
R2_MAX = 1

# Number of top models to consider for scoring
TOP_N_MODELS = 3
# ===========================================


def normalize_mse(mse: float) -> float:
    """Normalize MSE (lower is better, so invert)."""
    clamped = max(MSE_MIN, min(MSE_MAX, mse))
    return 1 - (clamped - MSE_MIN) / (MSE_MAX - MSE_MIN)


def normalize_mae(mae: float) -> float:
    """Normalize MAE (lower is better, so invert)."""
    clamped = max(MAE_MIN, min(MAE_MAX, mae))
    return 1 - (clamped - MAE_MIN) / (MAE_MAX - MAE_MIN)


def normalize_r2(r2: float) -> float:
    """Normalize R2 (higher is better)."""
    clamped = max(R2_MIN, min(R2_MAX, r2))
    return (clamped - R2_MIN) / (R2_MAX - R2_MIN)


def calculate_model_score(model_data: dict) -> float:
    """Calculate score for a single model based on normalized metrics."""
    mse = model_data.get("MSE", MSE_MAX)
    mae = model_data.get("MAE", MAE_MAX)
    r2 = model_data.get("R2", R2_MIN)
    
    return normalize_mse(mse) + normalize_mae(mae) + normalize_r2(r2)


def calculate_score_from_data(data: dict) -> float | None:
    """
    Calculate the total score based on model metrics data.
    Returns the sum of scores for the best N models.
    """
    try:
        models = data.get("models", {})
        if not models:
            return None
        
        # Calculate score for each model
        model_scores = []
        for model_name, model_data in models.items():
            score = calculate_model_score(model_data)
            model_scores.append((model_name, score))
        
        # Sort by score descending and take top N
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = model_scores[:TOP_N_MODELS]
        
        # Sum the scores of top models
        total_score = sum(score for _, score in top_models)
        return total_score
    
    except KeyError as e:
        print(f"Error processing data: {e}")
        return None


def calculate_zip_score(zip_path: str) -> float | None:
    """
    Calculate the total score for a zip file containing model_metrics.json.
    Returns the sum of scores for the best N models.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Look for model_metrics.json in the zip (could be at root or in a folder)
            metrics_file = None
            for name in zf.namelist():
                if name.endswith('model_metrics.json'):
                    metrics_file = name
                    break
            
            if metrics_file is None:
                return None
            
            with zf.open(metrics_file) as f:
                data = json.load(f)
                return calculate_score_from_data(data)
    
    except (zipfile.BadZipFile, json.JSONDecodeError, KeyError) as e:
        print(f"Error processing {zip_path}: {e}")
        return None


def scan_directory(root_dir: str) -> list[tuple[str, float]]:
    """
    Scan the root directory for zip files containing model_metrics.json.
    Returns a list of (name, score) tuples.
    """
    results = []
    
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        return results
    
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        
        # Check if it's a zip file
        if os.path.isfile(entry_path) and entry.lower().endswith('.zip'):
            name = os.path.splitext(entry)[0]  # Remove .zip extension
            score = calculate_zip_score(entry_path)
            if score is not None:
                results.append((name, score))
                print(f"Processed: {name} -> Score: {score:.4f}")
            else:
                print(f"Skipped: {name} (invalid or no model_metrics.json found)")
    
    return results
    
    return results


def write_results(results: list[tuple[str, float]], output_path: str) -> None:
    """Write the results to a text file, sorted by score in descending order."""
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for rank, (name, score) in enumerate(results, start=1):
            f.write(f"{rank}. {name} {score:.4f}\n")
    
    print(f"\nResults written to: {output_path}")


def main():
    # Determine the directory path
    if DIRECTORY_PATH:
        target_dir = DIRECTORY_PATH
    elif len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        print("Usage: python module_checker.py <directory_path>")
        print("Or set DIRECTORY_PATH constant in the script.")
        sys.exit(1)
    
    # Convert to absolute path
    target_dir = os.path.abspath(target_dir)
    print(f"Scanning directory: {target_dir}\n")
    
    # Scan and calculate scores
    results = scan_directory(target_dir)
    
    if not results:
        print("No valid model_metrics.json files found in zip files.")
        sys.exit(1)
    
    # Write results to file in the root directory
    output_path = os.path.join(target_dir, "results.txt")
    write_results(results, output_path)
    
    # Print summary
    print(f"\nSummary: Processed {len(results)} directories")
    print("\nTop results:")
    for name, score in sorted(results, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
