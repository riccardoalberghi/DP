#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import inquirer
from inquirer.themes import GreenPassion

def find_experiments(results_path: str) -> List[str]:
    """Find all experiment directories in the given path."""
    if not os.path.exists(results_path):
        print(f"Error: The path {results_path} does not exist.")
        return []
    
    # List all directories in the results path
    experiments = [d for d in os.listdir(results_path) 
                   if os.path.isdir(os.path.join(results_path, d))]
    
    # Filter to only include directories that have csv files
    valid_experiments = []
    for exp in experiments:
        csv_files = glob.glob(os.path.join(results_path, exp, "*.csv"))
        if csv_files:
            valid_experiments.append(exp)
    
    return valid_experiments

def get_metrics_from_experiments(results_path: str, experiments: List[str]) -> List[str]:
    """Extract all available metrics from the given experiments."""
    all_metrics = set()
    
    for exp in experiments:
        csv_files = glob.glob(os.path.join(results_path, exp, "*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_metrics.update(df.columns)
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
    
    # Remove common non-metric columns
    non_metrics = {'step', 'epoch', 'timestamp', 'iteration', 'time'}
    metrics = [m for m in all_metrics if m.lower() not in non_metrics]
    
    return sorted(metrics)

def load_experiment_data(results_path: str, experiment: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV data from an experiment directory."""
    csv_files = glob.glob(os.path.join(results_path, experiment, "*.csv"))
    data_dict = {}
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Use the filename without extension as the key
            key = os.path.splitext(os.path.basename(csv_file))[0]
            data_dict[key] = df
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
    
    return data_dict

def plot_all_metrics(results_path: str, experiments: List[str], metrics: List[str], output_dir: str = None, sync_axis: bool = False):
    """Create a single plot comparing all selected metrics across all experiments."""
    plt.figure(figsize=(14, 10))
    
    for exp in experiments:
        exp_data = load_experiment_data(results_path, exp)
        
        for data_name, df in exp_data.items():
            for metric in metrics:
                if metric in df.columns:
                    # Generate a unique label that includes experiment name and metric name
                    label = f"{exp}/{metric}"
                    
                    # Use checkpoint column if available, otherwise use the index
                    x_data = df["checkpoint"] if "checkpoint" in df.columns else df.index
                    
                    # Normalize x-axis if sync_axis is enabled
                    if sync_axis and len(x_data) > 1:
                        # Scale checkpoints from 0 to 1
                        x_data_normalized = x_data / x_data.max()
                        plt.plot(x_data_normalized, df[metric], marker='o', label=label)
                    else:
                        # Plot the data with a marker and the formatted label
                        plt.plot(x_data, df[metric], marker='o', label=label)
    
    plt.title("Comparison of metrics across experiments")
    
    # Update x-axis label based on sync_axis setting
    if sync_axis:
        plt.xlabel("Normalized training progress (0-1)")
    else:
        plt.xlabel("Training steps")
        
    plt.ylabel("Metric values")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add tight layout to prevent clipping of labels
    plt.tight_layout()
    
    # Save the plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "all_metrics_comparison.png"), dpi=300)
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare metrics across experiments")
    parser.add_argument("--results_path", type=str, required=True,
                        help="Path to the directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory where to save the plots (optional)")
    parser.add_argument("--sync_axis", action="store_true",
                        help="Normalize the x-axis (checkpoints) to scale from 0 to 1 for all experiments")
    
    args = parser.parse_args()
    
    # Find all valid experiments
    experiments = find_experiments(args.results_path)
    
    if not experiments:
        print("No valid experiments found! Ensure your results path contains experiment directories with CSV files.")
        return
    
    # Ask user to select experiments
    questions = [
        inquirer.Checkbox('selected_experiments',
                         message="Select experiments to compare",
                         choices=experiments,
                         default=experiments),
    ]
    
    answers = inquirer.prompt(questions, theme=GreenPassion())
    
    if not answers or not answers['selected_experiments']:
        print("No experiments selected. Exiting.")
        return
    
    selected_experiments = answers['selected_experiments']
    
    # Get all available metrics from the selected experiments
    all_metrics = get_metrics_from_experiments(args.results_path, selected_experiments)
    
    if not all_metrics:
        print("No metrics found in the selected experiments!")
        return
    
    # Ask user to select metrics
    questions = [
        inquirer.Checkbox('selected_metrics',
                         message="Select metrics to plot",
                         choices=all_metrics),
    ]
    
    answers = inquirer.prompt(questions, theme=GreenPassion())
    
    if not answers or not answers['selected_metrics']:
        print("No metrics selected. Exiting.")
        return
    
    selected_metrics = answers['selected_metrics']
    
    # Create a single plot with all selected metrics and experiments
    plot_all_metrics(args.results_path, selected_experiments, selected_metrics, args.output_dir, args.sync_axis)

if __name__ == "__main__":
    main()