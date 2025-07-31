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

def extract_temperature_info(filename: str) -> str:
    """Extract temperature information from a filename."""
    temp_info = ""
    if "temp_" in filename:
        # Extract temperature value from name like "all_checkpoint_evaluations_temp_0_5"
        temp_parts = filename.split("temp_")
        if len(temp_parts) > 1:
            temp_value = temp_parts[1].replace("_", ".")
            temp_info = f" (temp={temp_value})"
    return temp_info

def group_similar_experiments(experiments: List[str], use_confidence_intervals: bool) -> Dict[str, List[str]]:
    """Group similar experiments for confidence interval calculation.
    
    Two experiments are considered similar if they only differ in the last integer after an underscore.
    """
    if not use_confidence_intervals:
        # Return each experiment as its own group when not using confidence intervals
        return {exp: [exp] for exp in experiments}
    
    experiment_groups = {}
    
    for exp in experiments:
        # Check if experiment ends with _INTEGER
        parts = exp.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            # The base experiment name without the last integer
            base_name = '_'.join(parts[:-1])
            if base_name not in experiment_groups:
                experiment_groups[base_name] = []
            experiment_groups[base_name].append(exp)
        else:
            # If the experiment doesn't match the pattern, keep it as a standalone
            experiment_groups[exp] = [exp]
    
    return experiment_groups

def plot_all_metrics(results_path: str, experiments: List[str], metrics: List[str], output_dir: str = None, 
                     sync_axis: bool = False, use_confidence_intervals: bool = False):
    """Create a single plot comparing all selected metrics across all experiments."""
    plt.figure(figsize=(14, 10))
    
    # Group similar experiments if using confidence intervals
    experiment_groups = group_similar_experiments(experiments, use_confidence_intervals)
    
    for group_name, group_exps in experiment_groups.items():
        # If we have multiple experiments in a group and using confidence intervals
        if len(group_exps) > 1 and use_confidence_intervals:
            for metric in metrics:
                # Collect all data files from experiments in this group, grouped by temperature
                temp_grouped_data = {}  # temperature -> list of dataframes
                
                for exp in group_exps:
                    exp_data = load_experiment_data(results_path, exp)
                    
                    # Group the data by temperature
                    for data_name, df in exp_data.items():
                        if metric in df.columns:
                            # Extract temperature info
                            temp_info = extract_temperature_info(data_name)
                            
                            # Keep only necessary columns
                            x_col = "checkpoint" if "checkpoint" in df.columns else df.index.name or "index"
                            if x_col == "index":
                                df = df.reset_index()
                            selected_cols = [x_col, metric]
                            
                            # Add to temperature group
                            if temp_info not in temp_grouped_data:
                                temp_grouped_data[temp_info] = []
                            
                            temp_grouped_data[temp_info].append(df[selected_cols])
                
                # Process each temperature group separately
                for temp_info, all_dfs in temp_grouped_data.items():
                    if all_dfs:
                        # Ensure all dataframes have the same number of rows
                        if all(len(df) == len(all_dfs[0]) for df in all_dfs):
                            # Stack the metric values from each experiment
                            stacked_values = np.stack([df[metric].values for df in all_dfs])
                            
                            # Calculate mean and std
                            mean_values = np.mean(stacked_values, axis=0)
                            std_values = np.std(stacked_values, axis=0)
                            
                            # Get x data from first dataframe
                            x_col = "checkpoint" if "checkpoint" in all_dfs[0].columns else "index"
                            x_data = all_dfs[0][x_col].values
                            
                            # Normalize x-axis if sync_axis is enabled
                            if sync_axis and len(x_data) > 1:
                                x_data_normalized = x_data / x_data.max()
                                plt.plot(x_data_normalized, mean_values, marker='o', label=f"{group_name}/{metric}{temp_info}")
                                plt.fill_between(
                                    x_data_normalized, 
                                    mean_values - std_values,
                                    mean_values + std_values,
                                    alpha=0.3
                                )
                            else:
                                plt.plot(x_data, mean_values, marker='o', label=f"{group_name}/{metric}{temp_info}")
                                plt.fill_between(
                                    x_data, 
                                    mean_values - std_values,
                                    mean_values + std_values,
                                    alpha=0.3
                                )
                        else:
                            print(f"Warning: Skipping confidence intervals for {group_name}/{metric}{temp_info} as not all experiments have the same number of rows")
        else:
            # Handle single experiments or when not using confidence intervals
            for exp in group_exps:
                exp_data = load_experiment_data(results_path, exp)
                
                # Group data by temperature
                for data_name, df in exp_data.items():
                    for metric in metrics:
                        if metric in df.columns:
                            # Extract temperature information
                            temp_info = extract_temperature_info(data_name)
                            
                            # Generate a unique label that includes experiment name, metric name, and temperature if available
                            label = f"{exp}/{metric}{temp_info}"
                            
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
    parser.add_argument("--confidence_intervals", action="store_true",
                        help="Merge similar experiments (ending with _INTEGER) and plot with confidence intervals")
    
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
                         # default=experiments
                         ),
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
    plot_all_metrics(args.results_path, selected_experiments, selected_metrics, args.output_dir, 
                    args.sync_axis, args.confidence_intervals)

if __name__ == "__main__":
    main()