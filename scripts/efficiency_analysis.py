#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Add the parent directory to the path to import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dp_planning.utils import graph_generator, generate_CoT_A, evaluate_A

def analyze_efficiency_impact(cfg, efficiency_values=None, n_graphs=10000):
    """
    Analyze the impact of efficiency parameter on CoT length and wait tokens
    
    Args:
        cfg: Hydra configuration
        efficiency_values: List of efficiency values to test
        n_graphs: Number of graphs to generate for each efficiency value
    
    Returns:
        Dictionary mapping efficiency values to average CoT steps and wait tokens
    """
    if efficiency_values is None:
        efficiency_values = [-5, -10, -20] # [-5, 0, 5]
    
    # Extract graph generation parameters from config
    L = cfg.dataset.graph_generation.max_graph_depth
    K = cfg.dataset.graph_generation.max_num_nodes_per_layer
    C = cfg.dataset.graph_generation.max_edge_cost
    p = cfg.dataset.graph_generation.connectivity_probability
    
    # Get max token length from config
    max_length = cfg.training.max_length
    
    results = {}
    
    print(f"Graph parameters: L={L}, K={K}, C={C}, p={p}")
    print(f"Analyzing {len(efficiency_values)} efficiency values with {n_graphs} graphs each")
    print(f"Max token length: {max_length}")
    
    # Import tokenizer to check sequence lengths
    from src.dp_planning.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg)
    
    for eff in efficiency_values:
        print(f"\nAnalyzing efficiency = {eff}")
        cot_steps = []
        wait_tokens = []
        skipped_samples = 0
        
        for i in tqdm(range(n_graphs)):
            # Generate a random graph
            graph = graph_generator(L=L, K=K, C=C, p=p)
            
            # Generate CoT with the current efficiency
            q = graph.Q_string()
            cot, a = generate_CoT_A(graph, efficiency=eff, redundancy=2)
            
            # Check if the tokenized sequence would exceed max_length
            combined_text = q + " " + cot + " " + a
            tokenized = tokenizer(combined_text)
            token_length = len(tokenized['input_ids'])
            
            # Skip samples that exceed max token length
            if token_length > max_length:
                skipped_samples += 1
                continue
            
            # Evaluate to get the number of CoT steps
            ev = evaluate_A(graph, cot + " " + a)
            
            # Store the number of CoT steps
            cot_steps.append(ev.n_CoT_steps)
            
            # Count wait tokens in the CoT string
            wait_count = cot.split().count("wait")
            wait_tokens.append(wait_count)
        
        # Calculate averages only if we have samples
        if cot_steps:
            avg_steps = np.mean(cot_steps)
            std_steps = np.std(cot_steps)
            avg_wait = np.mean(wait_tokens)
            std_wait = np.std(wait_tokens)
            results[eff] = (avg_steps, std_steps, avg_wait, std_wait)
            
            print(f"Efficiency {eff}: Avg steps = {avg_steps:.2f} ± {std_steps:.2f}, Avg wait tokens = {avg_wait:.2f} ± {std_wait:.2f}")
            print(f"Skipped {skipped_samples} samples that exceeded max token length of {max_length}")
        else:
            print(f"Efficiency {eff}: All {n_graphs} samples exceeded max token length of {max_length}")
            results[eff] = (0, 0, 0, 0)
    
    return results

def plot_results(results, output_dir):
    """Plot the relationship between efficiency, CoT length, and wait tokens"""
    efficiencies = sorted(results.keys())
    avg_steps = [results[e][0] for e in efficiencies]
    std_steps = [results[e][1] for e in efficiencies]
    avg_wait = [results[e][2] for e in efficiencies]
    std_wait = [results[e][3] for e in efficiencies]
    
    # Plot for CoT steps
    plt.figure(figsize=(10, 6))
    plt.errorbar(efficiencies, avg_steps, yerr=std_steps, fmt='o-', capsize=5, label='CoT Steps')
    plt.xlabel('Efficiency Parameter')
    plt.ylabel('Average Number of CoT Steps')
    plt.title('Relationship Between Efficiency Parameter and CoT Length')
    plt.grid(True)
    
    # Add log scale for x-axis if we have negative values
    if min(efficiencies) < 0:
        plt.xscale('symlog')
    
    output_path = os.path.join(output_dir, 'efficiency_vs_cot_length.png')
    plt.savefig(output_path)
    plt.close()
    
    # Plot for wait tokens
    plt.figure(figsize=(10, 6))
    plt.errorbar(efficiencies, avg_wait, yerr=std_wait, fmt='o-', capsize=5, color='red', label='Wait Tokens')
    plt.xlabel('Efficiency Parameter')
    plt.ylabel('Average Number of Wait Tokens')
    plt.title('Relationship Between Efficiency Parameter and Wait Tokens')
    plt.grid(True)
    
    # Add log scale for x-axis if we have negative values
    if min(efficiencies) < 0:
        plt.xscale('symlog')
    
    output_path_wait = os.path.join(output_dir, 'efficiency_vs_wait_tokens.png')
    plt.savefig(output_path_wait)
    plt.close()
    
    # Combined plot
    plt.figure(figsize=(12, 6))
    plt.errorbar(efficiencies, avg_steps, yerr=std_steps, fmt='o-', capsize=5, label='CoT Steps')
    plt.errorbar(efficiencies, avg_wait, yerr=std_wait, fmt='s-', capsize=5, color='red', label='Wait Tokens')
    plt.xlabel('Efficiency Parameter')
    plt.ylabel('Average Count')
    plt.title('Efficiency Impact on CoT Steps and Wait Tokens')
    plt.grid(True)
    plt.legend()
    
    # Add log scale for x-axis if we have negative values
    if min(efficiencies) < 0:
        plt.xscale('symlog')
    
    output_path_combined = os.path.join(output_dir, 'efficiency_vs_combined.png')
    plt.savefig(output_path_combined)
    plt.close()
    
    print(f"\nPlots saved as '{output_path}', '{output_path_wait}', and '{output_path_combined}'")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Set up output directory
    output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # Default parameters
    efficiency_values = None
    n_graphs = 100_000
    
    # Run the analysis
    results = analyze_efficiency_impact(
        cfg=cfg,
        efficiency_values=efficiency_values,
        n_graphs=n_graphs
    )
    
    # Save results to file
    results_path = os.path.join(output_dir, 'efficiency_results.txt')
    with open(results_path, 'w') as f:
        f.write("Summary of Results:\n")
        f.write("===================\n")
        f.write("Efficiency | Avg CoT Steps | Std Dev | Avg Wait Tokens | Std Dev\n")
        f.write("-" * 65 + "\n")
        for eff in sorted(results.keys()):
            avg_steps, std_steps, avg_wait, std_wait = results[eff]
            f.write(f"{eff:10.2f} | {avg_steps:12.2f} | {std_steps:7.2f} | {avg_wait:14.2f} | {std_wait:7.2f}\n")
    
    # Print formatted results
    print("\nSummary of Results:")
    print("===================")
    print("Efficiency | Avg CoT Steps | Std Dev | Avg Wait Tokens | Std Dev")
    print("-" * 65)
    for eff in sorted(results.keys()):
        avg_steps, std_steps, avg_wait, std_wait = results[eff]
        print(f"{eff:10.2f} | {avg_steps:12.2f} | {std_steps:7.2f} | {avg_wait:14.2f} | {std_wait:7.2f}")
    
    # Generate plot
    plot_results(results, output_dir)

if __name__ == "__main__":
    main()