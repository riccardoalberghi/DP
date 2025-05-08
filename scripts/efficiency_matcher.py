#!/usr/bin/env python3
import os
import sys
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the parent directory to the path to import the utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dp_planning.utils import graph_generator, generate_CoT_A

def match_efficiency_token_lengths(cfg, target_efficiency_high, target_efficiency_low, n_graphs=1000, p_redundancy_range=(0, 0.5), n_steps=10):
    """
    Finds the p_redundancy value that matches the token lengths of CoTs generated with two different efficiency values.
    
    Args:
        cfg: Hydra configuration
        target_efficiency_high: Higher efficiency value (typically more efficient, shorter CoTs)
        target_efficiency_low: Lower efficiency value (typically less efficient, longer CoTs)
        n_graphs: Number of graphs to generate for testing each p_redundancy value
        p_redundancy_range: Tuple with min and max p_redundancy values to sweep
        n_steps: Number of steps in the p_redundancy sweep
        
    Returns:
        Dictionary with results of the search
    """
    # Extract graph generation parameters from config
    L = cfg.dataset.graph_generation.max_graph_depth
    K = cfg.dataset.graph_generation.max_num_nodes_per_layer
    C = cfg.dataset.graph_generation.max_edge_cost
    p = cfg.dataset.graph_generation.connectivity_probability
    
    # Get max token length from config
    max_length = cfg.training.max_length
    
    print(f"Graph parameters: L={L}, K={K}, C={C}, p={p}")
    print(f"Target efficiency values: high={target_efficiency_high}, low={target_efficiency_low}")
    print(f"Testing with {n_graphs} graphs for each p_redundancy value")
    print(f"p_redundancy range: {p_redundancy_range[0]} to {p_redundancy_range[1]} in {n_steps} steps")
    
    # Import tokenizer to check sequence lengths
    from src.dp_planning.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(cfg)
    
    # Generate a fixed set of random graphs to use for all trials
    print("Generating random graphs...")
    graphs = [graph_generator(L=L, K=K, C=C, p=p) for _ in tqdm(range(n_graphs))]
    
    # Compute token lengths for the high efficiency value (with no redundancy)
    print(f"\nAnalyzing high efficiency = {target_efficiency_high} (with p_redundancy=0)")
    high_efficiency_token_lengths = []
    
    for graph in tqdm(graphs):
        q = graph.Q_string()
        cot, a = generate_CoT_A(graph, efficiency=target_efficiency_high, p_redundancy=0)
        
        combined_text = q + " " + cot + " " + a
        tokenized = tokenizer(combined_text)
        token_length = len(tokenized['input_ids'])
        
        # Skip samples that exceed max token length
        if token_length > max_length:
            continue
        
        high_efficiency_token_lengths.append(token_length)
    
    # Calculate average token length for high efficiency
    avg_high_efficiency_length = np.mean(high_efficiency_token_lengths)
    print(f"Average token length for efficiency={target_efficiency_high}: {avg_high_efficiency_length:.2f}")
    
    # Test different p_redundancy values for the low efficiency setting
    p_redundancy_values = np.linspace(p_redundancy_range[0], p_redundancy_range[1], n_steps)
    results = {}
    
    for p_redundancy in p_redundancy_values:
        print(f"\nTesting p_redundancy = {p_redundancy:.3f} with efficiency = {target_efficiency_low}")
        token_lengths = []
        
        for graph in tqdm(graphs):
            q = graph.Q_string()
            cot, a = generate_CoT_A(graph, efficiency=target_efficiency_low, p_redundancy=p_redundancy)
            
            combined_text = q + " " + cot + " " + a
            tokenized = tokenizer(combined_text)
            token_length = len(tokenized['input_ids'])
            
            # Skip samples that exceed max token length
            if token_length > max_length:
                continue
            
            token_lengths.append(token_length)
        
        avg_length = np.mean(token_lengths)
        length_diff = avg_length - avg_high_efficiency_length
        
        results[p_redundancy] = {
            'avg_token_length': avg_length,
            'length_difference': length_diff
        }
        
        print(f"Average token length: {avg_length:.2f}")
        print(f"Difference from high efficiency: {length_diff:.2f}")
    
    # Find the p_redundancy value that minimizes the absolute difference in token lengths
    best_p_redundancy = min(results.keys(), key=lambda p: abs(results[p]['length_difference']))
    
    print("\n===== Results =====")
    print(f"High efficiency ({target_efficiency_high}) average token length: {avg_high_efficiency_length:.2f}")
    print(f"Best p_redundancy for low efficiency ({target_efficiency_low}): {best_p_redundancy:.3f}")
    print(f"Low efficiency average token length with best p_redundancy: {results[best_p_redundancy]['avg_token_length']:.2f}")
    print(f"Length difference: {results[best_p_redundancy]['length_difference']:.2f}")
    
    return {
        'high_efficiency': target_efficiency_high,
        'low_efficiency': target_efficiency_low,
        'high_efficiency_token_length': avg_high_efficiency_length,
        'p_redundancy_values': dict(results),
        'best_p_redundancy': best_p_redundancy,
        'matched_token_length': results[best_p_redundancy]['avg_token_length'],
        'length_difference': results[best_p_redundancy]['length_difference']
    }

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Default parameters
    high_efficiency = -5.0
    low_efficiency = 5.0
    n_graphs = 20000
    p_redundancy_range = (0.0, 0.7)
    n_steps = 17  # 0.0, 0.05, 0.1, ..., 0.5
    
    # Run the matching algorithm
    results = match_efficiency_token_lengths(
        cfg=cfg,
        target_efficiency_high=high_efficiency,
        target_efficiency_low=low_efficiency,
        n_graphs=n_graphs,
        p_redundancy_range=p_redundancy_range,
        n_steps=n_steps
    )
    
    # Print formatted results
    print("\nEfficiency Matcher Results:")
    print("=========================")
    print(f"Best p_redundancy: {results['best_p_redundancy']:.3f}")
    print(f"High Efficiency ({results['high_efficiency']}) Token Length: {results['high_efficiency_token_length']:.2f}")
    print(f"Low Efficiency ({results['low_efficiency']}) Token Length with p_redundancy={results['best_p_redundancy']:.3f}: {results['matched_token_length']:.2f}")
    print(f"Length Difference: {results['length_difference']:.2f}")

if __name__ == "__main__":
    main()