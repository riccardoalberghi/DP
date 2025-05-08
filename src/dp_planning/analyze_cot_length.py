#!/usr/bin/env python3
import numpy as np
from utils import graph_generator, generate_CoT_A, evaluate_A

def main():
    print("Analyzing CoT lengths for 10000 graphs with different redundancy settings...")
    
    # Parameters
    num_graphs = 10000
    redundancy_values = [0, 2]
    results = {}
    
    for redundancy in redundancy_values:
        print(f"\nGenerating {num_graphs} graphs with redundancy={redundancy}...")
        cot_lengths = []
        
        for i in range(num_graphs):
            # Progress indicator every 1000 graphs
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1} graphs")
            
            # Generate a graph
            graph = graph_generator()
            
            # Generate CoT with specified redundancy
            cot, answer = generate_CoT_A(graph, redundancy=redundancy)
            
            # Use evaluate_A to get n_cot_steps
            evaluation = evaluate_A(graph, cot)
            cot_lengths.append(evaluation.n_CoT_steps)
        
        # Calculate statistics
        avg_length = np.mean(cot_lengths)
        std_length = np.std(cot_lengths)
        results[redundancy] = {
            "avg_length": avg_length,
            "std_length": std_length
        }
        
        print(f"Redundancy {redundancy} - Average CoT Length: {avg_length:.2f} ± {std_length:.2f} steps")
    
    # Print summary
    print("\nSummary of Results:")
    print(f"Redundancy 0 - Average CoT Length: {results[0]['avg_length']:.2f} ± {results[0]['std_length']:.2f} steps")
    print(f"Redundancy 2 - Average CoT Length: {results[2]['avg_length']:.2f} ± {results[2]['std_length']:.2f} steps")
    print(f"Ratio (Redundancy 2 / Redundancy 0): {results[2]['avg_length'] / results[0]['avg_length']:.2f}x")

if __name__ == "__main__":
    main()