import os
import glob
import hydra
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from hydra.utils import to_absolute_path
from transformers import AutoConfig
from vllm import LLM, SamplingParams

# Import from dp_planning package
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from dp_planning.tokenizer import get_tokenizer
from dp_planning.utils import layered_graph, evaluate_A, from_Q_to_graph, create_eval_dataframe


def generate_ood_graph(cfg, num_graphs=5):
    """
    Generate graphs that are in-distribution except for one layer that has 
    max_num_nodes_per_layer + 1 nodes.
    
    Args:
        cfg: Configuration object
        num_graphs: Number of graphs to generate
    
    Note: We ensure tokens are within vocabulary to avoid [UNK] tokens
    """
    ood_graphs = []
    
    # Get max values from config
    max_depth = cfg.dataset.graph_generation.max_graph_depth
    max_nodes = cfg.dataset.graph_generation.max_num_nodes_per_layer
    max_cost = cfg.dataset.graph_generation.max_edge_cost
    
    for i in range(num_graphs):
        # Generate graphs with random number of layers between 3-5
        ls = np.random.randint(3, 6)
        
        # Select a random layer (excluding first and last) to have max_nodes + 1 nodes
        ood_layer_idx = np.random.randint(1, ls)
        
        # Create node count for each layer
        ks = np.array([1])  # First layer always has 1 node (start)
        for l in range(ls-1):
            if l == ood_layer_idx - 1:  # -1 because we already added the first layer
                # This is our OOD layer with max_nodes + 1 nodes
                ks = np.append(ks, max_nodes + 1)
            else:
                # Use node counts within training distribution
                ks = np.append(ks, np.random.randint(2, max_nodes + 1))
        ks = np.append(ks, 1)  # Last layer always has 1 node (end)
        
        # Create adjacency matrices
        As = []
        for l in range(ls):
            A = np.zeros((ks[l], ks[l+1]), dtype=int)
            for j in range(ks[l]):
                # Use edge costs within training distribution
                A[j] = np.random.randint(1, max_cost + 1, size=ks[l+1]) * (np.random.rand(ks[l+1]) < 0.8)
                
                # Rule out dead ends
                if np.sum(A[j]) == 0:
                    A[j, np.random.randint(ks[l+1])] = np.random.randint(1, max_cost + 1)
            
            # Rule out disconnected nodes
            for k in range(ks[l+1]):
                if np.sum(A[:,k]) == 0:
                    A[np.random.randint(ks[l]),k] = np.random.randint(1, max_cost + 1)
            
            A[A == 0] = 100000  # inf
            As.append(A)
        
        # Create the graph
        graph = layered_graph(ls, ks, As)
        ood_graphs.append(graph)
        
    return ood_graphs


def validate_tokens(cfg, graph):
    """Check if the graph contains any tokens that would be [UNK] in the tokenizer"""
    tokenizer = get_tokenizer(cfg)
    graph_str = graph.Q_string()
    tokens = tokenizer.encode(graph_str)
    contains_unk = tokenizer.unk_token_id in tokens
    return not contains_unk


def generate_valid_ood_graphs(cfg, num_graphs=5):
    """
    Generate OOD graphs that don't contain [UNK] tokens
    
    Args:
        cfg: Configuration object
        num_graphs: Number of graphs to generate
    """
    valid_graphs = []
    attempts = 0
    max_attempts = num_graphs * 5  # Limit the number of attempts
    
    while len(valid_graphs) < num_graphs and attempts < max_attempts:
        attempts += 1
        graphs = generate_ood_graph(cfg, 1)
        graph = graphs[0]
        
        # Validate that the graph doesn't contain [UNK] tokens
        if validate_tokens(cfg, graph):
            valid_graphs.append(graph)
    
    print(f"Generated {len(valid_graphs)} valid OOD graphs after {attempts} attempts")
    return valid_graphs


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    """
    Main function to evaluate model on OOD graphs.
    
    Generates graphs that are in-distribution except for one layer that has 
    max_num_nodes_per_layer + 1 nodes.
    """
    # Find latest checkpoint
    checkpoint_paths = glob.glob(os.path.join(cfg.results_dir, cfg.experiment_name, "checkpoint-*"))
    if not checkpoint_paths:
        print("No checkpoints found")
        return
    
    # Sort checkpoints by step
    checkpoints = {int(os.path.basename(path).split("checkpoint-")[1]): path for path in checkpoint_paths}
    checkpoints = dict(sorted(checkpoints.items()))
    
    # Use the latest checkpoint
    latest_step = max(checkpoints.keys())
    checkpoint_path = checkpoints[latest_step]
    print(f"Using latest checkpoint at step {latest_step}: {checkpoint_path}")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(cfg)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Generating graphs with one layer having max_nodes + 1 nodes")
    
    # Generate OOD graphs
    ood_graphs = generate_valid_ood_graphs(cfg, num_graphs=1000)
    if not ood_graphs:
        print("Failed to generate valid OOD graphs")
        return
    
    # Create directory for evaluation results
    os.makedirs(os.path.join(cfg.results_dir, cfg.experiment_name, "ood_evaluation"), exist_ok=True)
    
    # Prepare graph questions
    questions = [graph.Q_string() for graph in ood_graphs]
    
    # Initialize sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic sampling
        max_tokens=8192,
        stop_token_ids=[tokenizer.encode("EoS")[0]]
    )
    
    print("Loading model...")
    try:
        # Initialize vLLM model
        model = LLM(
            model=checkpoint_path,
            dtype="float16" if torch.cuda.is_available() else "float32",
            trust_remote_code=True,
            gpu_memory_utilization=0.85
        )
        
        print("Generating responses...")
        # Generate responses for all OOD graphs
        outputs = model.generate(questions, sampling_params)
        
        # Create evaluation dataframe
        eval_df = create_eval_dataframe()
        
        print("Evaluating responses...")
        # Process and evaluate the results
        for i, output in enumerate(outputs):
            graph = ood_graphs[i]
            question = questions[i]
            generated_text = output.outputs[0].text
            
            # Extract predicted answer (everything after the question)
            predicted_text = generated_text[len(question):].strip()
            
            # Add BoS token to the predicted text if it's missing
            if not predicted_text.startswith("BoS"):
                predicted_text = "BoS " + predicted_text
            
            try:
                # Evaluate the predicted answer
                evaluation = evaluate_A(
                    graph, predicted_text,
                    BoS_tokens=True, BoT_tokens=True, 
                    aha_token=True, wait_token=True
                )
                
                # Add evaluation results to dataframe
                evaluation.add_row_df(eval_df)
                
                # Save the model's response
                with open(os.path.join(cfg.results_dir, cfg.experiment_name, "ood_evaluation", f"ood_response_{i+1}.txt"), "w") as f:
                    f.write(f"Question:\n{question}\n\nResponse:\n{predicted_text}")
                
            except Exception as e:
                print(f"Error evaluating OOD graph {i+1}: {e}")
                # Add a row with NaN values if evaluation fails
                eval_df.loc[eval_df.index.max() + 1 if not eval_df.empty else 0] = [np.nan] * len(eval_df.columns)
        
        # Save overall evaluation results
        filename = "evaluation_results_ood_max_nodes_plus_one.csv"
        eval_df.to_csv(os.path.join(cfg.results_dir, cfg.experiment_name, "ood_evaluation", filename))
        
        # Calculate average for each metric in the dataframe
        # Include all numeric columns for a complete evaluation
        all_metrics = eval_df.columns.tolist()
        avg_metrics = {metric: eval_df[metric].astype(float).mean() for metric in all_metrics}
        
        # Print average metrics
        print(f"\nAverage Metrics for OOD Graphs with max_nodes + 1 (Checkpoint {latest_step}):")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nSaved OOD evaluation results to {os.path.join(cfg.results_dir, cfg.experiment_name, 'ood_evaluation')}")
        
    except Exception as e:
        print(f"Error running model inference: {e}")
    finally:
        # Cleanup resources
        if 'model' in locals():
            from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            import ray
            import gc
            import contextlib
            
            destroy_model_parallel()
            destroy_distributed_environment()
            del model
            with contextlib.suppress(AssertionError):
                torch.distributed.destroy_process_group()
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()
            
            # Add a small delay to ensure cleanup is complete
            import time
            time.sleep(2)
            print("Cleared model from GPU memory")


if __name__ == "__main__":
    main()