import os
import glob
import math
import hydra
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from hydra.utils import to_absolute_path
from tokenizer import get_tokenizer
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from datasets import load_from_disk
from utils import evaluate_A, from_Q_to_graph, create_eval_dataframe


@hydra.main(config_path="../../configs", config_name="config")
def main(cfg):
    # Find all checkpoint directories
    print(os.path.join(cfg.results_dir, cfg.experiment_name))
    checkpoint_paths = [os.path.join(cfg.results_dir, cfg.experiment_name, "checkpoint-7898")]# glob.glob(os.path.join(cfg.results_dir, cfg.experiment_name, "checkpoint-15796*"))
    
    # Create a dictionary mapping checkpoint steps to paths
    checkpoints = {int(os.path.basename(path).split("checkpoint-")[1]): path for path in checkpoint_paths}
    
    # Sort checkpoints by step
    checkpoints = dict(sorted(checkpoints.items()))
    
    print(f"Found {len(checkpoints)} checkpoints: {list(checkpoints.keys())}")
    
    # Get only the last checkpoint
    if checkpoints:
        last_step = max(checkpoints.keys())
        checkpoints = {last_step: checkpoints[last_step]}
        print(f"Using only the last checkpoint: {last_step}")
    else:
        print("No checkpoints found.")
        return
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(cfg)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load test dataset
    data_dir = to_absolute_path(cfg.dataset.data_dir)
    dataset_dict = load_from_disk(data_dir)
    test_dataset = dataset_dict['test']
    
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Initialize sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=cfg.training.temperature,  # Deterministic sampling
        max_tokens=8192,
        stop_token_ids=[tokenizer.encode("EoS")[0]],
        logprobs=2,
        prompt_logprobs=2,
    )
    
    # Loop through each checkpoint (now only the last one)
    for step, checkpoint_path in checkpoints.items():
        print(f"\nEvaluating checkpoint {step} from {checkpoint_path}")
        
        # Load model configuration
        config = AutoConfig.from_pretrained(checkpoint_path)
        
        # Initialize vLLM model
        model = LLM(
            model=checkpoint_path,
            dtype="float16" if torch.cuda.is_available() else "float32",
            trust_remote_code=True,
            gpu_memory_utilization=0.95
        )
        
        # Evaluate on test samples
        num_test_samples = min(10_000, len(test_dataset))
        
        # Prepare all prompts for this checkpoint
        all_indices = list(range(num_test_samples))
        all_questions = [test_dataset[i]['question'] for i in all_indices]
        
        # Create a list to store sample data
        samples_data = []
        
        print(f"Generating responses for {num_test_samples} test samples...")
        # Generate responses for all samples at once
        outputs = model.generate(all_questions, sampling_params)
        
        # Prepare ground truth prompts for batch inference
        all_ground_truth = [test_dataset[i]['question'] + " " + test_dataset[i]['cot'] for i in all_indices]
        
        print(f"Running batch inference on ground truth CoTs...")
        # Generate with all ground truths as input to get logprobs (batch operation)
        gt_outputs = model.generate(all_ground_truth, sampling_params)

        # Initialize tables of size (L-1)*(C-1)+1, C-1
        sum_table = np.zeros(shape=[(cfg.dataset.graph_generation.max_graph_depth-1)*(cfg.dataset.graph_generation.max_edge_cost-1)+1, cfg.dataset.graph_generation.max_edge_cost-1])
        correct_sum_table = np.zeros(shape=[(cfg.dataset.graph_generation.max_graph_depth-1)*(cfg.dataset.graph_generation.max_edge_cost-1)+1, cfg.dataset.graph_generation.max_edge_cost-1])
        
        # Process results
        for i, (output, gt_output) in tqdm(enumerate(zip(outputs, gt_outputs)), 
                                          total=len(outputs), 
                                          desc=f"Processing results for checkpoint {step}"):
            # Get the original sample
            sample = test_dataset[i]
            question = sample['question']
            ground_truth_cot = sample['cot']
            answer = sample['answer']
            
            # Get generated text
            predicted_cot = output.outputs[0].text
            
            # Get the logprobs from the ground truth analysis
            gt_logprobs = gt_output.prompt_logprobs
            
            # Get question tokens and ground truth tokens
            question_tokens = question.split()
            ground_truth_tokens = ground_truth_cot.split()
            
            # We need to find where the CoT starts in the combined sequence
            # The offset is the number of tokens in the question plus 1 for the space
            offset = len(question_tokens) #  + 1 check
            
            # Initialize a binary string to track cost prediction correctness
            cost_correctness = ""
            
            # Analyze each token in the ground truth
            for j in range(len(ground_truth_tokens) - 1):  # -1 because we're looking at next token
                current_token = ground_truth_tokens[j]
                next_token = ground_truth_tokens[j + 1]
                
                # Position in the full sequence (question + cot)
                pos_in_sequence = offset + j
                
                # If the next token is '|', then the current token is a cost
                if next_token == '|':
                    # Get the probability distribution for this position
                    if pos_in_sequence < len(gt_logprobs): # not necessary
                        logprob_dict = gt_logprobs[pos_in_sequence]
                        
                        # Convert logprobs to probs and find argmax
                        probs = {token: math.exp(logprob.logprob) for token, logprob in logprob_dict.items()}
                        predicted_token = max(probs.items(), key=lambda x: x[1])[0]
                        
                        # Check if prediction matches the ground truth
                        is_correct = tokenizer.decode([predicted_token]) == current_token
                        
                        # Update binary string: 1 for correct, 0 for incorrect
                        cost_correctness += "1" if is_correct else "0"
            
            _ = evaluate_A(from_Q_to_graph(question), ground_truth_cot + " " + answer, correct_costs=cost_correctness, sum_table=sum_table, correct_sum_table=correct_sum_table)
            
            # Store sample data
            sample_data = {
                'question': question,
                'ground_truth_cot': ground_truth_cot,
                'predicted_cot': predicted_cot,
                'cost_correctness': cost_correctness,
            }
            
            samples_data.append(sample_data)
        
        # Create rows for each sample (one row per sample)
        sample_rows = []
        for idx, sample in enumerate(samples_data):
            sample_rows.append({
                'sample_idx': idx,
                'question': sample['question'],
                'ground_truth_cot': sample['ground_truth_cot'],
                'predicted_cot': sample['predicted_cot'],
                'cost_correctness': sample['cost_correctness'],
            })
        
        # Convert to DataFrames and save as CSVs
        sample_df = pd.DataFrame(sample_rows)
        
        # Save sample-level data (with binary strings)
        sample_csv_path = os.path.join(cfg.results_dir, cfg.experiment_name, f"checkpoint_{step}_samples.csv")
        sample_df.to_csv(sample_csv_path, index=False)

        # Calculate elementwise division of correct_sum_table by sum_table
        import matplotlib.pyplot as plt
        
        # Avoid division by zero by adding a small epsilon where sum_table is 0
        epsilon = 1e-10
        sum_table_safe = np.copy(sum_table)
        sum_table_safe[sum_table_safe == 0] = epsilon
        
        # Calculate accuracy table (elementwise division)
        accuracy_table = correct_sum_table / sum_table_safe

        # ------------------------------------------------------------------
        # Save accuracy table to CSV with descriptive row and column labels
        # ------------------------------------------------------------------
        accuracy_df = pd.DataFrame(
            accuracy_table,
            index=[str(i) for i in range(accuracy_table.shape[0])],       # Row labels: 0, 1, 2, ...
            columns=[f"+{i+1}" for i in range(accuracy_table.shape[1])]    # Column labels: +1, +2, ...
        )
        accuracy_csv_path = os.path.join(
            cfg.results_dir,
            cfg.experiment_name,
            f"checkpoint_{step}_accuracy_table.csv"
        )
        accuracy_df.to_csv(accuracy_csv_path, index=True)
        print(f"Saved accuracy table to {accuracy_csv_path}")
        
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create the heatmap
        im = ax.imshow(accuracy_table, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        
        # Add a colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Accuracy', rotation=-90, va="bottom")
        
        # Set labels for x and y axes
        ax.set_xticks(np.arange(accuracy_table.shape[1]))
        ax.set_yticks(np.arange(accuracy_table.shape[0]))
        
        # Label with +1, +2, +3, ... for columns
        ax.set_xticklabels([f"+{i+1}" for i in range(accuracy_table.shape[1])])
        
        # Label with 0, 1, 2, 3, ... for rows
        ax.set_yticklabels([str(i) for i in range(accuracy_table.shape[0])])
        
        # Add grid lines for better readability
        ax.set_xticks(np.arange(accuracy_table.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(accuracy_table.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        
        # Add title and axis labels
        plt.title(f'Cost Prediction Accuracy - Checkpoint {step}')
        plt.xlabel('Addend')
        plt.ylabel('Answer Value')
        
        # Save the plot
        plot_path = os.path.join(cfg.results_dir, cfg.experiment_name, f"checkpoint_{step}_cost_accuracy.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved cost accuracy plot to {plot_path}")
        
        # Clean up resources
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
