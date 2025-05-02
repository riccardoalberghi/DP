import os
import glob
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
    checkpoint_paths = glob.glob(os.path.join(cfg.results_dir, cfg.experiment_name, "checkpoint-*"))
    
    # Create a dictionary mapping checkpoint steps to paths
    checkpoints = {int(os.path.basename(path).split("checkpoint-")[1]): path for path in checkpoint_paths}
    
    # Sort checkpoints by step
    checkpoints = dict(sorted(checkpoints.items()))
    
    print(f"Found {len(checkpoints)} checkpoints: {list(checkpoints.keys())}")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer(cfg)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load test dataset
    data_dir = to_absolute_path(cfg.dataset.data_dir)
    dataset_dict = load_from_disk(data_dir)
    test_dataset = dataset_dict['test']
    
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Create dataframe to store evaluation results
    results_df = pd.DataFrame(columns=['checkpoint', 'syntax_errors', 'is_A_path_possible', 
                                     'is_A_cost_consistent', 'is_A_cost_optimal', 
                                     'is_A_path_length_correct'])
    
    # Initialize sampling parameters for vLLM
    sampling_params = SamplingParams(
        temperature=0,  # Deterministic sampling (equivalent to do_sample=False)
        max_tokens=8192,
        stop_token_ids=[tokenizer.encode("EoS")[0]]
    )
    
    # Loop through each checkpoint
    for step, checkpoint_path in checkpoints.items():
        print(f"\nEvaluating checkpoint {step} from {checkpoint_path}")
        
        # Load model configuration
        config = AutoConfig.from_pretrained(checkpoint_path)
        
        # Initialize vLLM model
        model = LLM(
            model=checkpoint_path,
            dtype="float16" if torch.cuda.is_available() else "float32",
            trust_remote_code=True,
            gpu_memory_utilization=0.85  # Increase memory utilization
        )
        
        # Create evaluation dataframe for this checkpoint
        eval_df = create_eval_dataframe()
        
        # Evaluate on test samples (limit to a smaller number for faster evaluation)
        num_test_samples = min(10_000, len(test_dataset))
        
        # Prepare all prompts for this checkpoint
        all_indices = list(range(num_test_samples))
        all_questions = [test_dataset[i]['question'] for i in all_indices]
        
        print(f"Generating responses for {num_test_samples} test samples...")
        # Generate responses for all samples at once (vLLM handles batching internally)
        outputs = model.generate(all_questions, sampling_params)
        
        # Process results
        for i, output in tqdm(enumerate(outputs), total=len(outputs), desc=f"Processing results for checkpoint {step}"):
            # Get the original sample
            sample = test_dataset[i]
            question = sample['question']
            ground_truth_cot = sample['cot']
            ground_truth_answer = sample['answer']
            
            # Get generated text
            generated_text = output.outputs[0].text
            
            # Extract predicted answer (everything after the question)
            predicted_text = generated_text
            
            # Parse the graph from the question to use for evaluation
            try:
                graph = from_Q_to_graph(question)
                
                # Add BoS token to the predicted text if it's missing
                if not predicted_text.startswith("BoS"):
                    predicted_text = "BoS " + predicted_text
                
                # Evaluate the predicted answer
                evaluation = evaluate_A(graph, predicted_text, 
                                       BoS_tokens=True, BoT_tokens=True, 
                                       aha_token=True, wait_token=True)
                
                # Add evaluation results to dataframe
                evaluation.add_row_df(eval_df)
            except Exception as e:
                print(f"Error evaluating sample {i}: {e}")
                # Add a row with NaN values if evaluation fails
                eval_df.loc[eval_df.index.max() + 1 if not eval_df.empty else 0] = [np.nan] * len(eval_df.columns)
        
        # Compute average metrics for this checkpoint
        avg_metrics = {
            'checkpoint': step
        }
        
        # Calculate overall metrics for all columns except num_layers
        for col in eval_df.columns:
            if col != 'num_layers':
                # Only include True/False values in the mean calculation, not NaN values
                avg_metrics[col] = eval_df[col].astype(float).mean()
        
        # Get unique layer counts
        unique_layers = eval_df['num_layers'].dropna().unique()
        
        # For each layer count, add layer-specific metrics for all columns
        for num_l in unique_layers:
            # Filter the dataframe for this layer count
            layer_df = eval_df[eval_df['num_layers'] == num_l]
            
            # Add all metrics for this layer count
            for col in eval_df.columns:
                # Skip num_layers column itself
                if col != 'num_layers':
                    # Create a new metric name with layer count suffix
                    layer_metric_name = f"{col}_l{int(num_l)}"
                    # Compute and store the average for this layer count
                    avg_metrics[layer_metric_name] = layer_df[col].astype(float).mean()
        
        # Add results to the main dataframe
        results_df = pd.concat([results_df, pd.DataFrame([avg_metrics])], ignore_index=True)
        
        # Save checkpoint-specific results
        eval_df.to_csv(os.path.join(checkpoint_path, "evaluation_results.csv"))
        
        # Print summary of results for this checkpoint
        print(f"Checkpoint {step} evaluation results:")
        for metric, value in avg_metrics.items():
            if metric != 'checkpoint':
                print(f"  {metric}: {value:.4f}")
        
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
    
    # Save overall results
    results_path = os.path.join(cfg.results_dir, cfg.experiment_name, "all_checkpoint_evaluations.csv")
    results_df.to_csv(results_path)
    print(f"\nSaved overall evaluation results to {results_path}")
    
    # Plot performance over checkpoints
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        key_metrics = ['is_A_path_possible', 'is_A_cost_consistent', 'is_A_cost_optimal', 'is_A_path_length_correct', 'is_A_path_correct']
        for metric in key_metrics:
            plt.plot(results_df['checkpoint'], results_df[metric], marker='o', label=metric)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics Across Checkpoints')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(cfg.results_dir, cfg.experiment_name, "evaluation_metrics_plot.png")
        plt.savefig(plot_path)
        print(f"Saved evaluation metrics plot to {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
