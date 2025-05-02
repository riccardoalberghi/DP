import hydra
import os
import torch
import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path
from tokenizer import get_tokenizer
from transformers import Phi3Config, Phi3ForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_from_disk
from utils import evaluate_A, from_Q_to_graph, create_eval_dataframe


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    #### Init tokenizer
    tokenizer = get_tokenizer(cfg)
    # Set pad token to EoS token for consistent handling
    tokenizer.pad_token = "[PAD]"

    #### Init model
    config = AutoConfig.for_model(
        cfg.model.model_name,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.encode("BoS")[0],
        eos_token_id=tokenizer.encode("EoS")[0],
        pad_token_id=tokenizer.encode("[PAD]")[0],
        attn_implementation="sdpa",
        **cfg.model.parameters,
    )

    model = AutoModelForCausalLM.from_config(config)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    #### Load dataset
    data_dir = to_absolute_path(cfg.dataset.data_dir)
    dataset_dict = load_from_disk(data_dir)

    if cfg.training.limit_num_of_samples is not None:
        dataset_dict["train"] = dataset_dict["train"].select(range(cfg.training.limit_num_of_samples))
    
    print(f"Loaded dataset with {len(dataset_dict['train'])} train and {len(dataset_dict['test'])} test samples")
    
    def tokenize_function(examples):
        texts = []
        if cfg.training.train_on_cots:
            texts = [f"{q} {c} {a}" for q, c, a in zip(examples["question"], examples["cot"], examples["answer"])]
        else:
            texts = [f"{q} {a}" for q, a in zip(examples["question"], examples["answer"])]
        
        # Pre-allocate the tokenizer output dictionary
        batch_size = len(texts)
        max_length = cfg.training.max_length
        
        # Tokenize
        encodings = tokenizer(texts, 
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
        pad_id = tokenizer.pad_token_id

        question_encodings = tokenizer(examples["question"].copy(), add_special_tokens=False)
        
        # Get a view of the data as a NumPy array (no copy)
        input_ids = np.asarray(encodings["input_ids"])
        
        # Pre-allocate output array with -100s
        labels = np.full_like(input_ids, -100)
        
        # Set non-pad positions directly (fastest approach)
        # mask = input_ids != pad_id
        # labels[mask] = input_ids[mask]
        for i in range(batch_size):
            # Get question length for this example (add 1 for the initial special token)
            q_length = len(question_encodings["input_ids"][i]) + 1
            
            # Set only non-pad and non-question positions to their corresponding input_ids
            # This combines two conditions:
            # 1. Token is not a pad token
            # 2. Token position is beyond the question part
            mask = (input_ids[i] != pad_id) & (np.arange(len(input_ids[i])) >= q_length)
            labels[i, mask] = input_ids[i, mask]
        
        encodings["labels"] = labels.tolist()
        return encodings
    
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=["question", "cot", "answer"],
        batch_size=1000,
    )

    def count_non_pad_tokens(examples):
        # Get direct array view when possible
        input_ids = np.asarray(examples["input_ids"])
        pad_id = tokenizer.pad_token_id
        
        # Single vectorized operation across the whole batch
        counts = (input_ids != pad_id).sum(axis=1)
        
        return {"non_pad_count": counts.tolist()}

    # Process in very large batches with parallel execution
    token_counts = tokenized_datasets["train"].map(
        count_non_pad_tokens,
        batched=True,
        batch_size=1000,
    )

    # Calculate average in one step using NumPy
    average_tokens_per_sample = int(np.mean(np.array(token_counts["non_pad_count"])))
    
    # Data collator for language modeling (handles masking for MLM and full sequence for CLM)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal language modeling, not masked
        return_tensors="pt"
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(cfg.results_dir, cfg.experiment_name),
        overwrite_output_dir=True,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.tokens_per_step // average_tokens_per_sample,
        per_device_eval_batch_size=cfg.training.tokens_per_step // average_tokens_per_sample,
        eval_strategy="steps",
        save_strategy="steps",
        # Save every 10% of total steps
        save_steps=int(cfg.training.num_epochs * len(tokenized_datasets["train"]) / 
                      (cfg.training.tokens_per_step // average_tokens_per_sample) / 40),
        eval_steps=int(cfg.training.num_epochs * len(tokenized_datasets["train"]) / 
                      (cfg.training.tokens_per_step // average_tokens_per_sample) / 40),
        logging_dir=os.path.join(cfg.results_dir, cfg.experiment_name, "logs"),
        logging_steps=cfg.training.logging_steps,
        lr_scheduler_type="constant" if cfg.training.constant_lr else "linear",
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        fp16=True,
        report_to="wandb",
    )

    # Initialize trainer with custom scheduler
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {os.path.join(cfg.results_dir, cfg.experiment_name, 'final_model')}")
    trainer.save_model(os.path.join(cfg.results_dir, cfg.experiment_name, "final_model"))
    tokenizer.save_pretrained(os.path.join(cfg.results_dir, cfg.experiment_name, "final_model"))
    
    # Evaluate model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")


if __name__ == "__main__":
    main()