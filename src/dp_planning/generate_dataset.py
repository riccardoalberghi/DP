import os
import tqdm
import hydra
from tokenizer import get_tokenizer
import pandas as pd
from hydra.utils import to_absolute_path
from datasets import Dataset, DatasetDict
from utils import generate_CoT_A, graph_generator


def generate_samples(cfg):
    samples = []
    questions_set = set()
    tot_tokens = 0

    tokenizer = get_tokenizer(cfg)

    pbar = tqdm.tqdm(total=cfg.training.tokens_per_epoch, desc="Generating samples...", unit="tokens")

    while tot_tokens < cfg.training.tokens_per_epoch:
        graph = graph_generator(L=cfg.dataset.graph_generation.max_graph_depth, 
                                K=cfg.dataset.graph_generation.max_num_nodes_per_layer,
                                C=cfg.dataset.graph_generation.max_edge_cost,
                                p=cfg.dataset.graph_generation.connectivity_probability,
                                )
        
        question = graph.Q_string()
        cot, answer = generate_CoT_A(graph, efficiency=cfg.dataset.cot_generation.efficiency, aha_token=cfg.training.use_aha, wait_token=cfg.training.use_wait)
        
        # Set CoT to empty string if train_on_cots is False
        if not cfg.training.train_on_cots:
            cot = ""

        question_tok = tokenizer.encode(question)
        cot_tok = tokenizer.encode(cot)
        answer_tok = tokenizer.encode(answer)
        
        length = len(question_tok) + len(cot_tok) + len(answer_tok)

        if length > cfg.training.max_length:
            continue

        if question in questions_set:
            continue

        if tokenizer.unk_token in question_tok:
            print("Unkonw token")
            print(question)
            print(tokenizer.decode(question_tok))
        
        if tokenizer.unk_token in cot_tok:
            print("Unkonw token")
            print(cot)
            print(tokenizer.decode(cot_tok))
        
        if tokenizer.unk_token in answer_tok:
            print("Unkonw token")
            print(answer_tok)
            print(tokenizer.decode(answer_tok))

        samples.append({"question": question, "cot": cot, "answer": answer})
        questions_set.add(question)

        pbar.update(length)
        tot_tokens += length
    
    pbar.close()
    
    print(f"Total number of tokens: {tot_tokens}")
    print(f"Num of samples: {len(samples)}")

    return samples


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    data_dir = to_absolute_path(cfg.dataset.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    samples = generate_samples(cfg)
    
    dataset = Dataset.from_list(samples)
    
    dataset_dict = dataset.train_test_split(test_size=0.05)

    dataset_dict.save_to_disk(data_dir)


if __name__ == "__main__":
    main()