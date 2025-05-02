from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def get_vocab(cfg):
    vocab = ["BoT", "EoT", "BoS", "EoS", "|", "[", "]", "aha", "wait", "[UNK]", "[PAD]"]

    # Add layers
    vocab.extend([f"l{i}" for i in range(cfg.dataset.graph_generation.max_graph_depth)])

    # Add nodes
    vocab.extend([f"n{i}" for i in range(cfg.dataset.graph_generation.max_graph_depth * cfg.dataset.graph_generation.max_num_nodes_per_layer)])

    # Add edge costs
    vocab.extend([str(c) for c in range(cfg.dataset.graph_generation.max_edge_cost * cfg.dataset.graph_generation.max_graph_depth)])

    return vocab

def get_tokenizer(cfg):
    vocab = get_vocab(cfg)

    word_to_id = {word: idx for idx, word in enumerate(vocab)}

    # Build the Tokenizers tokenizer
    tokenizer = Tokenizer(models.WordLevel(vocab=word_to_id, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Wrap it directly as a Fast tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="BoS",
        eos_token="EoS",
    )

    return fast_tokenizer
