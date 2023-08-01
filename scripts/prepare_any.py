"""Implementation derived from https://github.com/tloen/alpaca-lora"""
import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import requests
import json
from torch.utils.data import random_split
from lit_gpt import Tokenizer, Config
from tqdm import tqdm


IGNORE_INDEX = -1

DATA_FILE_NAME = "input.txt"


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample/arxiv_sample.jsonl"),
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf"),
    destination_path: Path = Path("data/mydata"),
    test_split_ratio: float = 0.9,  # default 90% train, 10% validation
    max_seq_length: int = 256,
    seed: int = 42,
) -> None:
    """Prepare any dataset for finetuning (akin to Shakespheare full tuning).

    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))
    tokenizer = Tokenizer(checkpoint_dir)

    destination_path.mkdir(parents=True, exist_ok=True)
    if not source_path.exists():
        raise AssertionError(f"{source_file_path} does not exist")

    data = []

    with open(source_path, encoding="utf-8") as f:
        for row in tqdm(f):
            text = json.loads(row)["text"]
            data.append(text)
    # Partition the dataset into train and test
    train_split_size = int(len(data) * test_split_ratio)
    test_split_size = len(data) - train_split_size
    train_set, test_set = random_split(
        data,
        lengths=(train_split_size, test_split_size),
        generator=torch.Generator().manual_seed(seed),
    )
    train_set, test_set = list(train_set), list(test_set)

    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    print("Processing train split ...")
    train_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(train_set)
    ]
    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = [
        prepare_line(line, tokenizer, max_seq_length) for line in tqdm(test_set)
    ]
    torch.save(test_set, destination_path / "test.pt")


def prepare_line(line: str, tokenizer: Tokenizer, max_length: int):
    """Processes a single sample.

    This function processes the line to produce the tokenized version of it.
    """
    encoded_full_prompt = tokenize(tokenizer, line, max_length=max_length, eos=False)
    return {
        "input_ids": encoded_full_prompt,
        "labels": encoded_full_prompt,
    }


def tokenize(
    tokenizer: Tokenizer, string: str, max_length: int, eos=True
) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)