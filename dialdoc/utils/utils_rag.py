import linecache
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset

import torch

from transformers import BartTokenizer, RagTokenizer, T5Tokenizer


def load_bm25_results(in_path):
    d_query_results = {}
    return d_query_results


def load_bm25(in_path):
    from rank_bm25 import BM25Okapi

    dataset = load_dataset("csv", data_files=[in_path], split="train", delimiter="\t", column_names=["title", "text"])
    passages = []
    for ex in dataset:
        for ele in ex["text"].split("####"):
            passages.append(ele)
    passages_tokenized = [passage.strip().lower().split() for passage in passages]
    bm25 = BM25Okapi(passages_tokenized)
    return bm25


def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )


def encode_line2(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    line = tuple(line.split("[SEP]"))
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        # Need to add eos token manually for T5
        if isinstance(self.tokenizer, T5Tokenizer):
            source_line += self.tokenizer.eos_token
            tgt_line += self.tokenizer.eos_token

        # Pad source and target to the right
        source_tokenizer = (
            self.tokenizer.question_encoder if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer
        )
        target_tokenizer = self.tokenizer.generator if isinstance(self.tokenizer, RagTokenizer) else self.tokenizer

        source_inputs = encode_line2(source_tokenizer, source_line, self.max_source_length, "right")
        target_inputs = encode_line(target_tokenizer, tgt_line, self.max_target_length, "right")

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        src_token_type_ids = source_inputs["token_type_ids"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "token_type_ids": src_token_type_ids,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        token_type_ids = torch.stack([x["token_type_ids"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        tgt_pad_token_id = (
            self.tokenizer.generator.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        src_pad_token_id = (
            self.tokenizer.question_encoder.pad_token_id
            if isinstance(self.tokenizer, RagTokenizer)
            else self.tokenizer.pad_token_id
        )
        y = trim_batch(target_ids, tgt_pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, src_pad_token_id, attention_mask=masks)
        keep_col_mask = input_ids.ne(src_pad_token_id).any(dim=0)
        token_type_ids = token_type_ids[:, keep_col_mask]
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "token_type_ids": token_type_ids,
            "decoder_input_ids": y,
        }
        return batch
