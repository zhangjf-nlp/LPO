import json

import torch
from tqdm import tqdm

from config_and_utils import get_tokenizer, get_comparison_dataset, input_max_length, output_max_length, pad_and_concat

def pad_or_truncate(ids, target_length, pad_id):
    if ids.shape[1] > target_length:
        ids = ids[:,:target_length]
    elif ids.shape[1] < target_length:
        padding_ids = torch.LongTensor(size=(ids.shape[0], target_length-ids.shape[1]))
        padding_ids.fill_(pad_id)
        padding_ids = padding_ids.to(ids.device)
        ids = torch.cat([ids, padding_ids], dim=1)
    return ids


class SummaryComparisonDataset:
    def __init__(self, usage="train", preference="human"):
        if preference=="human":
            dataset = get_comparison_dataset()
            if usage=="train":
                dataset = dataset["train"]
            elif usage=="validation":
                dataset = list(dataset["valid1"]) + list(dataset["valid2"])
                dataset = dataset[:len(dataset)//10]
                # DPO peforms slowly on validation
                # for fairness and generalizability, we only use 10% of the orginal validation data
                # train / validation / test - sft
                # 92534 / 8379 / 6553
            elif usage=="test":
                dataset = dataset["test"]
            else:
                raise NotImplementedError(usage)
            self.dataset = [data for data in tqdm(dataset) if data["chosen"]!=data["rejected"]]
        else:
            jsonl_path = f"./openai_summarize_comparisons_{preference}/{usage}.jsonl"
            with open(jsonl_path, "r", encoding="utf-8") as f:
                lines = [line for line in f]
            self.dataset = [json.loads(line) for line in tqdm(lines)]
        self.usage = usage
        self.preference = preference
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.dataset)

    def process_data(self, data):
        prompt: str = data["prompt"]
        chosen: str = data["chosen"]
        rejected: str = data["rejected"]
        tldr = "TL;DR:"
        assert chosen.startswith(tldr), chosen
        assert rejected.startswith(tldr), rejected

        eos = "<|endoftext|>"
        if eos in chosen:
            chosen = chosen[:chosen.index(eos)]
        if eos in rejected:
            rejected = rejected[:rejected.index(eos)]

        prompt = prompt.strip() + "\n" + tldr + " "
        chosen = chosen[len(tldr):].strip()
        rejected = rejected[len(tldr):].strip()

        return prompt, chosen, rejected

    def __getitem__(self, item):
        data = self.dataset[item]
        prompt, chosen, rejected = self.process_data(data)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        chosen_ids = self.tokenizer(chosen, return_tensors="pt").input_ids
        rejected_ids = self.tokenizer(rejected, return_tensors="pt").input_ids
        return prompt_ids, chosen_ids, rejected_ids

    def lpo_collate_fn(self, batch_items):
        batch_prompt_ids, batch_chosen_ids, batch_rejected_ids = [], [], []
        for prompt_ids, chosen_ids, rejected_ids in batch_items:
            batch_prompt_ids.append(prompt_ids)
            batch_chosen_ids.append(chosen_ids)
            batch_rejected_ids.append(rejected_ids)

        pad_token_id = self.tokenizer.pad_token_id
        return {
            "prompt_ids": pad_and_concat(batch_prompt_ids, pad_token_id)[:,:input_max_length].long(),
            "chosen_ids": pad_and_concat(batch_chosen_ids, pad_token_id)[:,:output_max_length].long(),
            "rejected_ids": pad_and_concat(batch_rejected_ids, pad_token_id)[:,:output_max_length].long(),
        }

    def dpo_collate_fn(self, batch_items):
        input_ids, labels = [], []
        for prompt_ids, chosen_ids, rejected_ids in batch_items:
            chosen_input_ids = torch.cat([prompt_ids, chosen_ids], dim=1)
            rejected_input_ids = torch.cat([prompt_ids, rejected_ids], dim=1)
            chosen_labels = torch.cat([prompt_ids.clone().fill_(-100), chosen_ids], dim=1)
            rejected_labels = torch.cat([prompt_ids.clone().fill_(-100), rejected_ids], dim=1)
            input_ids.append(chosen_input_ids)
            input_ids.append(rejected_input_ids)
            labels.append(chosen_labels)
            labels.append(rejected_labels)
        max_length = input_max_length + output_max_length
        inputs = {
            "input_ids": pad_and_concat(input_ids, self.tokenizer.pad_token_id).long()[:,:max_length],
            "labels": pad_and_concat(labels, -100).long()[:,:max_length]
        }
        return inputs