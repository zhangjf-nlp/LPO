import os
import json
import argparse

import numpy as np
import pandas as pd
import torch

from config_and_utils import (
    get_tokenizer, pad_and_concat,
    get_reward_fn, intent_label2id
)
from dailydialog_dataset import DailyDialogDataset
from gpu_map_reduce import TaskConsumer, TaskProducer, BatchTaskConsumer

def pad_or_truncate(ids, target_length, pad_id):
    if ids.shape[1] > target_length:
        ids = ids[:,:target_length]
    elif ids.shape[1] < target_length:
        padding_ids = torch.LongTensor(size=(ids.shape[0], target_length-ids.shape[1]))
        padding_ids.fill_(pad_id)
        padding_ids = padding_ids.to(ids.device)
        ids = torch.cat([ids, padding_ids], dim=1)
    return ids


class DailyDialogComparisonDataset:
    def __init__(self, usage="train", preference="consistency"):
        cache_file = f"one2many_inference_with_intent_scores.json"
        with open(cache_file, "r", encoding="utf-8") as f:
            inference = json.loads(f.read())
        assert not any([data is None for data in inference]), f"the data in cache file contains 'None': {cache_file}"
        self.data = [data for data in inference if data["usage"]==usage]
        self.usage = usage
        self.preference = preference
        self.tokenizer = get_tokenizer()

        self.many = 32
        self.num_pairs_per_group = 2

    def __len__(self):
        return len(self.data) * self.num_pairs_per_group

    def intent_analysis(self):
        preference2scores = {preference:[] for preference in (list(intent_label2id.keys())+["consistency"])}
        for data in self.data:
            for preference in preference2scores:
                preference2scores[preference] += data[f"{preference}_score"]
        preference2scores = pd.DataFrame({k:np.mean(v) for k,v in preference2scores.items()}, index=["mean"])
        print(preference2scores)

    def __getitem__(self, item):
        item, bias = item // self.num_pairs_per_group, item % self.num_pairs_per_group

        data = self.data[item]
        prompt_ids = data["input_ids"]#[0]
        prompt_len = len(prompt_ids)
        prompt_ids = torch.LongTensor(prompt_ids)
        prompt = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)[0]
        reward = torch.Tensor(data[f"{self.preference}_score"])

        sorted_reward, indices = torch.sort(reward, descending=True)
        indices = indices.tolist()
        chosen = data["generation"][indices[bias]]
        rejected = data["generation"][indices[bias-self.num_pairs_per_group]]
        chosen_ids = self.tokenizer(chosen+self.tokenizer.eos_token, return_tensors="pt").input_ids
        rejected_ids = self.tokenizer(rejected+self.tokenizer.eos_token, return_tensors="pt").input_ids
        return prompt_ids, chosen_ids, rejected_ids

    def lpo_collate_fn(self, batch_items):
        batch_prompt_ids, batch_chosen_ids, batch_rejected_ids = [], [], []
        for prompt_ids, chosen_ids, rejected_ids in batch_items:
            batch_prompt_ids.append(prompt_ids)
            batch_chosen_ids.append(chosen_ids)
            batch_rejected_ids.append(rejected_ids)

        pad_token_id = self.tokenizer.pad_token_id
        return {
            "prompt_ids": pad_and_concat(batch_prompt_ids, pad_token_id),
            "chosen_ids": pad_and_concat(batch_chosen_ids, pad_token_id),
            "rejected_ids": pad_and_concat(batch_rejected_ids, pad_token_id),
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
        inputs = {
            "input_ids": pad_and_concat(input_ids, self.tokenizer.pad_token_id).long(),
            "labels": pad_and_concat(labels, -100).long()
        }
        return inputs


class RewardProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        input_file = f"one2many_inference.json"
        with open(input_file, "r", encoding="utf-8") as f:
            tasks = json.loads(f.read())
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None and \
               all([task[key]==result[key]
                    for key in ["input_ids", "usage", "index", "prompt", "generation"]]) and \
               all([f"{preference}_score" in result for preference in
                    (list(intent_label2id.keys())+["consistency"])])


class RewardConsumer(TaskConsumer):
    def init_model(self) -> None:
        self.reward_fn = get_reward_fn(device=self.device)
        self.data = []
        for usage in ["train", "validation", "test"]:
            self.data += DailyDialogDataset(usage=usage).data

    def process_task(self, task: dict) -> dict:
        item = self.data[task["index"]]
        if not item["prompt"].replace(" ","").endswith(task["prompt"].replace(" ","")):
            print(f"prompt not completely matched:\n"
                  f"from dataset:\t{item['prompt'][-50:]}\n"
                  f"from task:\t{task['prompt'][-50:]}")
        prompts = [task["prompt"]] * len(task["generation"])
        labels = task["generation"]
        intent_probs = self.reward_fn(prompts, labels, return_probs=True)

        for intent, id in intent_label2id.items():
            task[f"{intent}_score"] = intent_probs[:,id].tolist()
        task["consistency_score"] = intent_probs[:,item["intent"]].tolist()

        return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for map-reduce processing on multi-gpu')
    parser.add_argument("--py_file_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6380)
    parser.add_argument("--task_name", type=str, default="one2many_inference_with_intent_scores")
    parser.add_argument("--output_file", type=str, default=f"one2many_inference_with_intent_scores.json")
    parser.add_argument("--erase", type=int, default=0)
    args = parser.parse_args()

    if args.py_file_name is None:
        args.py_file_name = os.path.basename(__file__)

    if args.local_rank == -1:
        producer = RewardProducer(args)
        producer.run()
        for usage in ["train", "validation", "test"]:
            dataset = DailyDialogComparisonDataset(usage)
            dataset.intent_analysis()
    else:
        consumer = RewardConsumer(args)
        consumer.run()
