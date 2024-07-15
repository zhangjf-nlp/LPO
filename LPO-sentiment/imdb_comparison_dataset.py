import os
import json
import argparse

import torch

from config_and_utils import RWD_MODEL_PATH, get_tokenizer, output_max_length, pad_and_concat
from gpu_map_reduce import TaskConsumer, TaskProducer
from collections import defaultdict

def pad_or_truncate(ids, target_length, pad_id):
    if ids.shape[1] > target_length:
        ids = ids[:,:target_length]
    elif ids.shape[1] < target_length:
        padding_ids = torch.LongTensor(size=(ids.shape[0], target_length-ids.shape[1]))
        padding_ids.fill_(pad_id)
        padding_ids = padding_ids.to(ids.device)
        ids = torch.cat([ids, padding_ids], dim=1)
    return ids


preference_to_reward_fn = {
    "positive": lambda logits:logits,
    "negative": lambda logits:-logits,
    "neutral": lambda logits:-torch.abs(logits)
}
available_preferences = list(preference_to_reward_fn.keys())


class IMDBComparisonDataset:
    def __init__(self, usage="train", preference="positive"):
        cache_file = f"one2many_inference_with_reward.json"
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

    def __getitem__(self, item):
        item, bias = item // self.num_pairs_per_group, item % self.num_pairs_per_group

        data = self.data[item]
        prompt_ids = data["input_ids"]
        prompt_ids = torch.LongTensor(prompt_ids)
        logits = torch.Tensor(data["reward_logits"])
        reward = preference_to_reward_fn[self.preference](logits)

        sorted_reward, indices = torch.sort(reward, descending=True)
        indices = indices.tolist()
        chosen = data["generation"][indices[bias]]
        rejected = data["generation"][indices[bias-self.num_pairs_per_group]]
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


    def logits_analysis(self):
        num_positive_counts = {num:0 for num in range(33)}
        for item in range(len(self.data)):
            data = self.data[item]
            logits = data["reward_logits"]
            num_positive = len([_ for _ in logits if _ > 0])
            num_positive_counts[num_positive] += 1
        print(f"{self.usage}: {num_positive_counts}")
        import matplotlib.pyplot as plt
        x = list(num_positive_counts.keys())
        y = list(num_positive_counts.values())
        y = [counts / len(self.data) for counts in y]
        plt.bar(x, y)
        plt.savefig(f"{self.usage}_comparison_logits_analysis.png")
        plt.clf()


class RewardProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        input_file = f"one2many_inference.json"
        with open(input_file, "r", encoding="utf-8") as f:
            tasks = json.loads(f.read())
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None and \
               all([task[key]==result[key]
                    for key in ["input_ids", "usage", "index", "prompt", "generation"]])


class RewardConsumer(TaskConsumer):
    def init_model(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RWD_MODEL_PATH,
            device_map=f"cuda:{self.args.local_rank}",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(RWD_MODEL_PATH)

    def process_task(self, task: dict) -> dict:
        input_sentences = [task["prompt"] + generation for generation in task["generation"]]
        with torch.no_grad():
            inputs = self.tokenizer(input_sentences, return_tensors="pt", padding="longest", truncation=True,
                                    max_length=self.model.config.max_position_embeddings)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits[:, 1] - outputs.logits[:, 0]
            logits = logits.tolist()
        task["reward_logits"] = logits
        return task


class ReferProducer(TaskProducer):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for map-reduce processing on multi-gpu')
    parser.add_argument("--py_file_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6380)
    parser.add_argument("--task_name", type=str, default="one2many_inference_reward")
    parser.add_argument("--output_file", type=str, default=f"one2many_inference_with_reward.json")
    parser.add_argument("--erase", type=int, default=0)
    args = parser.parse_args()

    if args.py_file_name is None:
        args.py_file_name = os.path.basename(__file__)

    if args.local_rank == -1:
        producer = RewardProducer(args)
        producer.run()
        for usage in ["train", "validation", "test"]:
            dataset = IMDBComparisonDataset(usage)
            dataset.logits_analysis()
    else:
        consumer = RewardConsumer(args)
        consumer.run()