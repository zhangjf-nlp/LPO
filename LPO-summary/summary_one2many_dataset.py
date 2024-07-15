import argparse
import json
import os
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from config_and_utils import (
    input_max_length, output_max_length,
    BASE_MODEL_PATH, get_tokenizer
)
from summary_comparison_dataset import SummaryComparisonDataset
from summary_dataset import SummaryDataset
from transformers import GPTJForCausalLM


def pad_or_truncate(ids, target_length, pad_id):
    if ids.shape[1] > target_length:
        ids = ids[:,:target_length]
    elif ids.shape[1] < target_length:
        padding_ids = torch.LongTensor(size=(ids.shape[0], target_length-ids.shape[1]))
        padding_ids.fill_(pad_id)
        padding_ids = padding_ids.to(ids.device)
        ids = torch.cat([ids, padding_ids], dim=1)
    return ids


class SummaryOne2ManyDataset(SummaryComparisonDataset):
    def __init__(self, usage="train", preference="human"):
        super().__init__(usage, preference)
        prompt2generation = defaultdict(set)
        for data in self.dataset:
            prompt, chosen, rejected = self.process_data(data)
            if len(chosen)>1: prompt2generation[prompt].add(chosen)
            if len(rejected)>1: prompt2generation[prompt].add(rejected)
        self.total_many = 4
        self.prompt2generation = {prompt:list(generation)
                                  for prompt,generation in prompt2generation.items() if len(generation)>=self.total_many}
        self.prompts = list(self.prompt2generation.keys())

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        prompt = self.prompts[item]
        generation = self.prompt2generation[prompt]
        generation = [generation[i] for i in list(np.random.permutation(len(generation))[:self.total_many])]
        prompt_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
        pred_ids: torch.Tensor = self.tokenizer(generation, return_tensors="pt", padding="max_length",
                                                truncation=True, max_length=output_max_length).input_ids
        prompt_len = prompt_ids.shape[1]
        input_ids = prompt_ids
        # [1, prompt_len]
        labels = torch.where(pred_ids == self.tokenizer.pad_token_id, -100, pred_ids)
        # [N, output_max_length]
        prior_ids = pad_or_truncate(prompt_ids, input_max_length, self.tokenizer.pad_token_id)
        # [1, input_max_length]
        post_ids = pad_or_truncate(pred_ids, output_max_length, self.tokenizer.pad_token_id)
        # [N, output_max_length]

        N = len(generation)
        assert input_ids.shape == (1, prompt_len), input_ids.shape
        assert labels.shape == (N, output_max_length), labels.shape
        assert prior_ids.shape == (1, input_max_length), prior_ids.shape
        assert post_ids.shape == (N, output_max_length), post_ids.shape
        assert not torch.any(input_ids==self.tokenizer.pad_token_id), input_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prior_ids": prior_ids,
            "post_ids": post_ids
        }

    def collate_fn(self, batch_items):
        assert len(batch_items) == 1, f"batch_size {len(batch_items)} > 1 " \
                                      f"is not supported in one2many training"
        return batch_items[0]

    def sft_collate_fn(self, batch_items):
        assert len(batch_items) == 1, f"batch_size {len(batch_items)} > 1 " \
                                      f"is not supported in one2many training"
        prompt_ids = batch_items[0]["input_ids"]
        pred_labels = batch_items[0]["labels"]
        prompt_ids = prompt_ids.repeat(pred_labels.shape[0], 1)
        input_ids = torch.cat([prompt_ids, torch.where(pred_labels==-100, self.tokenizer.pad_token_id, pred_labels)], dim=1)
        labels = torch.cat([torch.zeros_like(prompt_ids).fill_(-100), pred_labels], dim=1)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def vae_collate_fn(self, batch_items):
        assert len(batch_items) == 1, f"batch_size {len(batch_items)} > 1 " \
                                      f"is not supported in one2many training"
        post_ids = batch_items[0]["post_ids"]
        prompt_ids = batch_items[0]["input_ids"]
        pred_labels = batch_items[0]["labels"]
        prompt_ids = prompt_ids.repeat(pred_labels.shape[0], 1)
        input_ids = torch.cat([prompt_ids, torch.where(pred_labels==-100, self.tokenizer.pad_token_id, pred_labels)], dim=1)
        labels = torch.cat([torch.zeros_like(prompt_ids).fill_(-100), pred_labels], dim=1)
        return {
            "post_ids": post_ids,
            "input_ids": input_ids,
            "labels": labels,
        }


def extend_one2many_dataset_with_mini_many(one2many_dataset_class, drop_other_mini=False):
    class one2many_dataset_class_with_across_batch(one2many_dataset_class):
        def __init__(self, total_many, mini_many, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.total_many = total_many
            self.mini_many = mini_many
            self.index_queue = []
            self.present_index = None
            self.present_order = torch.arange(self.total_many).view(self.mini_many, -1).T.reshape(-1)
            self.present_bias = 0

        def __getitem__(self, index):
            self.index_queue.append(index)
            if self.present_bias == 0:
                self.present_index = self.index_queue.pop(0)
            self.present_order = torch.roll(self.present_order, self.mini_many, dims=0)
            self.present_bias = (self.present_bias + self.mini_many) % self.total_many

            item = super().__getitem__(self.present_index)
            item_new = {}
            for k,v in item.items():
                if v.shape[0] == self.total_many:
                    if drop_other_mini:
                        v = v[self.present_order[:self.mini_many]]
                    else:
                        v = v[self.present_order]
                        v = v.view(self.total_many // self.mini_many, self.mini_many, *v.shape[1:])
                    item_new[k] = v
                else:
                    item_new[k] = v
            return item_new

    return one2many_dataset_class_with_across_batch


class GeneratedSummaryOne2ManyDataset(SummaryOne2ManyDataset):
    def __init__(self, usage="train"):
        self.usage = usage
        self.tokenizer = get_tokenizer()
        with open("summary_one2many_generation.json", "r", encoding="utf-8") as f:
            data = json.loads(f.read())
            data = [d for d in data if d is not None]

        data = [d for d in data if d["usage"]==usage]

        self.total_many = 32
        self.prompt2generation = {d["prompt"]:d["generation"] for d in data}
        self.prompts = list(self.prompt2generation.keys())


from gpu_map_reduce import TaskConsumer, TaskProducer
class SummaryGenerationProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        tasks = []
        for usage in ["train", "validation"]:
            dataset = SummaryDataset(usage=usage)
            for data in tqdm(dataset.dataset, desc=f"reading prompts for {usage}"):
                tasks.append({"prompt": data["prompt"], "usage": usage, "generation": []})
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None and task["prompt"] == result["prompt"] and not any([gen.strip()=="" for gen in result["generation"]])

    def load_cached_results(self):
        if os.path.exists(self.args.output_file) and not self.args.erase:
            with open(self.args.output_file, "r", encoding="utf-8") as f:
                results = json.loads(f.read())
            if len(results) == len(self.tasks):
                for index, result in enumerate(results):
                    if result is not None and self.tasks[index]["prompt"] == result["prompt"]:
                        result["usage"] = self.tasks[index]["usage"]
                        self.tasks[index]["generation"] = [gen for gen in result["generation"] if gen.strip()!=""]
                return results
        results = [None for index in range(len(self.tasks))]
        return results


class SummaryGenerationConsumer(TaskConsumer):
    def init_model(self) -> None:
        self.model = GPTJForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map=self.device,
        ).eval()
        self.tokenizer = get_tokenizer()
        self.tokenizer.truncation_side = "left"

    def process_task(self, task: dict) -> dict:
        many, mini_many = 32, 8
        inputs = self.tokenizer([task["prompt"]], return_tensors="pt",
                                truncation=True, max_length=input_max_length)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        inputs_len = inputs["input_ids"].shape[1]
        generation = []
        for _ in range(many // mini_many):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    num_return_sequences=mini_many,
                    do_sample=True,
                    top_p=0.95,
                    max_new_tokens=output_max_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generation += self.tokenizer.batch_decode(output_ids[:,inputs_len:], skip_special_tokens=True)
        task["generation"] = generation
        return task


def parse_args():
    parser = argparse.ArgumentParser(description='parse args for one2many generation and reward on multi-gpu')
    parser.add_argument("--py_file_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6380)
    parser.add_argument("--task_name", type=str, default="summary_one2many_generation")
    parser.add_argument("--output_file", type=str, default="summary_one2many_generation.json")
    parser.add_argument("--erase", type=int, default=0)
    args = parser.parse_args()

    if args.py_file_name is None:
        args.py_file_name = os.path.basename(__file__)

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        producer = SummaryGenerationProducer(args)
        producer.run()
    else:
        consumer = SummaryGenerationConsumer(args)
        consumer.run()

if __name__ == "__main__":
    main()
