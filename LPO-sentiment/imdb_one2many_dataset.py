import os
import json
import argparse

import torch
import numpy as np

from config_and_utils import get_tokenizer, input_max_length, output_max_length, \
    auto_find_checkpoint_dir, pad_and_concat
from imdb_dataset import IMDBDataset

from gpu_map_reduce import BatchTaskConsumer, TaskConsumer, TaskProducer


def pad_or_truncate(ids, target_length, pad_id):
    if ids.shape[1] > target_length:
        ids = ids[:,:target_length]
    elif ids.shape[1] < target_length:
        padding_ids = torch.LongTensor(size=(ids.shape[0], target_length-ids.shape[1]))
        padding_ids.fill_(pad_id)
        padding_ids = padding_ids.to(ids.device)
        ids = torch.cat([ids, padding_ids], dim=1)
    return ids


class IMDBDatasetForInference(IMDBDataset):
    def __getitem__(self, item):
        text = self.dataset[item]["text"]
        input_ids = self.tokenizer([text], return_tensors="pt", truncation=True,
                                   max_length=input_max_length).input_ids
        return {"input_ids": input_ids, "reference": text}


class IMDBOne2ManyDataset:
    def __init__(self, usage="train", many=32):
        cache_file = f"one2many_inference.json"
        with open(cache_file, "r", encoding="utf-8") as f:
            inference = json.loads(f.read())
        assert not any([data is None for data in inference]), f"the data in cache file contains 'None': {cache_file}"
        self.data = [data for data in inference if data["usage"]==usage]
        self.usage = usage
        self.tokenizer = get_tokenizer()
        self.many = many

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        assert len(data["generation"]) >= self.many
        data["generation"] = data["generation"][:self.many]

        prompt_ids = torch.LongTensor(data["input_ids"])
        prompt_len = prompt_ids.shape[1]
        pred_ids: torch.Tensor = self.tokenizer(data["generation"], return_tensors="pt", padding="max_length",
                                                truncation=True, max_length=output_max_length).input_ids
        input_ids = prompt_ids
        # [1, prompt_len]
        labels = torch.where(pred_ids == self.tokenizer.pad_token_id, -100, pred_ids)
        # [many, output_max_length]
        prior_ids = pad_or_truncate(prompt_ids, input_max_length, self.tokenizer.pad_token_id)
        # [1, input_max_length]
        post_ids = pad_or_truncate(pred_ids, output_max_length, self.tokenizer.pad_token_id)
        # [many, output_max_length]

        assert input_ids.shape == (1, prompt_len), input_ids.shape
        assert labels.shape == (self.many, output_max_length), labels.shape
        assert prior_ids.shape == (1, input_max_length), prior_ids.shape
        assert post_ids.shape == (self.many, output_max_length), post_ids.shape
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


def extend_one2many_dataset_with_mini_many(one2many_dataset_class):
    class one2many_dataset_class_with_across_batch(one2many_dataset_class):
        def __init__(self, total_many, mini_many, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.total_many = total_many
            self.mini_many = mini_many
            self.index_queue = []
            self.present_index = None
            self.present_order = None
            self.present_bias = 0

        def __getitem__(self, index):
            self.index_queue.append(index)
            if self.present_bias == 0:
                self.present_index = self.index_queue.pop(0)
                self.present_order = torch.from_numpy(np.random.permutation(self.total_many))
            else:
                self.present_order = torch.roll(self.present_order, self.mini_many, dims=0)
            self.present_bias = (self.present_bias + self.mini_many) % self.total_many

            item = super().__getitem__(self.present_index)
            item_new = {}
            for k,v in item.items():
                if v.shape[0] == self.many:
                    v = v[self.present_order]
                    v = v.view(self.total_many // self.mini_many, self.mini_many, *v.shape[1:])
                    item_new[k] = v
                else:
                    item_new[k] = v
            return item_new

    return one2many_dataset_class_with_across_batch


class One2ManyInferenceProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        tasks = []
        for usage in ["train", "validation", "test"]:
            dataset = IMDBDatasetForInference(usage)
            for item in dataset:
                task = {"input_ids": item["input_ids"].tolist(),
                        "usage": usage, "from_model": self.args.sft_model_path}
                tasks.append(task)
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None and \
               all([task[key]==result[key]
                    for key in ["input_ids", "usage", "from_model"]])


class One2ManyInferenceConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        from transformers import GPT2LMHeadModel
        self.model = GPT2LMHeadModel.from_pretrained(
            self.args.sft_model_path,
            device_map=f"cuda:{self.args.local_rank}",
            torch_dtype=torch.float16,
        ).eval()
        self.tokenizer = get_tokenizer()

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        many = 32
        prompt_len = max([len(task["input_ids"][0]) for task in tasks])
        input_ids = pad_and_concat(
            [torch.LongTensor(task["input_ids"]) for task in tasks],
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left"
        ).to(self.device)
        assert input_ids.shape[1] == prompt_len
        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_k=50,
            min_length=input_max_length+output_max_length,
            max_new_tokens=output_max_length,
            num_return_sequences=many,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generations = self.tokenizer.batch_decode(
            outputs[:,prompt_len:], skip_special_tokens=True
        )
        prompts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )
        for i,task in enumerate(tasks):
            task["prompt"] = prompts[i]
            task["generation"] = generations[i*many:(i+1)*many]
            assert torch.all(outputs[i*many:(i+1)*many,:prompt_len]==input_ids[i:i+1,:])
        return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for map-reduce processing on multi-gpu')
    parser.add_argument("--py_file_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6380)
    parser.add_argument("--task_name", type=str, default="one2many_inference")
    parser.add_argument("--output_file", type=str, default=f"one2many_inference.json")
    parser.add_argument("--erase", type=int, default=0)
    parser.add_argument("--sft_model_path", type=auto_find_checkpoint_dir,
                        default=auto_find_checkpoint_dir(f"sft/checkpoint"))
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    if args.py_file_name is None:
        args.py_file_name = os.path.basename(__file__)

    if args.local_rank == -1:
        producer = One2ManyInferenceProducer(args)
        producer.run()
    else:
        consumer = One2ManyInferenceConsumer(args)
        consumer.run()
