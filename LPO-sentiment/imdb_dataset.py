import torch
import numpy as np

from config_and_utils import (
    get_tokenizer, get_sft_dataset,
    input_max_length, output_max_length, pad_and_concat
)


def pad_or_truncate(ids, target_length, pad_id):
    if ids.shape[1] > target_length:
        ids = ids[:,:target_length]
    elif ids.shape[1] < target_length:
        padding_ids = torch.LongTensor(size=(ids.shape[0], target_length-ids.shape[1]))
        padding_ids.fill_(pad_id)
        padding_ids = padding_ids.to(ids.device)
        ids = torch.cat([ids, padding_ids], dim=1)
    return ids


class IMDBDataset:
    def __init__(self, usage="train"):
        if usage=="train":
            self.dataset = get_sft_dataset(split="train")
        else:
            dataset = get_sft_dataset(split="test")
            dataset_negative = [item for item in dataset if item["label"] == 0]
            dataset_positive = [item for item in dataset if item["label"] == 1]
            if usage=="validation":
                self.dataset = dataset_negative[:2500] + dataset_positive[:2500]
            elif usage=="test":
                self.dataset = dataset_negative[-2500:] + dataset_positive[-2500:]
            else:
                raise NotImplementedError(usage)
        self.usage = usage
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        text, label = self.dataset[item]["text"], self.dataset[item]["label"]
        input_ids = self.tokenizer([text], return_tensors="pt", truncation=True,
                                   max_length=input_max_length+output_max_length).input_ids
        label = torch.Tensor([label]).long()
        return {"input_ids": input_ids, "label": label}

    def sft_collate_fn(self, batch_items):
        input_ids = pad_and_concat(
            [item["input_ids"] for item in batch_items],
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left", dim=1
        )
        labels = torch.where(input_ids==self.tokenizer.pad_token_id, -100, input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def vae_collate_fn(self, batch_items):
        input_ids = pad_and_concat(
            [item["input_ids"] for item in batch_items],
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left", dim=1
        )
        labels = torch.where(input_ids==self.tokenizer.pad_token_id, -100, input_ids)
        labels[:,:input_max_length] = -100
        post_ids = input_ids[:,input_max_length:]
        return {
            "post_ids": post_ids,
            "input_ids": input_ids,
            "labels": labels,
        }

    def label_analysis(self):
        labels = [self.dataset[item]["label"] for item in range(len(self.dataset))]
        print(f"{self.usage}:")
        print(f"--positive: {len([_ for _ in labels if _ == 1])}")
        print(f"--negative: {len([_ for _ in labels if _ == 0])}")
        print(f"--average: {np.mean([labels])*100:.2f}%")


if __name__ == "__main__":
    for usage in ["train", "validation", "test"]:
        dataset = IMDBDataset(usage)
        dataset.label_analysis()
