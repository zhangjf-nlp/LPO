import torch

from config_and_utils import (
    get_tokenizer,
    get_sft_dataset,
    input_max_length,
    output_max_length,
    pad_and_concat
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


class SummaryDataset:
    def __init__(self, usage="train"):
        dataset = get_sft_dataset()
        if usage=="train":
            self.dataset = dataset["train"]
        elif usage=="validation":
            self.dataset = dataset["valid"]
        elif usage=="test":
            self.dataset = dataset["test"]
        else:
            raise NotImplementedError(usage)
        self.usage = usage
        self.tokenizer = get_tokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        prompt, label = self.dataset[item]["prompt"], self.dataset[item]["label"]
        input_ids = self.tokenizer([prompt + label], return_tensors="pt", truncation=True,
                                   max_length=input_max_length+output_max_length).input_ids
        return {"input_ids": input_ids}

    def sft_collate_fn(self, batch_items):
        input_ids = pad_and_concat(
            [item["input_ids"] for item in batch_items],
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left", dim=1
        )
        attention_mask = (input_ids==self.tokenizer.pad_token_id).long()
        labels = torch.where(attention_mask, -100, input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
