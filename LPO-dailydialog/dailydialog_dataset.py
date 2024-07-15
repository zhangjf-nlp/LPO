import json

import torch
from tqdm import tqdm

from config_and_utils import (
    get_tokenizer, pad_and_concat,
    input_max_length, output_max_length,
    BASE_RWD_MODEL_PATH, get_sft_dataset,
)
from transformers import AutoTokenizer


def pad_or_truncate(ids, target_length, pad_id):
    if ids.shape[1] > target_length:
        ids = ids[:,:target_length]
    elif ids.shape[1] < target_length:
        padding_ids = torch.LongTensor(size=(ids.shape[0], target_length-ids.shape[1]))
        padding_ids.fill_(pad_id)
        padding_ids = padding_ids.to(ids.device)
        ids = torch.cat([ids, padding_ids], dim=1)
    return ids


class DailyDialogDataset:
    EOU_TOKEN = " <EOU> "
    def __init__(self, usage="train", context_size=5):
        self.dataset = get_sft_dataset(usage=usage)
        self.usage = usage
        self.tokenizer = get_tokenizer()
        self.rwd_tokenizer = AutoTokenizer.from_pretrained(BASE_RWD_MODEL_PATH)
        self.data = []
        for item in tqdm(self.dataset):
            contexts = []
            for utterance, emotion, intent in zip(
                item["dialog"], item["emotion"], item["act"]
            ):
                utterance = utterance.strip()
                if len(contexts) >= context_size:
                    context = self.EOU_TOKEN.join(contexts[-context_size:]) + self.EOU_TOKEN
                    target = utterance + self.EOU_TOKEN
                    self.data.append({
                        "prompt": context,
                        "label": target,
                        "emotion": emotion,
                        "intent": intent - 1
                    })
                contexts.append(utterance)

    def intent_analysis(self, context_size=5):
        all_intents = []
        for item in tqdm(self.dataset):
            contexts = []
            for utterance, emotion, intent in zip(
                    item["dialog"], item["emotion"], item["act"]
            ):
                if len(contexts) >= context_size:
                    all_intents.append(intent-1)
                contexts.append(utterance)
        print(f"intent analysis with context size {context_size}:")
        for i in range(max(all_intents)+1):
            print(f"number / ratio of intent-{i}: {all_intents.count(i)} / {all_intents.count(i)/len(all_intents):.5f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        prompt, label, intent = self.data[item]["prompt"], self.data[item]["label"], self.data[item]["intent"]
        self.tokenizer.truncation_side = "left"
        prompt_ids = self.tokenizer([prompt], return_tensors="pt", truncation=True,
                                   max_length=input_max_length).input_ids
        self.tokenizer.truncation_side = "right"
        label_ids = self.tokenizer([label+self.tokenizer.eos_token], return_tensors="pt", truncation=True,
                               max_length=output_max_length).input_ids
        return {"prompt_ids": prompt_ids, "label_ids": label_ids, "intent": torch.LongTensor([intent])}

    def sft_collate_fn(self, batch_items):
        input_ids = pad_and_concat(
            [torch.cat([item["prompt_ids"], item["label_ids"]], dim=1)
             for item in batch_items],
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left", dim=1
        )
        labels = pad_and_concat(
            [torch.cat([item["prompt_ids"].clone().fill_(-100), item["label_ids"]], dim=1)
             for item in batch_items],
            pad_token_id=-100,
            padding_side="left", dim=1
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def context_utterance_to_rwd_inputs(self, context, utterance):
        from config_and_utils import context_utterance_to_rwd_inputs
        return context_utterance_to_rwd_inputs(
            context=context,
            utterance=utterance,
            tokenizer=self.rwd_tokenizer
        )

    def rwd_collate_fn(self, batch_items):
        batch_inputs = []
        for batch_item in batch_items:
            context = self.tokenizer.decode(batch_item["prompt_ids"][0], skip_special_tokens=True)
            utterance = self.tokenizer.decode(batch_item["label_ids"][0], skip_special_tokens=True)
            inputs = self.context_utterance_to_rwd_inputs(context, utterance)
            batch_inputs.append(inputs)

        input_ids = pad_and_concat([batch_input[0] for batch_input in batch_inputs],
                                   pad_token_id=self.rwd_tokenizer.pad_token_id)
        attention_mask = pad_and_concat([batch_input[1] for batch_input in batch_inputs],
                                        pad_token_id=0)
        token_type_ids = pad_and_concat([batch_input[2] for batch_input in batch_inputs],
                                        pad_token_id=2)
        labels = torch.cat([item["intent"] for item in batch_items], dim=0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

    def save_to_json(self):
        with open(f"{self.usage}_data.json", "w") as f:
            f.write(json.dumps(self.data, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    for usage in ["train", "validation", "test"]:
        dataset = DailyDialogDataset(usage)
        print(f"{usage}: {len(dataset)}")
        dataset.intent_analysis(context_size=5)
        dataset.intent_analysis(context_size=0)
        dataset.save_to_json()