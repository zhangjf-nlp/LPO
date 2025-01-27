import os
import re

offline_mode = True
# this mode requires datasets and base models are pre-downloaded and put into this project
# our experiments are all carried in this mode since our machines are not connected to Internet
#   offline_mode = False
# this mode has not been tested yet

if offline_mode:
    BASE_MODEL_PATH = "./gpt2"
    BASE_RWD_MODEL_PATH = "./roberta-intent-classification"
    BERT_SCORE_MODEL_PATH = "microsoft/deberta-v2-xlarge-mnli"
else:
    BASE_MODEL_PATH = "openai-community/gpt2"
    BASE_RWD_MODEL_PATH = "rajkumarrrk/roberta-daily-dialog-intent-classifier"
    BERT_SCORE_MODEL_PATH = "microsoft/deberta-v2-xlarge-mnli"
# it should be noted that, we use this BASE_RWD_MODEL_PATH her since it is used in RL4LMs.
# however, we find it is underfitting to the intent labels in DailyDialog, so we will
# further fine-tune it, i.e., train_classifier.py, before using it as the reward model

intent_label2id = {
    "Inform": 0,
    "Questions": 1,
    "Directives": 2,
    "Commissive": 3
}

input_max_length = 78
output_max_length = 50

def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def truncate_generation_after_first_EOU(output):
    from dailydialog_dataset import DailyDialogDataset
    EOU = DailyDialogDataset.EOU_TOKEN
    if EOU not in output:
        return output
    else:
        return output[:output.index(EOU)] + EOU

def get_sft_dataset(usage="train"):
    from datasets import load_dataset, load_from_disk
    if offline_mode:
        dataset = load_from_disk(f"./daily_dialog/{usage}")
    else:
        dataset = load_dataset("daily_dialog")[usage]
    return dataset

def context_utterance_to_rwd_inputs(context, utterance, tokenizer):
    import torch
    context_ids = tokenizer(context, return_tensors="pt").input_ids
    utterance_ids = tokenizer(utterance, return_tensors="pt").input_ids
    input_ids = torch.cat([context_ids, utterance_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.cat([torch.zeros_like(context_ids),
                                torch.ones_like(utterance_ids)], dim=1)
    return input_ids, attention_mask, token_type_ids

def get_reward_fn(device):
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    rwd_model_path = auto_find_checkpoint_dir("classifier_fine_tuned/checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(rwd_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(rwd_model_path, device_map=device)
    def reward_fn(prompts, labels, intents=None, return_probs=False, return_scores=False, preferred_intent="consistency"):
        batch_inputs = []
        for context, utterance in zip(prompts, labels):
            inputs = context_utterance_to_rwd_inputs(context, utterance, tokenizer)
            batch_inputs.append(inputs)
        input_ids = pad_and_concat([batch_input[0] for batch_input in batch_inputs],
                                   pad_token_id=tokenizer.pad_token_id)
        attention_mask = pad_and_concat([batch_input[1] for batch_input in batch_inputs],
                                        pad_token_id=0)
        token_type_ids = pad_and_concat([batch_input[2] for batch_input in batch_inputs],
                                        pad_token_id=0)
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "token_type_ids": token_type_ids.to(device),
        }
        with torch.no_grad():
            outputs = model(**inputs)
            intents_probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        if return_probs:
            return intents_probs
        if preferred_intent == "consistency":
            assert intents is not None, "consistency preference on intent needs prompt intents specification"
            if return_scores:
                return [intents_probs[i, intents[i] - 1].item() for i in range(len(intents))]
            else:
                return (torch.argmax(intents_probs, dim=1).cpu().numpy() == np.array(intents) - 1).astype(np.int32).tolist()
        else:
            if type(preferred_intent) is str:
                preferred_intent = intent_label2id[preferred_intent]
            assert type(preferred_intent) is int, f"unrecognized preferred_intent: {preferred_intent}"
            if return_scores:
                return [intents_probs[i, preferred_intent].item() for i in range(intents_probs.shape[0])]
            else:
                return (torch.argmax(intents_probs, dim=1).cpu().numpy() == preferred_intent).astype(np.int32).tolist()
    return reward_fn

def get_trainer(args, model, tokenizer, train_dataset, valid_dataset, collate_fn, **kwargs):
    from transformers import TrainingArguments, Trainer
    import inspect
    gradient_accumulation_steps = args.global_batch_size // (args.n_devices * args.mini_batch_size)
    total_training_steps = args.epochs * len(train_dataset) // args.global_batch_size
    warmup_steps = int(total_training_steps * 0.1)
    eval_steps = int(total_training_steps * 0.1)
    save_steps = eval_steps

    deepspeed = None if args.no_deepspeed else {
        "train_micro_batch_size_per_gpu": args.mini_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": {
            "enabled": True,
            "min_loss_scale": 1,
            "opt_level": "O2"
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            }
        }
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.mini_batch_size,
        per_device_eval_batch_size=args.mini_batch_size,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=min(10, max(1, eval_steps // 10)),
        deepspeed=deepspeed,
        load_best_model_at_end=True,
        **{k:v for k,v in kwargs.items() if k in inspect.signature(TrainingArguments.__init__).parameters},
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        **{k:v for k,v in kwargs.items() if k in inspect.signature(Trainer.__init__).parameters}
    )
    return trainer

def auto_find_checkpoint_dir(path):
    if path is None or path == "None":
        return path
    def find_checkpoint_dir(path):
        probable_checkpoint_dirs = [dir for dir in os.listdir(path) if re.search(r"checkpoint-(\d+)", dir)]
        if len(probable_checkpoint_dirs) == 1:
            return probable_checkpoint_dirs[0]
        elif len(probable_checkpoint_dirs) > 1:
            print(f"multiple checkpoint dir in {path}, choose the last one")
            probable_checkpoint_dirs = sorted(probable_checkpoint_dirs, key=lambda dir:int(re.search(r"checkpoint-(\d+)", dir).group(1)))
            return probable_checkpoint_dirs[-1]
        else:
            raise Exception(f"no checkpoint under {path} is found")
    path_parts = os.path.normpath(path).split("/")
    for i, part in enumerate(path_parts):
        if part == "checkpoint":
            parent_path = "/".join(path_parts[:i])
            path_parts[i] = find_checkpoint_dir(parent_path)
    return "/".join(path_parts)

def load_checkpoint(path, device):
    import torch
    device = torch.device(device) if type(device) is str else device
    if os.path.exists(os.path.join(path, "pytorch_model.bin")):
        state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device)
    elif os.path.exists(os.path.join(path, "model.safetensors")):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(path, "model.safetensors"), device="cpu")
        state_dict = {k:v.to(device) for k,v in state_dict.items()}
    else:
        raise ValueError(f"no checkpoint file found in {path}")
    return state_dict

def pad_and_concat(list_tensors, pad_token_id, padding_side="right", dim=-1):
    import torch
    max_length = max([tensor.shape[dim] for tensor in list_tensors])
    list_padded_tensors = []
    for tensor in list_tensors:
        tensor_shape = list(tensor.shape)
        padding_shape = tensor_shape
        padding_shape[dim] = max_length - tensor_shape[dim]
        padding = torch.empty(size=padding_shape, dtype=tensor.dtype).fill_(pad_token_id)
        if padding_side == "right":
            list_padded_tensors.append(torch.cat([tensor, padding], dim=dim))
        elif padding_side == "left":
            list_padded_tensors.append(torch.cat([padding, tensor], dim=dim))
        else:
            raise NotImplementedError(padding_side)
    concated_tensors = torch.cat(list_padded_tensors, dim=0)
    return concated_tensors