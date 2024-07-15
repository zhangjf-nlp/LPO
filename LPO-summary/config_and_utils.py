import os
import re

offline_mode = True
# this mode requires datasets and base models are pre-downloaded and put into this project
# our experiments are all carried in this mode since our machines are not connected to Internet
#   offline_mode = False
# this mode has not been tested yet

if offline_mode:
    BASE_MODEL_PATH = "./gpt-j-6b"
    RWD_MODEL_PATH = "./openai_summarize_tldr_rwd"
else:
    BASE_MODEL_PATH = "EleutherAI/gpt-j-6b"
    RWD_MODEL_PATH = "CarperAI/openai_summarize_tldr_rwd"

input_max_length = 500
output_max_length = 50

def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_sft_dataset():
    from datasets import load_dataset, load_from_disk
    if offline_mode:
        dataset = load_from_disk("./openai_summarize_tldr")
    else:
        dataset = load_dataset("CarperAI/openai_summarize_tldr")
    return dataset

def get_comparison_dataset():
    from datasets import load_dataset, load_from_disk
    if offline_mode:
        dataset = load_from_disk("./openai_summarize_comparisons")
    else:
        dataset = load_dataset("CarperAI/openai_summarize_comparisons")
    return dataset

def get_trainer(args, model, tokenizer, train_dataset, valid_dataset, collate_fn,
                early_eval=False, **kwargs):
    from transformers import Trainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
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

    class EarlyEvaluationCallback(TrainerCallback):
        def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True
            else:
                control.should_training_stop = True

    callbacks = []
    if early_eval:
        callbacks.append(EarlyEvaluationCallback())

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
        **{k:v for k,v in kwargs.items() if k in inspect.signature(Trainer.__init__).parameters}
    )
    return trainer

def auto_find_checkpoint_dir(path):
    if path is None or path == "None":
        return None
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