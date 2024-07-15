import json
import os

import torch
import random
import argparse
import numpy as np
from transformers import GPT2LMHeadModel

from config_and_utils import get_tokenizer, get_trainer, BASE_MODEL_PATH, auto_find_checkpoint_dir, load_checkpoint
from modeling_gpt2_cvae import DPOModel, PTModel
from dailydialog_comparison_dataset import (
    DailyDialogComparisonDataset,
)

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def compute_metrics(eval_preds):
    acc = eval_preds.predictions
    return {"acc": acc.mean()}


tokenizer = get_tokenizer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for modeling and training')
    parser.add_argument('--sft_model_path', type=auto_find_checkpoint_dir,
                        default=auto_find_checkpoint_dir(f"sft/checkpoint"))
    parser.add_argument('--pt_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument("--preference", type=str, default="consistency",
                        choices=["consistency", "Inform", "Questions", "Directives", "Commissive"])
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lora_dim', type=int, default=0)

    parser.add_argument('--no_deepspeed', action="store_true")

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=2)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    if args.output_dir is None:
        if args.pt_model_path is not None:
            args.output_dir = os.path.join(args.pt_model_path, f"dpo_{args.preference}_preference_beta_{args.beta}")
        else:
            if args.lora_dim:
                args.output_dir = os.path.join(args.sft_model_path,
                                               f"lora-{args.lora_dim}_dpo_{args.preference}_preference_beta_{args.beta}")
            else:
                args.output_dir = os.path.join(args.sft_model_path,
                                               f"dpo_{args.preference}_preference_beta_{args.beta}")

    if not os.path.exists(args.output_dir):
        if args.local_rank == 0:
            os.mkdir(args.output_dir)
    elif os.path.exists(os.path.join(args.output_dir, "trainer_state.json")):
        print(f"output_dir is not empty, exit this training: {args.output_dir}")
        quit()

    set_seed(args.seed)

    train_dataset = DailyDialogComparisonDataset(usage="train", preference=args.preference)
    valid_dataset = DailyDialogComparisonDataset(usage="validation", preference=args.preference)

    device = f"cuda:{args.local_rank}"
    if args.pt_model_path is not None:
        model = PTModel.from_pretrained(args.pt_model_path, device_map=device)
        model.switch_into_dpo_mode()
        model.config.beta = args.beta
        model.beta = args.beta
        if args.learning_rate == 1e-6:
            args.learning_rate = 1e-4
    else:
        pretrained_model = GPT2LMHeadModel.from_pretrained(args.sft_model_path, device_map=device)
        config = pretrained_model.config
        config.update({
            "beta": args.beta, "lora_dim": args.lora_dim
        })
        model = DPOModel(config).to(device)
        model.load_pretrained(pretrained_model=pretrained_model)
        model.prepare_refer_model()

        if args.lora_dim:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config.lora_dim, lora_alpha=config.lora_dim, lora_dropout=0.0
            )
            model.base = get_peft_model(model.base, peft_config)
            if args.local_rank == 0:
                model.base.print_trainable_parameters()

            if args.learning_rate == 1e-6:
                args.learning_rate = 1e-5

    if args.local_rank == 0:
        print(f"args: {json.dumps(vars(args), ensure_ascii=False, indent=4)}")
        with open(f"{args.output_dir}/args.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=4))

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=train_dataset.dpo_collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_state()

    if args.lora_dim and args.local_rank == 0:
        model.base.save_pretrained(os.path.join(args.output_dir, "best_model"))