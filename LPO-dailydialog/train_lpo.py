import os

import torch
import random
import argparse
import numpy as np

from config_and_utils import get_tokenizer, get_trainer, auto_find_checkpoint_dir
from modeling_gpt2_cvae import LPOModel
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
    parser.add_argument('--cvae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--vae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument("--preference", type=str, default="consistency",
                        choices=["consistency", "Inform", "Questions", "Directives", "Commissive"])
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lpo_sampling_times', type=int, default=16)
    parser.add_argument('--lpo_post_sampling', type=int, default=1)
    parser.add_argument('--lpo_evidence_ratio', type=float, default=0.5)
    parser.add_argument('--expectation_lpo', type=int, default=0)
    parser.add_argument('--marginal_lpo', type=int, default=0)

    parser.add_argument('--no_deepspeed', action="store_true")

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=8)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    lpo_kwargs = {
        key:getattr(args, key)
        for key in [
            "beta", "lpo_sampling_times", "lpo_post_sampling", "lpo_evidence_ratio",
            "marginal_lpo", "expectation_lpo"
        ]
    }

    if args.cvae_model_path is not None:
        model = LPOModel.from_cvae(args.cvae_model_path, **lpo_kwargs)
        model_path = args.cvae_model_path
    elif args.vae_model_path is not None:
        model = LPOModel.from_vae(args.vae_model_path, **lpo_kwargs)
        model_path = args.vae_model_path
    else:
        raise ValueError(f"at least one of [cvae_model_path, vae_model_path] needs to be specified")

    if args.output_dir is None:
        args.output_dir = os.path.join(model_path, "exp_lpo" if args.expectation_lpo else "lpo")
        if not args.expectation_lpo:
            if args.lpo_post_sampling:
                args.output_dir += f"_post_sampling_{args.lpo_sampling_times}"
            else:
                args.output_dir += f"_sampling_{args.lpo_sampling_times}"
            args.output_dir += f"_evidence_{str(args.lpo_evidence_ratio).replace('.','')}"
        if args.marginal_lpo:
            args.output_dir += f"_mrg"
        if args.beta != 0.1:
            args.output_dir += f"_beta_{args.beta}"
        if args.learning_rate != 1e-4:
            args.output_dir += f"_lr_{args.learning_rate:.0e}"
        if args.epochs != 1:
            args.output_dir += f"_epochs_{args.epochs}"

    set_seed(args.seed)

    train_dataset = DailyDialogComparisonDataset(usage="train", preference=args.preference)
    valid_dataset = DailyDialogComparisonDataset(usage="validation", preference=args.preference)

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=train_dataset.lpo_collate_fn,
        compute_metrics=compute_metrics,
        label_names=["prompt_ids"],
        metric_for_best_model="eval_loss",
    )
    trainer.train()
    trainer.save_state()
