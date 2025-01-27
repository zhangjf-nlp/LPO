import json
import os

import torch
import random
import argparse
import numpy as np

from config_and_utils import get_tokenizer, get_trainer, BASE_MODEL_PATH, auto_find_checkpoint_dir
from modeling_gpt2_cvae import GPT2LMHeadModel, PTModel, GPT2CVAEConfig
from dailydialog_dataset import DailyDialogDataset
from dailydialog_one2many_dataset import DailyDialogOne2ManyDataset

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

tokenizer = get_tokenizer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for modeling and training')
    parser.add_argument('--base_model_path', type=auto_find_checkpoint_dir, default=BASE_MODEL_PATH)
    parser.add_argument('--p_tuning', action="store_true")
    parser.add_argument('--p_tuning_on_generation', action="store_true")

    parser.add_argument('--no_deepspeed', action="store_true")

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=8)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    if args.output_dir is None:
        if args.p_tuning:
            args.output_dir = f"pt-sft"
        elif args.p_tuning_on_generation:
            args.output_dir = f"pt-sft-on-generation"
        else:
            args.output_dir = f"sft"

    if not os.path.exists(args.output_dir):
        if args.local_rank == 0:
            os.makedirs(args.output_dir)
    elif os.path.exists(os.path.join(args.output_dir, "trainer_state.json")):
        print(f"output_dir is not empty, exit this training: {args.output_dir}")
        quit()

    if args.local_rank == 0:
        print(f"args: {json.dumps(vars(args), ensure_ascii=False, indent=4)}")
        with open(f"{args.output_dir}/args.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=4))

    set_seed(args.seed)

    if args.p_tuning:
        train_dataset = DailyDialogDataset(usage="train")
        valid_dataset = DailyDialogDataset(usage="validation")
        sft_model = GPT2LMHeadModel.from_pretrained(args.base_model_path,
                                                    device_map=f"cuda:{args.local_rank}")
        config = GPT2CVAEConfig(
            **vars(args),
            **vars(sft_model.config)
        )
        config.update({
            "attn_pdrop": 0.0, "embd_pdrop": 0.0, "resid_pdrop": 0.0,
        })
        model = PTModel(config)
        model.load_pretrained(sft_model)
    elif args.p_tuning_on_generation:
        train_dataset = DailyDialogOne2ManyDataset(usage="train")
        valid_dataset = DailyDialogOne2ManyDataset(usage="validation")
        sft_model = GPT2LMHeadModel.from_pretrained(args.base_model_path,
                                                    device_map=f"cuda:{args.local_rank}")
        config = GPT2CVAEConfig(
            **vars(args),
            **vars(sft_model.config)
        )
        config.update({
            "attn_pdrop": 0.0, "embd_pdrop": 0.0, "resid_pdrop": 0.0,
        })
        model = PTModel(config)
        model.load_pretrained(sft_model)
    else:
        train_dataset = DailyDialogDataset(usage="train")
        valid_dataset = DailyDialogDataset(usage="validation")
        model = GPT2LMHeadModel.from_pretrained(args.base_model_path)

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        remove_unused_columns=False,
        collate_fn=train_dataset.sft_collate_fn
    )
    trainer.train()
    trainer.save_state()
