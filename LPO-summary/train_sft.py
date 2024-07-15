import json
import os

import torch
import random
import argparse
import numpy as np

from config_and_utils import get_tokenizer, get_trainer, BASE_MODEL_PATH
from modeling_gptj_cvae import GPTJForCausalLM, PTModel, GPTJCVAEConfig
from summary_dataset import SummaryDataset
from summary_one2many_dataset import GeneratedSummaryOne2ManyDataset, extend_one2many_dataset_with_mini_many


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

tokenizer = get_tokenizer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for modeling and training')
    parser.add_argument('--base_model_path', type=str, default=BASE_MODEL_PATH)
    parser.add_argument('--p_tuning', action="store_true")
    parser.add_argument('--p_tuning_on_generation', action="store_true")

    parser.add_argument('--no_deepspeed', action="store_true")

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=4)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    if args.output_dir is None:
        if args.p_tuning:
            if args.p_tuning_on_generation:
                args.output_dir = f"pt-sft-on-generation"
            else:
                args.output_dir = f"pt-sft"
        else:
            args.output_dir = f"sft"

    if args.local_rank == 0:
        print(f"args: {json.dumps(vars(args), ensure_ascii=False, indent=4)}")
        with open(f"{args.output_dir}/args.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=4))

    if not os.path.exists(args.output_dir):
        if args.local_rank == 0:
            os.mkdir(args.output_dir)
    elif os.path.exists(os.path.join(args.output_dir, "trainer_state.json")):
        print(f"output_dir is not empty, exit this training: {args.output_dir}")
        quit()

    set_seed(args.seed)

    if args.p_tuning:
        if args.p_tuning_on_generation:
            args.total_many, args.mini_many = 32, 4
            One2ManyDataset = extend_one2many_dataset_with_mini_many(
                GeneratedSummaryOne2ManyDataset, drop_other_mini=True
            )
            train_dataset = One2ManyDataset(
                total_many=args.total_many, mini_many=args.mini_many,
                usage="train"
            )
            valid_dataset = One2ManyDataset(
                total_many=args.total_many, mini_many=args.mini_many,
                usage="validation"
            )
            assert args.mini_batch_size == 1
            args.epochs *= args.total_many // args.mini_many
            args.global_batch_size *= args.total_many // args.mini_many
        else:
            train_dataset = SummaryDataset(usage="train")
            valid_dataset = SummaryDataset(usage="validation")

        sft_model = GPTJForCausalLM.from_pretrained(args.base_model_path,
                                                    device_map=f"cuda:{args.local_rank}")
        config = GPTJCVAEConfig(
            **vars(args),
            **vars(sft_model.config)
        )
        model = PTModel(config)
        model.load_pretrained(sft_model)
        if args.learning_rate == 1e-5:
            args.learning_rate = 1e-3
    else:
        train_dataset = SummaryDataset(usage="train")
        valid_dataset = SummaryDataset(usage="validation")
        model = GPTJForCausalLM.from_pretrained(args.base_model_path)

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=train_dataset.sft_collate_fn
    )
    trainer.train()
    trainer.save_state()
