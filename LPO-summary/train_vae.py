import os
import json
import torch

import random
import argparse
import numpy as np

from modeling_gptj_cvae import GPTJForVAE, GPTJCVAEConfig

from summary_one2many_dataset import (
    SummaryOne2ManyDataset,
    GeneratedSummaryOne2ManyDataset, extend_one2many_dataset_with_mini_many
)
from transformers import (
    AutoConfig,
)

from config_and_utils import (
    get_tokenizer,
    get_trainer,
    BASE_MODEL_PATH,
    auto_find_checkpoint_dir
)


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def compute_metrics(eval_preds):
    loss_lm, loss_kld, loss_contra = eval_preds.predictions

    result = {
        "loss_lm": loss_lm.mean(),
        "loss_kld": loss_kld.mean(),
        "loss_contra": loss_contra.mean(),
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for modeling and training')

    parser.add_argument('--vae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--base_model_path', default=BASE_MODEL_PATH)

    parser.add_argument('--num_q', type=int, default=4)
    parser.add_argument('--dim_z', type=int, default=32)
    parser.add_argument('--num_p', type=int, default=4)
    parser.add_argument('--latent_aggregator_layers', type=int, default=2)

    parser.add_argument('--frozen_pretrained', type=int, default=0)
    parser.add_argument('--marginal_kl', type=int, default=1)
    parser.add_argument('--lm_sampling_times', type=int, default=1)
    parser.add_argument('--kl_sampling_times', type=int, default=16)
    parser.add_argument('--add_skip_connection', type=int, default=0)
    parser.add_argument('--add_contra_loss', type=int, default=0)
    parser.add_argument('--without_dg_kld', type=int, default=0)

    parser.add_argument('--no_deepspeed', action="store_true")

    parser.add_argument('--data_from', type=str, default="openai",
                        choices=["openai", "generation"])

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=8)
    parser.add_argument('--mini_batch_size', type=int, default=1)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = f"gptj_vae"
        if (args.num_q, args.dim_z, args.num_p) != (4, 32, 4):
            args.output_dir += f"_{args.num_q}q_{args.dim_z}z_{args.num_p}p"
        if args.frozen_pretrained:
            args.output_dir += f"_frozen_base"
        if args.add_skip_connection:
            args.output_dir += f"_skip"
        if args.add_contra_loss:
            args.output_dir += f"_contra"
        if not args.learning_rate == 1e-5:
            args.output_dir += f"_lr{args.learning_rate}"
        if not args.epochs == 1:
            args.output_dir += f"_{args.epochs}epochs"
    tokenizer = get_tokenizer()

    if not os.path.exists(args.output_dir):
        if args.local_rank == 0:
            os.makedirs(args.output_dir)
            print(f"args: {json.dumps(vars(args), ensure_ascii=False, indent=4)}")
            with open(f"{args.output_dir}/args.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(vars(args), ensure_ascii=False, indent=4))
    elif os.path.exists(os.path.join(args.output_dir, "trainer_state.json")):
        print(f"output_dir is not empty, exit this training: {args.output_dir}")
        quit()

    if args.local_rank == 0:
        print(f"args: {json.dumps(vars(args), ensure_ascii=False, indent=4)}")
        with open(f"{args.output_dir}/args.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=4))

    if args.data_from == "openai":
        args.total_many, args.mini_many = 4, 4
        One2ManyDataset = extend_one2many_dataset_with_mini_many(SummaryOne2ManyDataset)
    elif args.data_from == "generation":
        args.total_many, args.mini_many = 32, 4
        One2ManyDataset = extend_one2many_dataset_with_mini_many(GeneratedSummaryOne2ManyDataset)
    else:
        raise NotImplementedError(args.data_from)
    train_dataset = One2ManyDataset(
        total_many=args.total_many, mini_many=args.mini_many,
        usage="train"
    )
    valid_dataset = One2ManyDataset(
        total_many=args.total_many, mini_many=args.mini_many,
        usage="validation"
    )

    training_kwargs = {
        key:getattr(args, key) for key in [
            "frozen_pretrained", "marginal_kl", "lm_sampling_times", "kl_sampling_times",
            "add_skip_connection", "add_contra_loss", "without_dg_kld"
        ]
    }

    if args.vae_model_path is not None:
        model = GPTJForVAE.from_pretrained(args.vae_model_path)
        model.config.update(training_kwargs)
        model.update_requires_grad_()
    else:
        # construct with base model
        gptj_config = AutoConfig.from_pretrained(args.base_model_path)
        gptj_cvae_config = GPTJCVAEConfig(
            **vars(args),
            **vars(gptj_config)
        )
        model = GPTJForVAE(gptj_cvae_config)
        model.load_pretrained()

    args.epochs *= args.total_many // args.mini_many
    args.global_batch_size *= args.total_many // args.mini_many

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=train_dataset.collate_fn,
        compute_metrics=compute_metrics,
        label_names=[],
    )
    trainer.train()
    trainer.save_state()

