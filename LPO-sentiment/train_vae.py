import os
import json
import torch

import random
import argparse
import numpy as np

from modeling_gpt2_cvae import GPT2LMHeadModel, GPT2ForVAE, GPT2CVAEConfig
from imdb_one2many_dataset import IMDBOne2ManyDataset
from config_and_utils import get_tokenizer, get_trainer, auto_find_checkpoint_dir, BASE_MODEL_PATH


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
    parser.add_argument('--base_model_path', type=auto_find_checkpoint_dir, default=BASE_MODEL_PATH)

    parser.add_argument('--num_q', type=int, default=4)
    parser.add_argument('--dim_z', type=int, default=32)
    parser.add_argument('--num_p', type=int, default=4)
    parser.add_argument('--latent_aggregator_layers', type=int, default=2)
    parser.add_argument('--post_with_x', type=int, default=1)
    parser.add_argument('--many', type=int, default=32)

    parser.add_argument('--frozen_pretrained', type=int, default=0)
    parser.add_argument('--marginal_kl', type=int, default=1)
    parser.add_argument('--lm_sampling_times', type=int, default=1)
    parser.add_argument('--kl_sampling_times', type=int, default=16)
    parser.add_argument('--without_dg_kld', type=int, default=0)
    parser.add_argument('--add_skip_connection', type=int, default=0)
    parser.add_argument('--add_contra_loss', type=int, default=0)

    parser.add_argument('--no_deepspeed', action="store_true")

    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=8)
    parser.add_argument('--mini_batch_size', type=int, default=1)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    set_seed(args.seed)

    train_dataset = IMDBOne2ManyDataset(usage="train", many=args.many)
    valid_dataset = IMDBOne2ManyDataset(usage="validation", many=args.many)

    if args.vae_model_path is None:
        # construct with base model
        sft_model = GPT2LMHeadModel.from_pretrained(args.base_model_path,
                                                    device_map=f"cuda:{args.local_rank}")
        config = GPT2CVAEConfig(
            **vars(args),
            **vars(sft_model.config)
        )
        config.update({
            "attn_pdrop": 0.0, "embd_pdrop": 0.0, "resid_pdrop": 0.0,
        })
        model = GPT2ForVAE(config)
        model.load_pretrained(sft_model)
    else:
        model = GPT2ForVAE.from_pretrained(args.vae_model_path)
        model.config.update({
            key: getattr(args, key) for key in [
                "frozen_pretrained", "marginal_kl", "without_dg_kld",
                "add_skip_connection", "add_contra_loss"
            ]
        })
        model.update_requires_grad_()

    if args.output_dir is None:
        args.output_dir = f"gpt2_vae" if args.lm_sampling_times > 0 else f"gpt2_ae"
        args.output_dir += f"_{args.latent_aggregator_layers}-layers_"
        if (args.num_q, args.dim_z, args.num_p) != (4, 32, 4):
            args.output_dir += f"_{args.num_q}q_{args.dim_z}z_{args.num_p}p"
        if args.frozen_pretrained:
            args.output_dir += f"_frozen_base"
        if args.many != 32:
            args.output_dir += f"_1-to-{args.many}"
        if args.with_bn:
            args.output_dir += f"_with_bn"
        if args.post_with_x:
            args.output_dir += f"_post_with_x"
        if args.add_skip_residue_contra:
            args.output_dir += f"_skip_residue_based_contra"
        else:
            if args.add_skip_connection:
                args.output_dir += f"_skip"
            if args.add_contra_loss:
                args.output_dir += f"_contra"
        if args.without_dg_kld:
            args.output_dir += "_w.o.dg_kld"
        if not args.learning_rate == 1e-4:
            args.output_dir += f"_lr{args.learning_rate}"
        if not args.epochs == 1:
            args.output_dir += f"_{args.epochs}epochs"
    tokenizer = get_tokenizer()

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

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=train_dataset.vae_collate_fn,
        compute_metrics=compute_metrics,
        label_names=[],
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    trainer.train()
    trainer.save_state()

