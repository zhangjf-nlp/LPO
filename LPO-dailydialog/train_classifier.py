import os

import torch
import random
import argparse
import numpy as np

from config_and_utils import get_trainer, BASE_RWD_MODEL_PATH
from dailydialog_dataset import DailyDialogDataset

from transformers import AutoTokenizer, RobertaForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(BASE_RWD_MODEL_PATH)

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    acc = (logits.argmax(axis=-1) == labels)

    result = {
        "acc": acc.mean(),
    }

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for modeling and training')
    parser.add_argument('--no_deepspeed', action="store_true")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--n_devices', type=int, default=torch.cuda.device_count())
    parser.add_argument('--global_batch_size', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=8)

    parser.add_argument('--output_dir', type=str, default=None)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "classifier_fine_tuned"

    if not os.path.exists(args.output_dir):
        if args.local_rank == 0:
            os.mkdir(args.output_dir)
    elif os.path.exists(os.path.join(args.output_dir, "trainer_state.json")):
        print(f"output_dir is not empty, exit this training: {args.output_dir}")
        quit()

    set_seed(args.seed)

    train_dataset = DailyDialogDataset(usage="train")
    valid_dataset = DailyDialogDataset(usage="validation")

    model = RobertaForSequenceClassification.from_pretrained(
        BASE_RWD_MODEL_PATH,
        num_labels=4,
        type_vocab_size=3,
        ignore_mismatched_sizes=True
    )

    from transformers import TrainerCallback
    class EvalAtStartCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer = get_trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collate_fn=train_dataset.rwd_collate_fn,
        compute_metrics=compute_metrics,
        remove_unused_columns=False,
        callbacks=[EvalAtStartCallback()]
    )
    trainer.train()
    trainer.save_state()
