import os
import json

import torch
import argparse
import numpy as np
from tqdm import tqdm

from gpu_map_reduce import BatchTaskConsumer, TaskProducer
from modeling_gptj_cvae import GPTJConfig, GPTJForCVAE, GPTJForCausalLM, PTModel, GPTJCVAEConfig, GPTJForVAE
from config_and_utils import (
    auto_find_checkpoint_dir,
    BASE_MODEL_PATH,
    RWD_MODEL_PATH,
    get_tokenizer,
    pad_and_concat,
    input_max_length,
    output_max_length,
)


class PPLProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        with open(self.args.input_file, "r", encoding="utf-8") as f:
            tasks = json.loads(f.read())
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return all([result is not None and task[key]==result[key]
                    for key in ["prompt", "output", "rewards"]])

class PPLConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        self.tokenizer = get_tokenizer()
        self.model = GPTJForCausalLM.from_pretrained(
            BASE_MODEL_PATH, device_map=self.device,
        ).eval()

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        input_sentences = [task["prompt"] + task["output"] for task in tasks]
        inputs = self.tokenizer(input_sentences, return_tensors="pt", padding="longest")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            lm_logits = self.model(**inputs).logits
            lm_logits = lm_logits[:,:-1,:].contiguous()
            labels = inputs["input_ids"][:,1:].contiguous()
            labels = torch.where(labels==self.tokenizer.pad_token_id, -100, labels)
            nlls = torch.nn.functional.cross_entropy(
                input=lm_logits.transpose(1,2),
                target=labels,
                reduction='none'
            ).sum(dim=-1).tolist()
            num_tokens = (labels!=-100).sum(dim=-1).tolist()
        for i,task in enumerate(tasks):
            task["nll"] = nlls[i]
            task["num_tokens"] = num_tokens[i]
        return tasks

class RewardProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        with open(self.args.input_file, "r", encoding="utf-8") as f:
            tasks = json.loads(f.read())
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return all([result is not None and task[key]==result[key]
                    for key in ["prompt", "output"]])

class RewardConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        from summary_dataset import SummaryDataset
        from reward_model import GPTRewardModel
        self.tokenizer = get_tokenizer()
        self.model = GPTRewardModel(BASE_MODEL_PATH, device=self.device)
        self.model.load_state_dict(torch.load(os.path.join(RWD_MODEL_PATH, "pytorch_model.bin"),
                                              map_location=self.device), strict=False)
        self.model = self.model.half().eval()
        self.post_summary_dict = {data["prompt"].split("TL;DR:")[0] + "TL;DR: ":data["label"]
                                  for data in SummaryDataset(usage="test").dataset}

    def get_scores(self, samples):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i: i + batch_size]
            sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
            encodings_dict = self.tokenizer(
                sub_samples,
                truncation=True,
                max_length=input_max_length+output_max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(self.device)
            attn_masks = encodings_dict["attention_mask"].to(self.device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = self.model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores

    def reward_fn(self, samples):
        original_samples = [text.split("TL;DR:")[0] + "TL;DR: " for text in samples]
        original_samples = [text + self.post_summary_dict[text] for text in original_samples]
        original_scores = self.get_scores(original_samples)
        scores = self.get_scores(samples)
        norms_scores = scores - original_scores
        return norms_scores

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        samples = [task["prompt"] + task["output"] for task in tasks]
        with torch.no_grad():
            rewards = self.reward_fn(samples).tolist()
        for i,task in enumerate(tasks):
            task["rewards"] = rewards[i]
        return tasks

class GenerateProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        from summary_dataset import SummaryDataset
        tasks = [{"prompt": data["prompt"].split("TL;DR:")[0] + "TL;DR: "}
                 for data in SummaryDataset(usage="test").dataset]
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return all([result is not None and task[key]==result[key]
                    for key in ["prompt"]])

class GenerateConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        tokenizer = get_tokenizer()
        if self.args.cvae_model_path is not None:
            model = GPTJForCVAE.from_pretrained(
                self.args.cvae_model_path,
                torch_dtype=torch.float16,
                device_map=self.device
            ).eval()
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            if self.args.lpo_model_path is not None:
                model.load_lpo_policy_latent_encoder(self.args.lpo_model_path)
        elif self.args.vae_model_path is not None:
            model = GPTJForVAE.from_pretrained(
                self.args.vae_model_path,
                torch_dtype=torch.float16,
                device_map=self.device
            ).eval()
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            if self.args.lpo_model_path is not None:
                model.load_lpo_policy_latent_encoder(self.args.lpo_model_path)
        elif self.args.dpo_model_path is not None:
            model = GPTJForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map=self.device,
            ).eval()
            sd = torch.load(os.path.join(self.args.dpo_model_path, "pytorch_model.bin"),
                            map_location=self.device)
            sd = {k.replace("base.", ""): v for k, v in sd.items() if k.startswith("base.")}
            model.load_state_dict(sd)
        elif self.args.pt_model_path is not None:
            model = PTModel.from_pretrained(
                self.args.pt_model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
            ).eval()
        elif self.args.ppo_model_path is not None:
            model = GPTJForCausalLM(GPTJConfig.from_pretrained(BASE_MODEL_PATH))
            num_shards = 4 # TODO: read this number from ppo_model_path
            sd = {}
            for i in tqdm(range(num_shards), desc="Loading PPO checkpoint shards"):
                sd_shard = torch.load(os.path.join(self.args.ppo_model_path,
                                                   f"pytorch_model-{i+1:05d}-of-{num_shards:05d}.bin"),
                                      map_location="cpu")
                sd.update(sd_shard)
            sd = {k[len("base_model."):]: v for k, v in sd.items() if k.startswith("base_model.")}
            model.load_state_dict(sd)
            model = model.half().to(self.device).eval()
        elif self.args.lora_model_path is not None:
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(self.args.lora_model_path)
            model = model.eval().half().to(self.device)
        else:
            model = GPTJForCausalLM.from_pretrained(
                self.args.sft_model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
            ).eval()
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        self.model = model

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        inputs = self.tokenizer([task["prompt"] for task in tasks], return_tensors="pt",
                                truncation=True, padding="longest", max_length=input_max_length)
        inputs_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids.to(self.device),
                attention_mask=inputs.attention_mask.to(self.device),
                do_sample=True,
                temperature=0.01,
                max_new_tokens=output_max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        outputs = self.tokenizer.batch_decode(output_ids[:,inputs_len:], skip_special_tokens=True)
        for task,output in zip(tasks,outputs):
            task['output'] = output
        return tasks

def visualize_reward_and_ppl(input_file):
    output_dir = os.path.dirname(input_file)
    def visualize_values(values, x_min, x_max, x_num, window_size, name):
        import matplotlib.pyplot as plt
        x = np.linspace(x_min, x_max, x_num)
        x_left, x_right = x - window_size, x + window_size
        x_left = np.where(x_left<x_min, x_min, x_left)
        x_right = np.where(x_right>x_max, x_max, x_right)
        y = ((x_left[:,None]<values[None,:]) & (values[None,:]<x_right[:,None])).sum(axis=-1) / (x_right-x_left)
        records_file = os.path.join(output_dir, f"{name}_density.json")
        with open(records_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"x":x.tolist(), "y":y.tolist()}))
        plot_png_file = os.path.join(output_dir, f"{name}_density.png")
        plt.plot(x, y)
        plt.xlim(x_min, x_max)
        plt.ylim(bottom=0.0)
        plt.savefig(plot_png_file)
        plt.clf()
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.loads(f.read())
    rewards = np.array([result["rewards"] for result in results])
    visualize_values(rewards, x_min=-6.0, x_max=6.0, x_num=100, window_size=0.3, name=f"rewards")
    print(f"reward mean: {rewards.mean():.2f}")
    print(f"reward standard deviation: {rewards.std():.2f}")
    total_nll = sum([result["nll"] for result in results])
    total_num_tokens = sum([result["num_tokens"] for result in results])
    ppl = np.exp(total_nll / total_num_tokens)
    print(f"ppl: {ppl:.2f}")

def parse_args():
    parser = argparse.ArgumentParser(description='parse args for map-reduce generation and reward on multi-gpu')
    parser.add_argument("--py_file_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6380)
    parser.add_argument("--task_name", type=str, default="generate")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--erase", type=int, default=0)

    # checkpoint path of generation model to test
    parser.add_argument('--sft_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--pt_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--dpo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--cvae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--vae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--lpo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--ppo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--lora_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument("--batch_size", type=int, default=8)

    # the input file of generation to reward
    parser.add_argument("--input_file", type=str, default=None)

    args = parser.parse_args()

    if args.py_file_name is None:
        args.py_file_name = os.path.basename(__file__)

    if args.local_rank == -1:
        if args.pt_model_path is not None:
            args.output_file = os.path.join(args.pt_model_path, "inference.json")
        elif args.dpo_model_path is not None:
            args.output_file = os.path.join(args.dpo_model_path, "inference.json")
        elif args.cvae_model_path is not None:
            if args.lpo_model_path is not None:
                args.output_file = os.path.join(args.lpo_model_path, "inference.json")
            else:
                args.output_file = os.path.join(args.cvae_model_path, "inference.json")
        elif args.vae_model_path is not None:
            if args.lpo_model_path is not None:
                args.output_file = os.path.join(args.lpo_model_path, "inference.json")
            else:
                args.output_file = os.path.join(args.vae_model_path, "inference.json")
        elif args.sft_model_path is not None:
            args.output_file = os.path.join(args.sft_model_path, "inference.json")
        elif args.ppo_model_path is not None:
            args.output_file = os.path.join(args.ppo_model_path, "inference.json")
        elif args.lora_model_path is not None:
            args.output_file = os.path.join(args.lora_model_path, "inference.json")
        else:
            raise NotImplementedError(f"at least one kind of model path should be specified")

    return args

def main():
    args = parse_args()

    if args.local_rank == -1:
        # step1. generate output
        producer = GenerateProducer(args)
        producer.run()
        # step2. reward output
        args.task_name = "reward"
        args.input_file = args.output_file
        args.output_file = args.output_file.replace("inference.json", "inference_and_reward.json")
        producer = RewardProducer(args)
        producer.run()
        # step3. ppl
        args.task_name = "ppl"
        args.input_file = args.output_file
        args.output_file = args.output_file.replace("inference_and_reward.json", "inference_and_reward_and_ppl.json")
        producer = PPLProducer(args)
        producer.run()
        # step4. visualize output reward and ppl
        visualize_reward_and_ppl(input_file=args.output_file)
    else:
        if args.task_name == "generate":
            consumer = GenerateConsumer(args)
            consumer.run()
        elif args.task_name == "reward":
            consumer = RewardConsumer(args)
            consumer.run()
        elif args.task_name == "ppl":
            consumer = PPLConsumer(args)
            consumer.run()
        else:
            raise NotImplementedError(args.task_name)

if __name__ == "__main__":
    main()
