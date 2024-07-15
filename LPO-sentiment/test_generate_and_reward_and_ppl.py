import os
import json

import torch
import argparse
import numpy as np
from tqdm import tqdm

from gpu_map_reduce import BatchTaskConsumer, TaskConsumer, TaskProducer
from config_and_utils import (
    get_tokenizer, BASE_MODEL_PATH, load_checkpoint,
    input_max_length, output_max_length, pad_and_concat,
)
from modeling_gpt2_cvae import (
    GPT2Config, GPT2ForVAE,
    GPT2LMHeadModel, PTModel, DPOModel
)

class PPLProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        with open(self.args.input_file, "r", encoding="utf-8") as f:
            tasks = json.loads(f.read())
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None \
               and all([(key in result and task[key]==result[key])
                        for key in ["prompt", "output", "reference", "logits"]]) \
               and all([key in result
                        for key in ["logits", "logits_along_length", "nll", "num_tokens",
                                    "more_positive", "more_negative", "more_neutral"]])

class PPLConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        from transformers import GPT2LMHeadModel
        from config_and_utils import BASE_MODEL_PATH, get_tokenizer
        self.tokenizer = get_tokenizer()
        self.model = GPT2LMHeadModel.from_pretrained(
            BASE_MODEL_PATH, device_map=self.device,
        ).eval()

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        input_sentences = [task["prompt"][0] + task["output"][0] for task in tasks]
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
        return result is not None \
               and all([(key in result and task[key]==result[key])
                        for key in ["prompt", "output", "reference"]]) \
               and all([key in result
                        for key in ["logits", "logits_along_length",
                                    "more_positive", "more_negative", "more_neutral"]])

class RewardConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from config_and_utils import RWD_MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(RWD_MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RWD_MODEL_PATH, device_map=self.device,
        ).eval()

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        input_sentences = [task["prompt"][i] + task["output"][i]
                           for i in range(self.args.best_of_N)
                           for task in tasks]
        input_references = [task["reference"] for task in tasks]
        inputs = self.tokenizer(input_sentences+input_references, return_tensors="pt",
                                padding="longest", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # for visualization of reward along length
        logits_along_length = []
        min_length, max_length, stride_length = 8, 112, 8
        for length in range(min_length, max_length+1, stride_length):
            with torch.no_grad():
                outputs = self.model(**{k:v[:,:length] for k,v in inputs.items()})
                logits = outputs.logits[:, 1] - outputs.logits[:, 0]
                logits_along_length.append(logits)
        logits_along_length = torch.stack(logits_along_length, dim=1)
        for i,task in enumerate(tasks):
            task["logits_along_length"] = logits_along_length[i, :].tolist()
            task["logits_along_length_reference"] = logits_along_length[i-len(tasks), :].tolist()

        with torch.no_grad():
            outputs = self.model(**inputs)
        assert outputs.logits.shape[0] == len(tasks) * (1 + self.args.best_of_N)
        logits = outputs.logits[:, 1] - outputs.logits[:, 0]
        sentences_logits = logits[:len(tasks)*self.args.best_of_N].view(self.args.best_of_N, len(tasks))
        most_positive_of_N_logits = sentences_logits.max(dim=0).values
        most_negative_of_N_logits = (-sentences_logits).max(dim=0).values
        most_neutral_of_N_logits = (-torch.abs(sentences_logits)).max(dim=0).values
        references_logits = logits[-len(tasks):]
        for i,task in enumerate(tasks):
            task["logits"] = sentences_logits[0, i].item()
            task["more_positive"] = (most_positive_of_N_logits[i] > references_logits[i]).item()
            task["more_negative"] = (most_negative_of_N_logits[i] > -references_logits[i]).item()
            task["more_neutral"] = (most_neutral_of_N_logits[i] > -torch.abs(references_logits[i])).item()
        return tasks

class GenerateProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        from imdb_one2many_dataset import IMDBDatasetForInference
        test_dataset = IMDBDatasetForInference(usage="test")
        tasks = [
            {"input_ids":inputs["input_ids"].tolist(),
             "reference": inputs["reference"]}
            for inputs in test_dataset
        ]
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None and all([(key in result and task[key]==result[key])
                                           for key in ["input_ids", "reference"]])

class GenerateConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        tokenizer = get_tokenizer()
        if self.args.vae_model_path is not None:
            model = GPT2ForVAE.from_pretrained(self.args.vae_model_path)
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            if self.args.lpo_model_path is not None:
                model.load_lpo_policy_latent_encoder(self.args.lpo_model_path)
            model = model.half().to(self.device).eval()
        elif self.args.dpo_model_path is not None:
            model = DPOModel.from_pretrained(self.args.dpo_model_path, device_map=self.device)
            model = model.base
        elif self.args.lora_model_path is not None:
            from peft import AutoPeftModelForCausalLM
            model = AutoPeftModelForCausalLM.from_pretrained(self.args.lora_model_path)
            model = model.eval().half().to(self.device)
        elif self.args.pt_model_path is not None:
            model = PTModel.from_pretrained(self.args.pt_model_path, device_map=self.device)
            model = model.half().to(self.device).eval()
        elif self.args.ppo_model_path is not None:
            model = GPT2LMHeadModel(GPT2Config.from_pretrained(BASE_MODEL_PATH))
            sd = load_checkpoint(self.args.ppo_model_path, device=self.device)
            sd = {k[len("base_model."):]: v for k, v in sd.items() if k.startswith("base_model.")}
            model.load_state_dict(sd)
            model = model.half().to(self.device).eval()
        else:
            model = GPT2LMHeadModel.from_pretrained(
                self.args.sft_model_path,
                device_map=self.device,
                torch_dtype=torch.float16,
            ).eval()
        self.tokenizer = tokenizer
        self.model = model

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        inputs_len = max([len(task["input_ids"][0]) for task in tasks])
        input_ids = pad_and_concat(
            [torch.LongTensor(task["input_ids"]) for task in tasks],
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left"
        ).to(self.device)
        assert input_ids.shape[1] == inputs_len
        for task in tasks:
            task['output'] = []
            task['prompt'] = []
        for i in range(self.args.best_of_N):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    min_length=input_max_length + output_max_length,
                    max_new_tokens=output_max_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            outputs = self.tokenizer.batch_decode(output_ids[:,inputs_len:], skip_special_tokens=True)
            prompts = self.tokenizer.batch_decode(output_ids[:,:inputs_len], skip_special_tokens=True)
            for task,output,prompt in zip(tasks,outputs,prompts):
                task['output'].append(output)
                task['prompt'].append(prompt)
        return tasks

def visualize_reward_and_ppl(input_file):
    import matplotlib.pyplot as plt
    output_dir = os.path.dirname(input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.loads(f.read())
    # global reward distribution
    def visualize_values(values, x_min, x_max, x_num, window_size, name):
        x = np.linspace(x_min, x_max, x_num)
        x_left, x_right = x - window_size, x + window_size
        x_left = np.where(x_left<x_min, x_min, x_left)
        x_right = np.where(x_right>x_max, x_max, x_right)
        y = ((x_left[:,None]<values[None,:]) & (values[None,:]<x_right[:,None])).sum(axis=-1) / (x_right-x_left)
        records_file = os.path.join(output_dir,f"{name}_density.json")
        with open(records_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"x":x.tolist(), "y":y.tolist()}))
        plot_png_file = os.path.join(output_dir,f"{name}_density.png")
        plt.plot(x, y)
        plt.xlim(x_min, x_max)
        plt.ylim(bottom=0.0)
        plt.savefig(plot_png_file)
        plt.clf()
    logits = np.array([result["logits"] for result in results])
    visualize_values(logits, x_min=-6.0, x_max=6.0, x_num=100, window_size=0.3, name="reward_logits")
    probs = 1 / (1 + np.exp(-logits))
    visualize_values(probs, x_min=0.0, x_max=1.0, x_num=100, window_size=0.01, name="positive_probs")
    print(f"positive expectation: {probs.mean()*100:.2f}%")
    print(f"positive standard deviation: {probs.std()*100:.2f}%")
    print(f"positive case proportion: {(logits>0).mean()*100:.2f}%")
    for preference in ["positive", "negative", "neutral"]:
        print(f"{preference} win-rate: {np.mean([result['more_'+preference] for result in results])*100:.2f}%")

    # reward along lengths
    logits_along_length = np.array([result["logits_along_length"]
                                    for result in results])
    logits_along_length_reference = np.array([result["logits_along_length_reference"]
                                              for result in results])
    min_length, max_length, stride_length = 8, 112, 8
    lengths = list(range(min_length, max_length + 1, stride_length))
    p_bottom, p_top = 25, 75
    plt.plot(lengths, logits_along_length_reference.mean(axis=0), color="blue")
    plt.fill_between(lengths,
                     np.percentile(logits_along_length_reference, p_bottom, axis=0),
                     np.percentile(logits_along_length_reference, p_top, axis=0),
                     color='blue', alpha=0.1)
    plt.plot(lengths, logits_along_length.mean(axis=0), color="red")
    plt.fill_between(lengths,
                     np.percentile(logits_along_length, p_bottom, axis=0),
                     np.percentile(logits_along_length, p_top, axis=0),
                     color='red', alpha=0.1)
    plt.xlim(min_length, max_length)
    plt.ylim(-6.0, 6.0)
    plt.savefig(os.path.join(output_dir, f"logits_along_length.png"))
    plt.clf()

    total_nll = sum([result["nll"] for result in results])
    total_num_tokens = sum([result["num_tokens"] for result in results])
    ppl = np.exp(total_nll / total_num_tokens)
    print(f"ppl: {ppl:.2f}")
    return probs.mean(), ppl

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
    from config_and_utils import auto_find_checkpoint_dir
    parser.add_argument('--sft_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--pt_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--dpo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--vae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--cvae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--lpo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--ppo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--lora_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--best_of_N', type=int, default=1)
    parser.add_argument('--num_run_times', type=int, default=1)

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
    args.erase = int(args.erase or args.num_run_times > 1)

    if args.local_rank == -1:
        positive_scores, ppl_scores = [], []
        input_file, output_file = args.input_file, args.output_file
        for i in range(args.num_run_times):
            print(f"run - {i+1}/{args.num_run_times}")
            # step1. generate output
            args.task_name = "generate"
            args.input_file, args.output_file = input_file, output_file
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
            positive_score, ppl_score = visualize_reward_and_ppl(input_file=args.output_file)
            positive_scores.append(positive_score)
            ppl_scores.append(ppl_score)
        print(f"positive score: {np.mean(positive_scores):.3f}±{np.std(positive_scores):.3f}")
        print(f"ppl score: {np.mean(ppl_scores):.2f}±{np.std(ppl_scores):.2f}")
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
