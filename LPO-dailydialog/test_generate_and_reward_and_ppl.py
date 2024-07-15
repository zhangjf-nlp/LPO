import os
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd

from dailydialog_dataset import DailyDialogDataset
from dailydialog_one2many_dataset import DailyDialogOne2ManyDataset
from gpu_map_reduce import BatchTaskConsumer, TaskConsumer, TaskProducer
from config_and_utils import (
    BASE_MODEL_PATH, output_max_length, get_tokenizer,
    get_reward_fn, intent_label2id, BERT_SCORE_MODEL_PATH,
    pad_and_concat, load_checkpoint, truncate_generation_after_first_EOU
)
from modeling_gpt2_cvae import GPT2Config, GPT2ForVAE, GPT2LMHeadModel, PTModel


class BERTScoreProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        with open(self.args.input_file, "r", encoding="utf-8") as f:
            tasks = json.loads(f.read())
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None \
               and all([(key in result and task[key]==result[key])
                        for key in ["prompt", "output", "reference"]]) \
               and all([(key in result and task[key]==result[key])
                        for key in [f"{preference}_{metric}"
                                    for metric in ["score", "acc"]
                                    for preference in (list(intent_label2id.keys())+["consistency"])]]) \
               and all([key in result
                        for key in ["bertscore"]])

class BERTScoreConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        from datasets import load_metric
        self.bertscore = load_metric("./bertscore")

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        predictions = [task["prompt"] + task["output"] for task in tasks]
        references = [task["prompt"] + task["reference"] for task in tasks]
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            model_type=BERT_SCORE_MODEL_PATH,
            device=self.device,
            batch_size=self.batch_size
        )
        for i in range(len(tasks)):
            tasks[i]["bertscore"] = results["f1"][i]
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
               and all([f"{preference}_{metric}" in result
                        for metric in ["score", "acc"]
                        for preference in (list(intent_label2id.keys())+["consistency"])])

class RewardConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        self.reward_fn = get_reward_fn(device=self.device)

    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        prompts = [task["prompt"] for task in tasks]
        labels = [task["output"] for task in tasks]
        intent_probs = self.reward_fn(prompts, labels, return_probs=True)
        for i, task in enumerate(tasks):
            output_intent = torch.argmax(intent_probs[i, :]).item()
            local_intent_label2id = {**intent_label2id, "consistency": task["intent"]}
            for intent, id in local_intent_label2id.items():
                task[f"{intent}_score"] = intent_probs[i, id].item()
                task[f"{intent}_acc"] = int(output_intent == id)
            task["intent_probs"] = intent_probs[i, :].tolist()
        return tasks

class GenerateProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        from dailydialog_dataset import DailyDialogDataset
        test_dataset = DailyDialogDataset(usage="test")
        tasks = [
            {"input_ids": test_dataset[i]["prompt_ids"].tolist(),
             "label_ids": test_dataset[i]["label_ids"].tolist(),
             "intent": test_dataset[i]["intent"].item()}
            for i in range(len(test_dataset))
        ]
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return result is not None \
               and all([(key in result and task[key]==result[key])
                        for key in ["input_ids", "label_ids", "intent"]]) \
               and all([key in result
                        for key in ["prompt", "output"]])

class GenerateConsumer(BatchTaskConsumer):
    def init_model(self) -> None:
        tokenizer = get_tokenizer()
        if self.args.test_ground_truth:
            model = None
        elif self.args.test_one2many_results:
            model = None
            dataset = DailyDialogOne2ManyDataset(usage="test")
            self.input_ids_to_generation = {}
            for data in dataset.data:
                self.input_ids_to_generation[str(data["input_ids"])] = data["generation"]
        elif self.args.vae_model_path is not None:
            model = GPT2ForVAE.from_pretrained(self.args.vae_model_path)
            model.config.eos_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = tokenizer.eos_token_id
            if self.args.lpo_model_path is not None:
                model.load_lpo_policy_latent_encoder(self.args.lpo_model_path)
            model = model.half().to(self.device).eval()
        elif self.args.dpo_model_path is not None:
            model = GPT2LMHeadModel.from_pretrained(
                BASE_MODEL_PATH,
                torch_dtype=torch.float16,
                device_map=self.device,
            ).eval()
            sd = load_checkpoint(self.args.dpo_model_path, self.device)
            sd = {k.replace("base.", ""): v for k, v in sd.items() if k.startswith("base.")}
            model.load_state_dict(sd)
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
        reference_ids = pad_and_concat(
            [torch.LongTensor(task["label_ids"]) for task in tasks],
            pad_token_id=self.tokenizer.pad_token_id,
            padding_side="left"
        ).to(self.device)
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(reference_ids, skip_special_tokens=True)
        if self.args.test_ground_truth:
            for task, prompt, reference in zip(tasks,prompts,references):
                task['prompt'] = prompt
                task["reference"] = reference
                task['output'] = reference
                assert truncate_generation_after_first_EOU(reference) == reference
        elif self.args.test_one2many_results:
            for task, prompt, reference in zip(tasks,prompts,references):
                task['prompt'] = prompt
                task["reference"] = reference
                task['output'] = self.input_ids_to_generation[str(task["input_ids"])][random.randint(0,31)]
                assert truncate_generation_after_first_EOU(reference) == reference
                assert truncate_generation_after_first_EOU(task['output']) == task['output']
        else:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=20,
                    min_length=2,
                    max_new_tokens=output_max_length,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            outputs = self.tokenizer.batch_decode(output_ids[:,inputs_len:], skip_special_tokens=True)
            prompts = self.tokenizer.batch_decode(output_ids[:,:inputs_len], skip_special_tokens=True)
            for task,output,prompt,reference in zip(tasks,outputs,prompts,references):
                task['prompt'] = prompt
                task["reference"] = reference
                task['output'] = truncate_generation_after_first_EOU(output)
        return tasks

def summarize_results(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        results = json.loads(f.read())
    pd_dict = {}
    pd_index = ["score", "acc"]
    for preference in (list(intent_label2id.keys())+["consistency"]):
        preference_score = np.mean([result[f"{preference}_score"] for result in results])
        preference_acc = np.mean([result[f"{preference}_acc"] for result in results])
        pd_dict[preference] = [preference_score, preference_acc]
    df = pd.DataFrame(pd_dict, index=pd_index)
    print(df)
    bertscore = np.mean([result["bertscore"] for result in results])
    print(f"bertscore: {bertscore:.3f}")
    return df, bertscore

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
    parser.add_argument('--test_ground_truth', type=int, default=0)
    parser.add_argument('--test_one2many_results', type=int, default=0)
    parser.add_argument('--sft_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--pt_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--dpo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--vae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--cvae_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--lpo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--ppo_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument('--lora_model_path', type=auto_find_checkpoint_dir, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--num_run_times', type=int, default=1)

    # the input file of generation to reward
    parser.add_argument("--input_file", type=str, default=None)

    args = parser.parse_args()

    if args.py_file_name is None:
        args.py_file_name = os.path.basename(__file__)

    if args.local_rank == -1:
        if args.test_ground_truth:
            args.output_file = "test_ground_truth_inference.json"
        elif args.test_one2many_results:
            args.output_file = "test_one2many_results_inference.json"
        elif args.pt_model_path is not None:
            args.output_file = os.path.join(args.pt_model_path, "inference.json")
        elif args.dpo_model_path is not None:
            args.output_file = os.path.join(args.dpo_model_path, "inference.json")
        elif args.vae_model_path is not None:
            if args.lpo_model_path is not None:
                args.output_file = os.path.join(args.lpo_model_path, "inference.json")
            else:
                args.output_file = os.path.join(args.vae_model_path, "inference.json")
        elif args.cvae_model_path is not None:
            if args.lpo_model_path is not None:
                args.output_file = os.path.join(args.lpo_model_path, "inference.json")
            else:
                args.output_file = os.path.join(args.cvae_model_path, "inference.json")
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
        #intent_accs, bertscores = [], []
        dfs, bertscores = [], []
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
            args.task_name = "bertscore"
            args.input_file = args.output_file
            args.output_file = args.output_file.replace("inference_and_reward.json", "inference_and_reward_and_bertscore.json")
            producer = BERTScoreProducer(args)
            producer.run()
            # step4. analyze the results
            df, bertscore = summarize_results(input_file=args.output_file)
            dfs.append(df)
            bertscores.append(bertscore)
        df = pd.concat(dfs)
        mean, std = df.groupby(level=0).mean(), df.groupby(level=0).std()
        print(f"{mean.round(3).astype(str) + '±' + std.round(3).astype(str)}")
        print(f"bertscore: {np.mean(bertscores):.3f}±{np.std(bertscores):.3f}")
    else:
        if args.task_name == "generate":
            consumer = GenerateConsumer(args)
            consumer.run()
        elif args.task_name == "reward":
            consumer = RewardConsumer(args)
            consumer.run()
        elif args.task_name == "bertscore":
            consumer = BERTScoreConsumer(args)
            consumer.run()
        else:
            raise NotImplementedError(args.task_name)

if __name__ == "__main__":
    main()
