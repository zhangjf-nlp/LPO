"""
This Python file implements a Map-Reduce framework with a TaskProducer abstract class and a
TaskConsumer abstract class. Each TaskConsumer instance independently utilizes a GPU to load
a Torch model for data processing, i.e., in data-parallel.

Classes:
- TaskProducer:
    load tasks from file
    load and check cached results from file
    clear the present tasks and results in redis queue (may be left by killed programmes)
    launch consumers and send tasks to them
    recieve results from consumers and save them to file
    stop consumers at the end
    - abstractmethod:
        load_tasks
        cached_result_is_valid
- TaskConsumer:
    init model on corresponding gpu (specified by local_rank)
    recieve tasks from producer
    process tasks with model
    send results to producer
    - abstractmethod:
        init_model
        process_task
Functions:
- parse_args (demo):
    parse args for TaskProducer and TaskConsumer
- main (demo):
    the programme entrance

Note:
This script requires redis server running in the background for Inter-Process Communication.
You can install redis server and run it through the following commands:
# apt-get install redis-server
# redis-server --port PORT
"""
import os
import re
import sys
import json
import time
import argparse
import threading
import concurrent

import redis
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

class TaskProducer(ABC):
    def __init__(self, args):
        self.args = args
        self.producer = redis.Redis(host=args.host, port=args.port, db=0)
        self.consumer = redis.Redis(host=args.host, port=args.port, db=1)
        self.task_queue_name = f'{self.args.task_name}_task'
        self.result_queue_name = f'{self.args.task_name}_result'
        self.has_slept_since_last_save = False

    def clear_queue(self):
        num_tasks_cleared = 0
        task = self.producer.lpop(self.task_queue_name)
        while task is not None:
            num_tasks_cleared += 1
            task = self.producer.lpop(self.task_queue_name)

        num_results_cleared = 0
        result = self.consumer.lpop(self.result_queue_name)
        while result is not None:
            num_results_cleared += 1
            result = self.consumer.lpop(self.result_queue_name)

        print(f"clear {num_tasks_cleared} tasks and {num_results_cleared} results in queue")
        return num_tasks_cleared, num_results_cleared

    def auto_clear_queue(self):
        stop_clearing = False
        while not stop_clearing:
            num_tasks_cleared, num_results_cleared = self.clear_queue()
            def input_choice():
                time.sleep(10)
                return num_tasks_cleared == 0 and num_results_cleared == 0
            executor = concurrent.futures.ThreadPoolExecutor()
            future = executor.submit(input_choice)
            try:
                stop_clearing = future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                stop_clearing = num_tasks_cleared == 0 and num_results_cleared == 0
            finally:
                executor.shutdown(wait=False)

    def wait_for_next_result(self):
        while True:
            result = self.consumer.lpop(self.result_queue_name)
            if result is not None:
                result = json.loads(result.decode('utf-8'))
                return result
            time.sleep(1)
            self.has_slept_since_last_save = True

    @abstractmethod
    def load_tasks(self) -> list[dict]:
        tasks = []
        return tasks

    def save_results(self):
        num_valid_results = len([_ for _ in self.results if _ is not None])
        print(f"save results ({num_valid_results}/{len(self.tasks)}) into {self.args.output_file}")
        with open(self.args.output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.results, ensure_ascii=False, indent=4))

    def load_cached_results(self):
        if os.path.exists(self.args.output_file) and not self.args.erase:
            with open(self.args.output_file, "r", encoding="utf-8") as f:
                results = json.loads(f.read())
            if len(results) == len(self.tasks):
                return results
        results = [None for index in range(len(self.tasks))]
        return results

    @abstractmethod
    def cached_result_is_valid(self, task, result) -> bool:
        return True

    def push_tasks(self):
        num_cached_results, num_pushed_tasks = 0, 0
        for index, task in enumerate(tqdm(self.tasks, desc=f"pushing tasks into {self.task_queue_name}...")):
            if "index" not in task:
                task["index"] = index
            if self.results[index] is not None:
                if self.cached_result_is_valid(task, self.results[index]):
                    num_cached_results += 1
                    continue
                else:
                    self.results[index] = None
            self.producer.rpush(self.task_queue_name, json.dumps(task, ensure_ascii=False))
            num_pushed_tasks += 1
        print(f"read {num_cached_results} results from cache_file and push {num_pushed_tasks} tasks into {self.task_queue_name}")
        return num_cached_results, num_pushed_tasks

    def recieve_results(self):
        first_result = self.wait_for_next_result()
        for num_received_results in tqdm(range(self.num_pushed_tasks), desc=f"receiving results from {self.result_queue_name}..."):
            if num_received_results == 0:
                result = first_result
            else:
                result = self.wait_for_next_result()
            index = result["index"]
            while not self.cached_result_is_valid(self.tasks[index], result):
                print(f"invalid result is received and ignored:\n{json.dumps(result, ensure_ascii=False, indent=4)}")
                result = self.wait_for_next_result()
                index = result["index"]
            self.results[index] = result
            num_valid_results = self.num_cached_results + num_received_results + 1
            if num_valid_results % max(len(self.tasks)//100, 100) == 0 and self.has_slept_since_last_save:
                self.save_results()
                self.has_slept_since_last_save = False

    def run(self):
        self.tasks = self.load_tasks()
        self.results = self.load_cached_results()
        self.auto_clear_queue()
        self.num_cached_results, self.num_pushed_tasks = self.push_tasks()

        if self.num_pushed_tasks > 0:
            self.launch_consumers()
            self.recieve_results()
            self.save_results()
            self.stop_consumers()

    def launch_consumers(self):
        def launch_consumers(args):
            args_input = vars(args)
            args_input = {k:v for k,v in args_input.items() if v is not None}
            for local_rank in range(torch.cuda.device_count()):
                args_input["local_rank"] = local_rank
                argv = ' '.join([f'--{k} "{v}"' for k, v in args_input.items()])
                cmd = f'nohup python {args.py_file_name} {argv}' \
                      f' > nohup.{args.py_file_name}.rank_{local_rank}.txt 2>&1 &'
                print(f"{cmd}")
                os.system(cmd)
        launch_consumers(self.args)

    def stop_consumers(self):
        for local_rank in range(torch.cuda.device_count()):
            task = {"signal": "exit"}
            self.producer.rpush(self.task_queue_name, json.dumps(task, ensure_ascii=False))

class TaskConsumer(ABC):
    def __init__(self, args):
        self.args = args
        self.consumer = redis.Redis(host=args.host, port=args.port, db=0)
        self.producer = redis.Redis(host=args.host, port=args.port, db=1)
        self.task_queue_name = f'{self.args.task_name}_task'
        self.result_queue_name = f'{self.args.task_name}_result'
        self.device = torch.device(f"cuda:{args.local_rank}")
        self.local_task_queue, self.task_queue_lock = [], threading.Lock()
        self.local_result_queue, self.result_queue_lock = [], threading.Lock()
        self.exit_signal = threading.Event()

    @abstractmethod
    def init_model(self) -> None:
        pass

    def listening(self):
        num_tasks_per_sleep = 0
        num_tasks_before_sleep = 0
        while not self.exit_signal.is_set():
            with self.task_queue_lock:
                num_local_tasks = len(self.local_task_queue)
            num_tasks_this_sleep = num_tasks_before_sleep - num_local_tasks
            num_tasks_per_sleep = max(int(num_tasks_per_sleep * 0.8), num_tasks_this_sleep)
            num_tasks_expected_in_queue = max(10, num_tasks_per_sleep * 2)
            if num_local_tasks >= num_tasks_expected_in_queue:
                num_tasks_before_sleep = num_local_tasks
                time.sleep(1)
                continue
            for _ in range(num_tasks_expected_in_queue-num_local_tasks):
                task = self.consumer.lpop(self.task_queue_name)
                if task is None:
                    break
                task = json.loads(task.decode('utf-8'))
                if "signal" in task and task["signal"] == "exit":
                    self.exit_signal.set()
                    break
                with self.task_queue_lock:
                    self.local_task_queue.append(task)
            with self.task_queue_lock:
                num_local_tasks = len(self.local_task_queue)
            num_tasks_before_sleep = num_local_tasks
            time.sleep(1)

    def sending(self):
        while not self.exit_signal.is_set():
            with self.result_queue_lock:
                num_local_results = len(self.local_result_queue)
            if num_local_results == 0:
                time.sleep(1)
                continue
            with self.result_queue_lock:
                result = self.local_result_queue.pop(0)
            self.producer.rpush(self.result_queue_name, result)

    def start_communication_threads(self):
        self.threads = [
            threading.Thread(target=self.listening),
            threading.Thread(target=self.sending)
        ]
        for thread in self.threads:
            thread.start()

    def join_communication_threads(self):
        for thread in self.threads:
            thread.join()

    @abstractmethod
    def process_task(self, task: dict) -> dict:
        result = task
        return result

    def processing_task_loop(self):
        while not self.exit_signal.is_set():
            with self.task_queue_lock:
                num_local_tasks = len(self.local_task_queue)
            if num_local_tasks == 0:
                time.sleep(1)
                continue
            with self.task_queue_lock:
                task = self.local_task_queue.pop(0)
            result = self.process_task(task)
            with self.result_queue_lock:
                self.local_result_queue.append(json.dumps(result, ensure_ascii=False))

    def run(self):
        self.init_model()
        self.start_communication_threads()
        self.processing_task_loop()
        self.join_communication_threads()

class BatchTaskConsumer(TaskConsumer):
    def __init__(self, args):
        super().__init__(args)
        self.batch_size = getattr(args, "batch_size", 2)

    def process_task(self, task: dict) -> dict:
        pass

    @abstractmethod
    def process_tasks(self, tasks: list[dict]) -> list[dict]:
        results = tasks
        return results

    def processing_task_loop(self):
        while not self.exit_signal.is_set():
            with self.task_queue_lock:
                num_local_tasks = len(self.local_task_queue)
            if num_local_tasks == 0:
                time.sleep(1)
                continue
            with self.task_queue_lock:
                tasks = self.local_task_queue[:self.batch_size]
                self.local_task_queue = self.local_task_queue[self.batch_size:]
            results = self.process_tasks(tasks)
            with self.result_queue_lock:
                for result in results:
                    self.local_result_queue.append(json.dumps(result, ensure_ascii=False))


# implementation examples
class MyProducer(TaskProducer):
    def load_tasks(self) -> list[dict]:
        tasks = []
        return tasks

    def cached_result_is_valid(self, task, result) -> bool:
        return True


class MyConsumer(TaskConsumer):
    def init_model(self) -> None:
        pass

    def process_task(self, task: dict) -> dict:
        result = task
        return result


def parse_args():
    parser = argparse.ArgumentParser(description='parse args for map-reduce processing on multi-gpu')
    parser.add_argument("--py_file_name", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6380)
    parser.add_argument("--task_name", type=str, default="inference")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--erase", type=int, default=0)
    args = parser.parse_args()

    if args.py_file_name is None:
        args.py_file_name = os.path.basename(__file__)

    return args


def main():
    args = parse_args()
    if args.local_rank == -1:
        producer = TaskProducer(args)
        producer.run()
    else:
        consumer = TaskConsumer(args)
        consumer.run()


if __name__ == "__main__":
    main()
