import argparse
import random
import textwrap
from typing import Dict, List
import onnx, time, os, gc, json
import onnxruntime as ort

import psutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import numpy as np
from onnxruntime.training.api import CheckpointState, Module, Optimizer, LinearLRScheduler
import onnxruntime as rt
from onnxruntime import SessionOptions
from datasets import Dataset
from datetime import datetime
from collections import defaultdict

from torch.utils.data import DataLoader
import yaml
from research.offline_train_eval import DATASET_MAPPING, PEFTBenchmarkDataset
from tools.utils import preload_dataset
from trainer.utils import DataCollatorForSupervisedDataset, taskname_to_deepeval_preprocess_function
from tools.parser_config import TASK_NAME_TO_DATASET, TRAIN_CONFIG, ARTIFACT_CONFIG, ARTIFACT_VALIDATOR_CONFIG

import numpy as np
import json
import os

class CosineLRScheduler:
    """Cosine Learning Rate Scheduler for ONNX Runtime Training."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, initial_lr=0.001):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.current_step = 0
        
        # Set initial learning rate
        if warmup_steps > 0:
            initial_warmup_lr = 0.0  # Start from 0 during warmup
        else:
            initial_warmup_lr = initial_lr  # No warmup, start at target LR
        self.optimizer.set_learning_rate(initial_warmup_lr)

    def step(self, increment=True):
        """Update learning rate with warmup + cosine decay."""
        if increment:
            self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup: Increase LR from 0 to initial_lr
            new_lr = (self.initial_lr * self.current_step) / self.warmup_steps
        else:
            # Cosine decay after warmup
            decay_step = self.current_step - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps
            
            # Handle edge case where decay_total might be 0
            if decay_total <= 0:
                new_lr = self.min_lr
            else:
                cos_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_total))
                new_lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_decay

        # Ensure LR doesn't go below min_lr
        new_lr = max(new_lr, self.min_lr)
        self.optimizer.set_learning_rate(new_lr)
        
    def get_learning_rate(self):
        """Get current learning rate from optimizer."""
        return self.optimizer.get_learning_rate()
    
    def state_dict(self):
        """Return the state of the scheduler as a dictionary."""
        return {
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'min_lr': self.min_lr,
            'initial_lr': self.initial_lr,
            'current_step': self.current_step,
        }
    
    def load_state_dict(self, state_dict, update_total_steps=None):
        """Load the scheduler state from a dictionary."""
        self.warmup_steps = state_dict['warmup_steps']
        self.min_lr = state_dict['min_lr']
        self.initial_lr = state_dict['initial_lr']
        self.current_step = state_dict['current_step']
        
        # Handle total_steps change
        if update_total_steps is not None:
            print(f"Updating total_steps from {state_dict['total_steps']} to {update_total_steps}")
            self.total_steps = update_total_steps
        else:
            self.total_steps = state_dict['total_steps']
        
        # Update optimizer with current learning rate without incrementing step
        self.step(increment=False)
    
    def save_checkpoint(self, filepath):
        """Save scheduler state to a file."""
        state = self.state_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_checkpoint(self, filepath):
        """Load scheduler state from a file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.load_state_dict(state)
    
    def reset(self):
        """Reset the scheduler to initial state."""
        self.current_step = 0
        if self.warmup_steps > 0:
            initial_lr = 0.0
        else:
            initial_lr = self.initial_lr
        self.optimizer.set_learning_rate(initial_lr)
    
    def get_last_lr(self):
        """Get the last computed learning rate (for compatibility with PyTorch schedulers)."""
        return [self.get_learning_rate()]
    
    def __repr__(self):
        return (f"CosineLRScheduler(warmup_steps={self.warmup_steps}, "
                f"total_steps={self.total_steps}, min_lr={self.min_lr}, "
                f"initial_lr={self.initial_lr}, current_step={self.current_step})")


class ORTDataCurator:

    def __init__(self,
                 model_id,
                 task_name,
                 max_dataset_length = None,
                 remove_long_samples = True,
                 max_context_length=512,
                 test_ratio=0.1,
                 batch_size=4,
                 split=True,
                 shuffle=False) -> None:
        
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.environ['HF_TOKEN'])
        self.collator = DataCollatorForSupervisedDataset(self.tokenizer)

        self.max_dataset_length = max_dataset_length
        self.max_context_length = max_context_length
        self.remove_long_samples = remove_long_samples

        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.split = split
        self.shuffle = shuffle

        self.dataset = None
        self._setup_dataset(task_name)

        ds = preload_dataset(self.dataset_id, self.dataset_name)
        self.prepare_dataset(ds, custom_preprocess=taskname_to_deepeval_preprocess_function(self.dataset_config.value))

    def _setup_dataset(self, dataset_input):
        """Setup dataset configuration from simple string identifier or enum."""
        # Handle both string and enum inputs

        if isinstance(dataset_input, PEFTBenchmarkDataset):
            self.dataset_config = dataset_input    
        else:
            if dataset_input.lower() not in list(DATASET_MAPPING.keys()):
                raise ValueError(f"Unsupported dataset: {dataset_input}. Choose from: {list(DATASET_MAPPING.keys())}")
            
            self.dataset_config = PEFTBenchmarkDataset[dataset_input.upper()]
        
        self.dataset_id = DATASET_MAPPING[self.dataset_config.value][0]
        self.preprocess_id = DATASET_MAPPING[self.dataset_config.value][1]

        if len(DATASET_MAPPING[self.dataset_config.value]) > 2:
            self.dataset_name = DATASET_MAPPING[self.dataset_config.value][2]
        else:
            self.dataset_name = None

    # Define preprocessing function for tokenization
    def prepare_dataset(self, dataset : Dataset, custom_preprocess = None):

        raw_columns = dataset["train"].column_names

        if self.split:
            test_size = self.test_ratio if self.max_dataset_length == None else int(self.max_dataset_length * self.test_ratio)
            train_size = (1 - self.test_ratio) if self.max_dataset_length == None else int(self.max_dataset_length * (1 - self.test_ratio))
            dataset = dataset["train"].train_test_split(test_size=test_size, train_size=train_size, shuffle=self.shuffle)    

        def process_sample(sample):
            
            if custom_preprocess:
                return custom_preprocess(sample, self.tokenizer, (self.batch_size > 1))

            return self.tokenizer(sample, return_dict=True, tokenize=True, return_tensors="np", padding=True, add_generation_prompt=False)
        
        def filter_sample(sample):
            return len(sample["input_ids"]) < self.max_context_length

        # Convert the list of tokenized samples into a Dataset
        dataset = dataset.map(process_sample, batched=(self.batch_size > 1), batch_size=self.batch_size)

        if self.remove_long_samples:
            dataset = dataset.filter(filter_sample, batched=False)

        dataset = dataset.remove_columns(raw_columns)

        self.dataset = dataset

class ORTTrainingArguments:

    def __init__(self,
                model_id=None,
                peft_method=None,
                peft_rank = None,
                peft_alpha = None,
                peft_target = None,
                export_inference=False,
                test_inference=False,
                test_evaluate=False,
                batch_size=1,
                learning_rate=1e-3,
                min_learning_rate=0,
                max_sequence_length=100,
                max_dataset_length=100,
                num_train_epochs=1,
                warmup_steps=10,
                max_steps=10,
                save_steps=100,
                remove_long_samples=True,
                dataset_split : bool = False,
                dataset_shuffle : bool = False,
                dataset_test_ratio : bool = 0.1,
                scheduler_type="linear",
                grad_accum_steps=4) -> None:
        
        self.model_id = model_id
        self.peft_method = peft_method
        self.export_inference = export_inference
        self.test_inference = test_inference
        self.test_evaluate = test_evaluate
        
        self.max_sequence_length = max_sequence_length
        self.max_dataset_length = max_dataset_length
        self.remove_long_samples = remove_long_samples
        self.batch_size = batch_size

        self.dataset_split = dataset_split
        self.dataset_shuffle = dataset_shuffle
        self.dataset_test_ratio = dataset_test_ratio

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.scheduler_type = scheduler_type
        self.grad_accum_steps = grad_accum_steps

        self.peft_rank = peft_rank
        self.peft_alpha = peft_alpha
        self.peft_target = peft_target
        self.trainable_parameter_count = 0
    
    def load_from_json(self, train_dir):
        with open(f"{train_dir}/training_config.json", "r") as f:
            data = json.load(f)

        self.export_inference = False
        self.test_inference = False
        self.test_evaluate = False

        scheduler_options = data.get("schedulerOptions", None)
        self.scheduler_type = data.get("schedulerType", "cosine")

        if self.scheduler_type == "linear" and scheduler_options:
            self.learning_rate = scheduler_options.get("linearLearningRate", 1e-3)
            self.start_factor = scheduler_options.get("startFactor", 1)
            self.end_factor = scheduler_options.get("endFactor", 0.333)
            self.warmup_steps = scheduler_options.get("warmupSteps", 10)
        elif self.scheduler_type == "cosine" and scheduler_options:
            self.min_learning_rate = scheduler_options.get("minLearningRate", 0)
            self.learning_rate = scheduler_options.get("cosineLearningRate", 1e-3)
            self.warmup_steps = scheduler_options.get("warmupSteps", 10)

        self.num_train_epochs = data.get("numTrainEpochs", 1)
        self.max_steps = data.get("maxSteps", 20)
        self.save_steps = data.get("saveSteps", 50)
        self.grad_accum_steps = data.get("gradAccumSteps", 2)
        self.batch_size = data.get("batchSize", 1)

        # Task name and train file if needed
        self.task_name = data.get("taskName", None)

        # Model id
        self.model_id = data.get("modelId", None)

        # Dataset configuration
        dataset = data.get("datasetOptions", None)

        if dataset:
            self.train_file = dataset.get("trainFile", None)
            self.dataset_split = dataset.get("datasetSplit", False)
            self.dataset_shuffle = dataset.get("datasetShuffle", False)
            self.dataset_batch_size = dataset.get("datasetBatchSize", 64)
            self.dataset_test_ratio = dataset.get("testRatio", 0.1)
            self.max_sequence_length = dataset.get("maxSequenceLength", 512)
            self.max_dataset_length = dataset.get("maxDatasetLength", 100)
            self.remove_long_samples = dataset.get("removeLongSamples", True)

            if self.dataset_batch_size is None:
                self.dataset_batch_size = 64
        
        # Peft method
        self.peft_method = data.get("peftMethod", None)
        self.peft_rank = data.get("rank", None)
        self.peft_alpha = data.get("alpha", None)
        self.peft_target = data.get("peft_target", None)
        self.trainable_parameter_count = data.get("trainable_parameter_count", 0)
        self.peft_mapping = data.get("peft_mapping", None)

        return self

class ORTTrainer:

    def __init__(self,
                training_model_dir,
                args : ORTTrainingArguments = None,
                load_from_state=False,
                inference_model_path="inference_model.onnx",
                callbacks=None,
                seed=42
                ) -> None:
        
        self.training_model_dir = training_model_dir
        self.inference_model_path = inference_model_path
        self.callbacks = callbacks
        self.args = args
        self.model = None
        self.state = None
        self.optimizer = None
        self.tokenizer = None
        self.seed = seed
        self.scheduler = None
        self.load_from_state = load_from_state

        if self.load_from_state:
            try:
                with open(f"{self.training_model_dir}/training_state.json", "r") as f:
                    self.state = json.load(f)
            except FileNotFoundError:
                print("No training state found.")
                self.load_from_state = False

        self._load_train_config()

        self._set_seed()

        self.data_curator = ORTDataCurator(model_id=self.args.model_id,
                                            task_name=self.args.task_name,
                                            max_dataset_length=self.args.max_dataset_length,
                                            remove_long_samples=self.args.remove_long_samples,
                                            max_context_length=self.args.max_sequence_length,
                                            test_ratio=self.args.dataset_test_ratio,
                                            split=self.args.dataset_split,
                                            shuffle=self.args.dataset_shuffle,
                                            batch_size=self.args.dataset_batch_size
                                            )
        
        self.train_model_name = self._create_model_name()
        
    def _load_train_config(self):

        if self.args is not None:
            return
        
        self.args = ORTTrainingArguments().load_from_json(self.training_model_dir)

    def set_scheduler_type(self, total_steps):

        if self.args.scheduler_type == "linear":
            return LinearLRScheduler(self.optimizer, self.args.warmup_steps, total_steps, initial_lr=self.args.learning_rate)
        elif self.args.scheduler_type == "cosine":
            return CosineLRScheduler(self.optimizer, self.args.warmup_steps, total_steps, min_lr=self.args.min_learning_rate, initial_lr=self.args.learning_rate)
        else:
            raise ValueError("Unsupported scheduler type. Use 'linear' or 'cosine'.")

    def load_onnx_trainer(self):
        self.checkpoint_state = CheckpointState.load_checkpoint(f"{self.training_model_dir}/checkpoint")

        # TODO: Customize
        sess_options = SessionOptions()
        sess_options.enable_profiling = False
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL#ORT_PARALLEL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.enable_cpu_mem_arena = False
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

        self.model = Module(f"{self.training_model_dir}/training_model.onnx", self.checkpoint_state, f"{self.training_model_dir}/eval_model.onnx", session_options=sess_options)
        self.optimizer = Optimizer(f"{self.training_model_dir}/optimizer_model.onnx", self.model)

    def train(self):
        """
        Checks the model if the outputs are training correctly as well as evaluation.
        Exports model for inference if needed or transfers the weights to an already existing inference model.
        """

        # Save the training config
        self.save_training_config()
        
        self.load_onnx_trainer()
        
        train_dataset = self.data_curator.dataset["train"]

        total_samples = len(train_dataset)
        steps_per_epoch = total_samples // self.args.batch_size
        total_epoch_steps = self.args.num_train_epochs * steps_per_epoch

        if self.args.max_steps is not None:
            total_steps = self.args.max_steps
            print(f"[INFO] Training for {self.args.max_steps} steps")
        else:
            total_steps = total_epoch_steps
            print(f"[INFO] Training for {self.args.num_train_epochs} epochs ({total_steps} steps)")

        global_step = 0
        epoch = 0

        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.dataset_shuffle,
            collate_fn=self.data_curator.collator.numpy_call
        )

        # Calculate total steps
        steps_per_epoch = len(dataloader)
        total_epoch_steps = self.args.num_train_epochs * steps_per_epoch
        
        # Determine whether to use steps or epochs
        total_steps = self.args.max_steps if self.args.max_steps is not None else total_epoch_steps
        
        # Scheduler
        self.scheduler = self.set_scheduler_type(total_steps)

        # Handle checkpoint resuming
        resume_from_step = 0
        resume_from_epoch = 0

        if self.load_from_state:
            # Load checkpoint and get the step/epoch we're resuming from
            resume_from_step = self.state.get('current_global_step', 0)
            resume_from_epoch = self.state.get('current_epoch', 0)
            
            # Load scheduler state
            if 'scheduler_state' in self.state:
                # Option 1: Strict loading (assumes total_steps hasn't changed)
                self.scheduler.load_state_dict(self.state['scheduler_state'])
                
                # Option 2: Allow total_steps to change
                # self.scheduler.load_state_dict(checkpoint_data['scheduler_state'], strict=False)
                
                # Option 3: Update total_steps and preserve progress
                # self.scheduler.load_state_dict(checkpoint_data['scheduler_state'], 
                #                               update_total_steps=total_steps, 
                #                               preserve_progress=True)
            
            print(f"Resuming training from step {resume_from_step}, epoch {resume_from_epoch}")

        # Main training loop
        global_step = resume_from_step
        epoch = resume_from_epoch
        pbar = tqdm(total=total_steps, desc="ONNX Runtime Training")
        
        accumulated_loss = 0.0
        start_time = time.time()

        logs = []

        if self.load_from_state and global_step > 0 and self.args.max_steps:
            total_steps += self.args.max_steps

        while global_step < total_steps:

            # Skip to the right epoch if resuming
            if epoch < resume_from_epoch:
                epoch += 1
                continue

            for batch_idx, batch in enumerate(dataloader):
                if global_step >= total_steps:
                    break

                # Skip batches if we're resuming mid-epoch
                if epoch == resume_from_epoch and batch_idx < (resume_from_step % steps_per_epoch):
                    continue
                
                pbar.write(f"[INFO] Epoch {epoch}, Step {global_step}")
                input_ids_np = batch['input_ids']
                position_ids = self.create_position_ids(input_ids_np, padding_idx=self.data_curator.tokenizer.pad_token_id)

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "position_ids": position_ids,
                    "labels": batch['labels']
                }

                self.model.train()
                forward = self.model(*inputs.values())

                current_loss = forward[0]

                accumulated_loss += current_loss

                if (global_step + 1) % self.args.grad_accum_steps == 0:

                    self.optimizer.step()
                    self.model.lazy_reset_grad()

                    avg_loss = accumulated_loss / self.args.grad_accum_steps

                    pbar.write(f"[INFO] Training average loss step: {avg_loss}")
                    pbar.write(f"[INFO] LR: {self.optimizer.get_learning_rate()}")

                    # Reset accumulated loss
                    accumulated_loss = 0.0
                
                self.scheduler.step()

                global_step += 1
                pbar.update(1)

                step_info = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": current_loss.item(),
                    "cpu_mem": psutil.Process().memory_info().rss / 1e9,
                    "gpu_mem": torch.cuda.memory_allocated() / 1e9,
                    "learning_rate": self.optimizer.get_learning_rate()
                }

                pbar.write(str(step_info))
                logs.append(step_info)

                if self.args.save_steps and (global_step + 1) % self.args.save_steps == 0:
                    self.save_checkpoint({
                        "scheduler_state": self.scheduler.state_dict()
                    })

                
                if global_step >= total_steps:
                    break
            
            if global_step >= total_steps:     
                epoch += 1

        pbar.close()

        # End time tracking
        end_time = time.time()
        total_runtime = end_time - start_time
        steps_per_second = global_step / total_runtime if total_runtime > 0 else 0

        # Print training performance
        print(f"\n[INFO] Total training time: {total_runtime:.4f} seconds")
        print(f"[INFO] Training steps per second: {steps_per_second:.4f}")

        logs.append({
                    "step": global_step,
                    "epoch": epoch,
                    "loss": current_loss.item(),
                    "cpu_mem": psutil.Process().memory_info().rss / 1e9,
                    "gpu_mem": torch.cuda.memory_allocated() / 1e9,
                    "train_runtime": total_runtime,
                    "train_steps_per_second": steps_per_second
                })
        
        # Save logs
        with open(f"{self.training_model_dir}/training_logs.json", mode="w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False)
        
        # Training state

        # Save the checkpoint and state
        self.save_checkpoint({
            "scheduler_state": self.scheduler.state_dict(),
            "current_global_step": global_step,
            "current_epoch": epoch,
        })

        if self.args.export_inference:

            exclude_nodes = ["loss"]

            # Model inference: we want to get only logits and hidden states for decoding
            self.model.export_model_for_inferencing(f"{self.training_model_dir}/{self.inference_model_path}", [ out_name for out_name in self.model.output_names() if out_name not in exclude_nodes])

            del self.model
            del self.checkpoint_state
            del self.optimizer
            gc.collect()

    def create_position_ids(self, input_ids: np.ndarray, padding_idx: int = 0):
        # Create a mask where input_ids are not padding
        mask = input_ids != padding_idx

        # Create position indices for each sequence
        position_ids = np.cumsum(mask, axis=1) - mask

        return position_ids

    def _create_model_name(self) -> str:
        """Create a unique model name based on configuration."""
        # Extract model name from model_id (e.g., "TinyLlama/TinyLlama_v1.1" -> "TinyLlama_v1.1")
        model_short_name = self.args.model_id.split('/')[-1] if '/' in self.args.model_id else self.args.model_id
        
        # Create name pattern: {model_name}-{peft_method}-{dataset}-r{rank}-a{alpha_ratio}
        alpha_ratio = self.args.peft_alpha // self.args.peft_rank if self.args.peft_rank > 0 else 1
        
        return f"{model_short_name}-{self.args.peft_method}-{self.data_curator.dataset_config.value.lower()}-r{self.args.peft_rank}-a{alpha_ratio}"

    def save_checkpoint(self, state_data=None):
        CheckpointState.save_checkpoint(self.checkpoint_state, f"{self.training_model_dir}/checkpoint")

        if state_data:
            with open(f"{self.training_model_dir}/training_state.json", "w") as f:
                json.dump(state_data, f, ensure_ascii=False)

    def _set_seed(self):
        """Set random seed for reproducibility."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def save_training_config(self):
        """Save training configuration to JSON."""
        config = {
            "model_id": self.args.model_id,
            "dataset": {
                "name": self.data_curator.dataset_config.name,
                "dataset_id": self.data_curator.dataset_id,
                "preprocess_id": self.data_curator.preprocess_id
            },
            "peft_config": {
                "method": self.args.peft_method,
                "rank": self.args.peft_rank,
                "alpha": self.args.peft_alpha,
                "target_modules": self.args.peft_target,
                "trainable_parameter_count": self.args.trainable_parameter_count
            },
            "training_config": {
                "max_dataset_length": self.data_curator.max_dataset_length,
                "batch_size": self.data_curator.batch_size,
                "per_device_batch_size": self.data_curator.batch_size,
                "gradient_accumulation_steps": self.args.grad_accum_steps,
                "learning_rate": self.args.learning_rate,
                "num_epochs": self.args.num_train_epochs,
                "warmup_steps": self.args.warmup_steps
            },
            "model_name": self.train_model_name,
            "output_dir": self.training_model_dir,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs(self.training_model_dir, exist_ok=True)
        config_path = os.path.join(self.training_model_dir, "training_information.json")
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Training information saved to: {config_path}")

    def _extract_parameters(self):
        """Extract base layer and adapter parameters from checkpoint state."""
        if self.checkpoint_state is None:
            raise ValueError("Trainer checkpoint state is None. Make sure training has been completed.")
        
        # Get parameters object from checkpoint state
        parameters = self.checkpoint_state.parameters
        
        # Get all parameter names and objects by iterating over parameters
        # Each item is a tuple: (param_name, Parameter object)
        checkpoint_params = list(parameters)
        
        print(f"[INFO] Found {len(checkpoint_params)} parameters in checkpoint")
        
        # Extract base layer parameters (quantized weights, scales, zero_points)
        self._extract_base_layer_params(checkpoint_params)
        
        # Extract adapter parameters
        self._extract_adapter_params(checkpoint_params)

        self.create_merged_parameters()
    
    def _extract_base_layer_params(self, checkpoint_params: list):
        """Extract quantized base layer parameters."""
        for base_layer_name in self.peft_mapping.keys():
            base_params = {}

            if base_layer_name.startswith('base_model.model.model.'):
                base_layer_name = base_layer_name.replace('base_model.model.model.', 'backbone.model.')
            
            # Look for quantized weight, scale, and zero_point parameters
            weight_quantized_name = f"{base_layer_name}.weight_quantized"
            weight_scale_name = f"{base_layer_name}.weight_scale"
            weight_zero_point_name = f"{base_layer_name}.weight_zero_point"
            weight_noquantized_name = f"{base_layer_name}.weight"
            
            # Iterate through parameter tuples: (param_name, Parameter object)
            for param_name, param_obj in checkpoint_params:
                if param_name == weight_quantized_name:
                    base_params['weight_quantized'] = param_obj.data
                    print(f"[INFO] Found quantized weight: {param_name}")
                elif param_name == weight_scale_name:
                    base_params['x_scale'] = param_obj.data
                    print(f"[INFO] Found weight scale: {param_name}")
                elif param_name == weight_zero_point_name:
                    base_params['x_zero_point'] = param_obj.data
                    print(f"[INFO] Found weight zero point: {param_name}")
                elif param_name == weight_noquantized_name:
                    base_params['weight'] = param_obj.data
                    print(f"[INFO] Found non-quantized weights: {param_name}")
            
            if base_params:
                self.base_layer_params[base_layer_name] = base_params
                print(f"[INFO] Extracted base layer params for: {base_layer_name}")
            else:
                print(f"[WARNING] No quantized parameters found for base layer: {base_layer_name}")
    
    def _extract_adapter_params(self, checkpoint_params: list):
        """Extract adapter parameters for merging."""
        # Get all unique adapter names from the mapping

        self.adapter_params = {}
        for base_layer_name, adapter_names in self.peft_mapping.items():

            # Prefix renaming if needed
            if base_layer_name.startswith('base_model.model.model.'):
                base_layer_name = base_layer_name.replace('base_model.model.model.', 'backbone.model.')

            if base_layer_name not in self.adapter_params:
                self.adapter_params[base_layer_name] = {}

            for input_name, adapter_name in adapter_names.items():

                # Do not handle non string names, just add them to adapter params
                if not isinstance(adapter_name, str):
                    self.adapter_params[base_layer_name][input_name] = adapter_name
                    continue

                # Prefix renaming if needed
                if adapter_name.startswith('base_model.model.model.'):
                    adapter_name = adapter_name.replace('base_model.model.model.', 'backbone.model.')

                adapter_name = f'{adapter_name}.weight'
                # Iterate through parameter tuples: (param_name, Parameter object)
                for param_name, param_obj in checkpoint_params:

                    # Prefix renaming if needed
                    if param_name.startswith('base_model.model.model.'):
                        param_name = param_name.replace('base_model.model.model.', 'backbone.model.')

                    if adapter_name == param_name:

                        self.adapter_params[base_layer_name][input_name] = param_obj.data

                        print(f"[INFO] Found adapter param: {param_name}")
    
    def get_base_layer_params(self, base_layer_name: str) -> Dict[str, np.ndarray]:
        """
        Get quantized parameters for a specific base layer.
        
        Args:
            base_layer_name: Name of the base layer
            
        Returns:
            Dictionary containing weight_quantized, weight_scale, weight_zero_point
        """
        return self.base_layer_params.get(base_layer_name, {})
    
    def get_adapter_params(self, adapter_name: str) -> Dict[str, np.ndarray]:
        """
        Get parameters for a specific adapter.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Dictionary containing adapter parameters
        """
        return self.adapter_params.get(adapter_name, {})
    
    def get_mapping_for_base_layer(self, base_layer_name: str):
        """
        Get the PEFT mapping configuration for a specific base layer.
        
        Args:
            base_layer_name: Name of the base layer
            
        Returns:
            Dictionary containing adapter mappings for the base layer
        """
        return self.peft_mapping.get(base_layer_name, {})
    
    def create_merged_parameters(self):
        self.input_adapter_parameters = {}
        self.output_adapter_parameters = {}

        for base_name, base_params in self.base_layer_params.items():
            self.input_adapter_parameters[base_name] = {
                **base_params,
                **self.adapter_params[base_name]
            }

            self.output_adapter_parameters[base_name] = {}

            # Quantized weight output
            if 'weight_quantized' in self.input_adapter_parameters[base_name]:
                self.output_adapter_parameters[base_name] = {
                    'merged_weight_quantized': None,
                    'merged_zero_point': None,
                    'merged_scale': None
                }
            # Full precision weight output
            elif 'weight' in self.input_adapter_parameters[base_name]:
                self.output_adapter_parameters[base_name] = {
                    'merged_weight': None
                }

            # check for None, empty, or empty numpy arrays
            for key, value in self.input_adapter_parameters[base_name].items():
                if value is None:
                    print(f"[WARNING]: {base_name} -> {key} is None")

                elif isinstance(value, (list, tuple, dict)) and len(value) == 0:
                    print(f"[WARNING]: {base_name} -> {key} is an empty {type(value).__name__}")

                elif isinstance(value, np.ndarray) and value.size == 0:
                    print(f"[WARNING]: {base_name} -> {key} is an empty numpy array")

                # convert plain Python ints or floats to numpy arrays
                if isinstance(value, (int, np.integer)):
                    self.input_adapter_parameters[base_name][key] = np.array(value, dtype=np.int64)
                    #print(f"[INFO]: Converted {base_name} -> {key} to numpy int64 array")

                elif isinstance(value, (float, np.floating)):
                    self.input_adapter_parameters[base_name][key] = np.array(value, dtype=np.float32)
                    #print(f"[INFO]: Converted {base_name} -> {key} to numpy float32 array")

    def _load_peft_mapping(self):
        """Load the training configuration containing PEFT mapping."""
        try:
            
            self.peft_mapping = self.args.peft_mapping

            # 1. Populate self.merger_models
            self.merger_models = {}

            # get all .onnx files ending with merger_model.onnx or qmerger_model.onnx
            all_merger_files = [
                f for f in os.listdir(self.training_model_dir)
                if f.endswith("merger_model.onnx")
            ]

            for fname in all_merger_files:
                # extract method name from the file name
                # e.g. "lora_merger_model.onnx" → "lora"
                method_name = fname.split("_")[0]
                method_dict = self.merger_models.setdefault(method_name, {})

                full_path = os.path.join(self.training_model_dir, fname)

                if fname.endswith("qmerger_model.onnx"):
                    method_dict["quantized"] = full_path
                else:
                    method_dict["full_precision"] = full_path

                if not self.peft_mapping:
                    raise ValueError("No 'peft_mapping' found in training config")
        except FileNotFoundError:
            raise FileNotFoundError(f"Training config file not found: {self.training_config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in training config file: {self.training_config_path}")

    def _build_merger_models(self):
        """
        Builds the onnxruntime sessions for each merger model
        and stores them in self.merger_models[method_name]["quantized"/"full_precision"]
        """
        for method_name, models in self.merger_models.items():
            for precision, path in models.items():
                print(f"Loading {precision} merger model for method {method_name} from {path}")
                session = ort.InferenceSession(path)
                self.merger_models[method_name][precision] = session
    
    def clear_merger_models(self):
        """
        Cleanly releases and deletes all loaded merger ONNX inference sessions
        from memory.
        """
        if hasattr(self, "merger_models"):
            for method_name, merger_types in self.merger_models.items():
                for precision, session in merger_types.items():
                    if isinstance(session, ort.InferenceSession):
                        print(f"[INFO] Releasing inference session for method '{method_name}' ({precision})")
                        session._sess = None
                        del session
                # remove references from dictionary
                self.merger_models[method_name].clear()
            self.merger_models.clear()
        
        print("[INFO] All merger inference sessions cleared from memory.")

    def export_model_for_inference(self, merged_weight_quantized = True, save_directory: str = None):

        self.merged_weight_quantized = merged_weight_quantized
        self.merger_models = {}
        self.base_layer_params = {}
        self.adapter_params = {}
        self.input_adapter_parameters = {}
        self.output_adapter_parameters = {}

        # Instantiate peft mapping
        self._load_peft_mapping()

        if self.model is None:
            self.load_onnx_trainer()
        
        # Extract parameters from checkpoint
        self._extract_parameters()

        self._build_merger_models()

        for base_layer, input_layers in self.input_adapter_parameters.items():

            print(f"[DEBUG] Computing merge for base_layer: {base_layer}")
            
            # Print input layer keys and their shapes
            for key, val in input_layers.items():
                if isinstance(val, np.ndarray):
                    print(f"[DEBUG] Input '{key}' shape: {val.shape}")
                else:
                    print(f"[WARNING] Input '{key}' is not a numpy array, type={type(val)}")
            
            # Run the MARS merger technique
            if 'shared_A' in input_layers:
                self._run_merger_model(self.merger_models['mars'], base_layer, input_layers)
            # Run the LoRA merger technique
            elif 'adapter_A' in input_layers:
                # We do not need rank input here
                input_layers.pop("rank", None)
                self._run_merger_model(self.merger_models['lora'], base_layer, input_layers)

        print(f"\n[DEBUG] Finished merging. Output adapter parameters:")
        for base_layer, output_dict in self.output_adapter_parameters.items():
            print(f"  - Base layer: {base_layer}")
            for name, arr in output_dict.items():
                if isinstance(arr, np.ndarray):
                    print(f"      * {name}: shape={arr.shape}")
                else:
                    print(f"      * {name}: type={type(arr)} (not a numpy array)")
        
        # Save to disk
        if not save_directory:
            save_directory = os.path.join(self.training_model_dir, "merged")
        
        os.makedirs(save_directory, exist_ok=True)
        for base_layer, output_dict in self.output_adapter_parameters.items():
            save_path = os.path.join(save_directory, f"{base_layer}.npz")
            np.savez(save_path, **output_dict)
            print(f"[INFO] Saved merged parameters for {base_layer} to {save_path}")
    
    def _run_merger_model(self, merger_models: dict, base_layer: str, input_layers: dict):
        """
        Runs the quantized or full precision merger technique of the PEFT method.
        """

        session = None
        
        if 'weight_quantized' in input_layers:

            session = merger_models["quantized"]

            if self.merged_weight_quantized:
                merged_weight_quantized, merged_zero_point, merged_scale = session.run(None, input_layers)

                are_equal = np.array_equal(input_layers['weight_quantized'], merged_weight_quantized)
                print(f"weight quantized are exactly equal: {are_equal}")

                are_equal = np.array_equal(input_layers['x_scale'], merged_scale)
                print(f"x scale are exactly equal: {are_equal}")

                are_equal = np.array_equal(input_layers['x_zero_point'], merged_zero_point)
                print(f"zero point are exactly equal: {are_equal}")

                #print(input_layers)

                # Debug outputs
                print(f"[DEBUG] merged_weight_quantized shape: {merged_weight_quantized.shape if isinstance(merged_weight_quantized, np.ndarray) else 'not ndarray'}")
                print(f"[DEBUG] merged_zero_point shape: {merged_zero_point.shape if isinstance(merged_zero_point, np.ndarray) else 'not ndarray'}")
                print(f"[DEBUG] merged_scale shape: {merged_scale.shape if isinstance(merged_scale, np.ndarray) else 'not ndarray'}")
                
                if isinstance(merged_weight_quantized, np.ndarray) and merged_weight_quantized.size == 0:
                    print(f"[WARNING] merged_weight_quantized is empty!")
                if isinstance(merged_zero_point, np.ndarray) and merged_zero_point.size == 0:
                    print(f"[WARNING] merged_zero_point is empty!")
                if isinstance(merged_scale, np.ndarray) and merged_scale.size == 0:
                    print(f"[WARNING] merged_scale is empty!")

                self.output_adapter_parameters[base_layer] = {
                    'weight_quantized': merged_weight_quantized,
                    'weight_scale': merged_scale,
                    'weight_zero_point': merged_zero_point
                }
            else:

                merged_weight = session.run(None, input_layers)[0]
                
                # Debug output
                print(f"[DEBUG] merged_weight type: {type(merged_weight)}")
                if isinstance(merged_weight, np.ndarray):
                    print(f"[DEBUG] merged_weight shape: {merged_weight.shape}")
                    if merged_weight.size == 0:
                        print(f"[WARNING] merged_weight is empty!")
                else:
                    print(f"[WARNING] merged_weight is not a numpy array")

                self.output_adapter_parameters[base_layer] = {
                    'weight': merged_weight
                }
        
        elif 'weight' in input_layers:

            session = merger_models["full_precision"]

            merged_weight = session.run(None, input_layers)[0]
            
            # Debug output
            print(f"[DEBUG] merged_weight type: {type(merged_weight)}")
            if isinstance(merged_weight, np.ndarray):
                print(f"[DEBUG] merged_weight shape: {merged_weight.shape}")
                if merged_weight.size == 0:
                    print(f"[WARNING] merged_weight is empty!")
            else:
                print(f"[WARNING] merged_weight is not a numpy array")

            self.output_adapter_parameters[base_layer] = {
                'weight': merged_weight
            }

    def print_summary(self):
        """Print a summary of extracted parameters."""
        print("\n" + "="*50)
        print("PEFT Merge Validator Summary")
        print("="*50)
        
        print(f"Base layers found: {len(self.base_layer_params)}")
        for base_layer_name, params in self.base_layer_params.items():
            print(f"  - {base_layer_name}: {list(params.keys())}")
        
        print(f"\nAdapters found: {len(self.adapter_params)}")
        for adapter_name, params in self.adapter_params.items():
            print(f"  - {adapter_name}: {list(params.keys())}")
        
        print(f"\nPEFT mappings: {len(self.peft_mapping)}")

def check_duplicate_initializers(onnx_model_path):
    """
    Loads an ONNX model and checks if there are duplicate initializers
    (same name, shape, and values).
    """
    model = onnx.load(onnx_model_path)

    # Check for duplicate names
    name_counts = defaultdict(int)
    for initializer in model.graph.initializer:
        name_counts[initializer.name] += 1
        if "down_project" in initializer.name:
            print(initializer.name)
            print(initializer.dims)

    duplicate_names = [name for name, count in name_counts.items() if count > 1]
    if duplicate_names:
        print(f"[WARNING] Duplicate initializer names found: {duplicate_names}")
    
    # Check for duplicate initializers with same shape and values
    initializer_dict = defaultdict(list)
    for initializer in model.graph.initializer:
        shape = tuple(initializer.dims)
        values = onnx.numpy_helper.to_array(initializer)
        
        # Convert values to bytes for efficient comparison
        values_bytes = values.tobytes()
        
        initializer_dict[(shape, values_bytes)].append(initializer.name)

    duplicate_initializers = {key: names for key, names in initializer_dict.items() if len(names) > 1}

    if duplicate_initializers:
        print("[WARNING] Duplicate initializers found with identical shape and values:")
        for (shape, _), names in duplicate_initializers.items():
            print(f" - Shape: {shape}, Names: {names}")
    else:
        print("[INFO] No duplicate initializers with the same shape and values found.")

def parse_extra_options(extra_options: List[str]) -> Dict[str, str]:
    """
    Parse additional options in KEY=VALUE format into a dictionary.
    """
    options_dict = {}
    for option in extra_options:
        if "=" in option:
            key, value = option.split("=", 1)
            options_dict[key] = value
        else:
            raise ValueError(f"Invalid format for extra option '{option}'. Use KEY=VALUE format.")
        
    print(f"Extra options: {options_dict}")
    return options_dict

def load_config_from_file(config_file: str):
    """Load configurations from a YAML file into a dictionary."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="Validator for exported ONNX artifacts for on-device training.", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--model_id",
        type=str,
        help="Identifier for the model to be converted."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to configuration file to load additional options. This config file will overwrite all other arguments."
    )
    parser.add_argument(
        "--training_artifact_dir",
        type=str,
        help="Path to training artifact directory."
    )
    parser.add_argument(
        "--test_training_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ...
            """
            )
    )
    parser.add_argument(
        "--test_scheduler_config",
        type=str,
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help=textwrap.dedent("""\
         Key value pairs for various options. Currently supports:
            ...
            """
            )
    )
    args = parser.parse_args()

    user_train_generation_config = {}
    default_train_generation_config = {
        "trainFile": "",
        "peftMethod": "",
        "taskName": "",
        "numTrainEpochs": 1,
        "maxSteps": 4,
        "saveSteps": 50,
        "gradAccumSteps": 2,
        "removeLongSample": True,
        "maxSequenceLength": 512,
        "maxDatasetLength": 256,
        "datasetBatchSize": 64,
        "testRatio": 0.1,
        "split": True,
        "shuffle": True,
        "schedulerType": "cosine"
    }

    user_scheduler_generation_config = {}
    default_scheduler_generation_config = {
       "minLearningRate": 0,
       "cosineLearningRate": 0.0001,
       "warmupSteps": 10,
       "linearLearningRate": 0.0001,
       "startFactor": 1,
       "endFactor": 0.333
    }

    config_dict = None

    if args.config_file:
        config_dict = load_config_from_file(args.config_file)

        # Specific
        setattr(args, "model_id", config_dict[TRAIN_CONFIG]["model_id"])
        setattr(args, "training_artifact_dir", os.path.join(config_dict[ARTIFACT_CONFIG]["build_path"], "train"))
        setattr(args, "test_scheduler_config", config_dict[ARTIFACT_VALIDATOR_CONFIG]["test_training_config"]["schedulerOptions"])
        
        # Override any command-line argument with values from the config file
        for key, value in config_dict[ARTIFACT_VALIDATOR_CONFIG].items():
            
            # Convert to the correct type
            if hasattr(args, key):
                setattr(args, key, value)

    else:
        user_train_generation_config = parse_extra_options(args.test_training_config)
        args.test_training_config = {**default_train_generation_config, **user_train_generation_config}
        user_scheduler_generation_config = parse_extra_options(args.test_scheduler_config)
        args.test_scheduler_config = {**default_scheduler_generation_config, **user_scheduler_generation_config}

    return args


if __name__ == "__main__":
    
    trainer = ORTTrainer("build/train-tinyllama-lora-r8", load_from_state=False)

    trainer.train()

    #trainer.export_model_for_inference(merged_weight_quantized=False)