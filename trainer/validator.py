import onnx, time, os, gc, json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
import numpy as np
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.training import onnxblock, artifacts
from onnxruntime.training.api import CheckpointState, Module, Optimizer, LinearLRScheduler
import onnxruntime as rt
from onnxruntime import InferenceSession, SessionOptions
from datasets import load_dataset, Dataset, DatasetDict
from trainer.utils import process_sample_commonsense, process_sample_dolly

from torch.utils.data import DataLoader

def preload_dataset(dataset_id):

    dataset_ids = dataset_id.split("/")
    
    # Take local data
    if len(dataset_ids) >= 2 and dataset_ids[-2] == "data":

        filepath = dataset_id
        data = None

        if os.path.exists(f'./{dataset_id}.json'):
            filepath = f'./{dataset_id}.json'

            with open(filepath, 'r', encoding="utf-8") as f:
                data = json.load(f)

        elif os.path.exists(f'./{dataset_id}.jsonl'):
            filepath = f'./{dataset_id}.jsonl'

            with open(filepath, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(data)

        # Create a DatasetDict with the "train" split
        dataset_dict = DatasetDict({"train": dataset})

        return dataset_dict

    return load_dataset(dataset_id)

class CosineLRScheduler:
    """Cosine Learning Rate Scheduler for ONNX Runtime Training."""
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=0.0, initial_lr=0.001):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.current_step = 0

    def step(self):
        """Update learning rate with warmup + cosine decay."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup: Increase LR from 0 to initial_lr
            new_lr = (self.initial_lr * self.current_step) / self.warmup_steps
        else:
            # Cosine decay after warmup
            decay_step = self.current_step - self.warmup_steps
            decay_total = self.total_steps - self.warmup_steps
            cos_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_total))
            new_lr = self.min_lr + (self.initial_lr - self.min_lr) * cos_decay

        self.optimizer.set_learning_rate(new_lr)


class ORTDataCurator:

    def __init__(self,
                 model_id,
                 max_dataset_length = None,
                 remove_long_samples = True,
                 max_context_length=512,
                 test_ratio=0.1,
                 batch_size=4,
                 split=True,
                 shuffle=False) -> None:
        
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.environ['HF_TOKEN'])
        self.collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False, return_tensors="np")

        self.max_dataset_length = max_dataset_length
        self.max_context_length = max_context_length
        self.remove_long_samples = remove_long_samples

        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.split = split
        self.shuffle = shuffle

        self.dataset = None

    # Define preprocessing function for tokenization
    def prepare_dataset(self, dataset : Dataset, custom_preprocess = None):

        raw_columns = dataset["train"].column_names

        if self.split:
            test_size = self.test_ratio if self.max_dataset_length == None else int(self.max_dataset_length * self.test_ratio)
            train_size = (1 - self.test_ratio) if self.max_dataset_length == None else int(self.max_dataset_length * (1 - self.test_ratio))
            dataset = dataset["train"].train_test_split(test_size=test_size, train_size=train_size, shuffle=self.shuffle)    

        def process_sample(sample):
            
            if custom_preprocess:
                return custom_preprocess(sample, self.tokenizer)

            return self.tokenizer(sample, return_dict=True, tokenize=True, return_tensors="np", padding=True, add_generation_prompt=False)
        
        def filter_sample(sample):
            return len(sample["input_ids"]) < self.max_context_length

        # Convert the list of tokenized samples into a Dataset
        dataset = dataset.map(process_sample, batched=(self.batch_size > 1), batch_size=self.batch_size)

        if self.remove_long_samples != None:
            dataset = dataset.filter(filter_sample, batched=False)

        dataset = dataset.remove_columns(raw_columns)

        self.dataset = dataset

class ORTTrainingArguments:

    def __init__(self,
                export_inference=False,
                test_inference=False,
                test_evaluate=False,
                learning_rate=1e-3,
                min_learning_rate=0,
                max_sequence_length=100,
                num_train_epochs=1,
                warmup_steps=10,
                max_steps=10,
                save_steps=100,
                scheduler_type="linear",
                grad_accum_steps=4) -> None:
        
        self.export_inference = export_inference
        self.test_inference = test_inference
        self.test_evaluate = test_evaluate
        
        self.max_sequence_length = max_sequence_length

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.save_steps = save_steps
        self.scheduler_type = scheduler_type
        self.grad_accum_steps = grad_accum_steps


class ORTTrainer:

    def __init__(self,
                training_model_dir,
                args : ORTTrainingArguments,
                data_curator : ORTDataCurator,
                inference_model_path="inference_model.onnx",
                callbacks=None
                ) -> None:
        
        self.training_model_dir = training_model_dir
        self.args = args
        self.data_curator = data_curator
        self.inference_model_path = inference_model_path
        self.callbacks = callbacks

        self.model = None
        self.state = None
        self.optimizer = None
        self.tokenizer = None

    
    def set_scheduler_type(self, total_steps):

        if self.args.scheduler_type == "linear":
            return LinearLRScheduler(self.optimizer, self.args.warmup_steps, total_steps, initial_lr=self.args.learning_rate)
        elif self.args.scheduler_type == "cosine":
            return CosineLRScheduler(self.optimizer, self.args.warmup_steps, total_steps, min_lr=self.args.min_learning_rate, initial_lr=self.args.learning_rate)
        else:
            raise ValueError("Unsupported scheduler type. Use 'linear' or 'cosine'.")

    def train(self):
        """
        Checks the model if the outputs are training correctly as well as evaluation.
        Exports model for inference if needed or transfers the weights to an already existing inference model.

        - `test_inference` - runs the model through an example prompt
        - `test_evaluate` - tests the model evaluation
        - `transfer_weights` - instead of exporting for inference, we only copy the subset of updated weights to an already created inference model after training
        - `inference_model_path` - model path of an already existing inference model or one to create
        - `max_sequence_length` - length of text generation sequence
        """
        
        self.state = CheckpointState.load_checkpoint(f"{self.training_model_dir}/checkpoint")

        # TODO: Customize
        sess_options = SessionOptions()
        sess_options.enable_profiling = False
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_options.add_session_config_entry("session.inter_op.allow_spinning", "0")

        self.model = Module(f"{self.training_model_dir}/training_model.onnx", self.state, f"{self.training_model_dir}/eval_model.onnx", session_options=sess_options)
        self.optimizer = Optimizer(f"{self.training_model_dir}/optimizer_model.onnx", self.model)
        
        train_dataset = self.data_curator.dataset["train"]

        total_samples = len(train_dataset)
        steps_per_epoch = total_samples // self.data_curator.batch_size
        total_epoch_steps = self.args.num_train_epochs * steps_per_epoch

        if self.args.max_steps is not None:
            total_steps = self.args.max_steps
            print(f"Training for {self.args.max_steps} steps")
        else:
            total_steps = total_epoch_steps
            print(f"Training for {self.args.num_train_epochs} epochs ({total_steps} steps)")

        global_step = 0
        epoch = 0

        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.data_curator.batch_size,
            shuffle=True,
            collate_fn=self.data_curator.collator.numpy_call
        )

         # Calculate total steps
        steps_per_epoch = len(dataloader)
        total_epoch_steps = self.args.num_train_epochs * steps_per_epoch
        
        # Determine whether to use steps or epochs
        total_steps = self.args.max_steps if self.args.max_steps is not None else total_epoch_steps
        
        # Scheduler
        scheduler = self.set_scheduler_type(total_steps)

        # Main training loop
        global_step = 0
        epoch = 0
        pbar = tqdm(total=total_steps, desc="ONNX Runtime Training")
        
        accumulated_loss = 0.0

        while global_step < total_steps:
            for batch in dataloader:
                if global_step >= total_steps:
                    break

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

                accumulated_loss += forward[0]

                if (global_step + 1) % self.args.grad_accum_steps == 0:

                    self.optimizer.step()
                    self.model.lazy_reset_grad()
                    scheduler.step()

                    avg_loss = accumulated_loss / self.args.grad_accum_steps

                    print(f"[INFO] Training loss result: {avg_loss}")
                    print(f"[INFO] LR: {self.optimizer.get_learning_rate()}")

                    # Reset accumulated loss
                    accumulated_loss = 0.0

                global_step += 1
                pbar.update(1)
                
                if global_step >= total_steps:
                    break
                    
            epoch += 1

        pbar.close()


        if self.args.export_inference:

            exclude_nodes = ["loss"]

            # Model inference: we want to get only logits and hidden states for decoding
            self.model.export_model_for_inferencing(f"{self.training_model_dir}/{self.inference_model_path}", [ out_name for out_name in self.model.output_names() if out_name not in exclude_nodes])

            del self.model
            del self.state
            del self.optimizer
            gc.collect()

    def create_position_ids(self, input_ids: np.ndarray, padding_idx: int = 0):
        # Create a mask where input_ids are not padding
        mask = input_ids != padding_idx

        # Create position indices for each sequence
        position_ids = np.cumsum(mask, axis=1) - mask

        return position_ids

    def save_checkpoint(self):
        CheckpointState.save_checkpoint(self.state, f"{self.training_model_dir}/checkpoint")



if __name__ == "__main__":
    data_cur = ORTDataCurator("TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_dataset_length=100, test_ratio=0.1)

    dataset_id = "data/commonsense"

    ds = preload_dataset(dataset_id)

    data_cur.prepare_dataset(ds, custom_preprocess=process_sample_commonsense)
    args = ORTTrainingArguments(export_inference=True, grad_accum_steps=2, max_steps=30, scheduler_type="cosine")
    trainer = ORTTrainer("build/train", args, data_cur)

    trainer.train()