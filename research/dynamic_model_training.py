import gc
import math
import os

import torch

from peft_models.mars.config import MarsConfig
from research.model_training import count_trainable_parameters, create_peft_model, get_training_args, list_trainable_layers, load_peft_model, set_manual_seed
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import Trainer, TrainerState

from research.visualization_trainer import PEFTUsageCallback

def compute_max_steps(train_dataloader, grad_accum_steps=1, epochs=1):
    len_dataloader = len(train_dataloader)
    num_update_steps_per_epoch = len_dataloader // grad_accum_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    return math.ceil(epochs * num_update_steps_per_epoch)

def train_peft_model(model, train_dataset, eval_dataset, data_collator, output_dir, training_args, peft_method, resume=None):
    
    # Manual seed
    set_manual_seed(42)

    # TODO: Trainer does not log out validation loss
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[PEFTUsageCallback()]
    )
    
    if resume is not None:
        # This loads in optimizer and scheduler, however if the training ends and LR reaches 0
        # (if using cosine or other decaying LR scheduler), we need to somehow save the learning rate and restart from there

        print("Loading in optimizer / scheduler / state...")
        trainer.create_optimizer_and_scheduler(num_training_steps=compute_max_steps(train_dataset))

        # TODO: If we modify weights, we cannot load the previous optimizer and scheduler, difference in parameter groups
        trainer._load_optimizer_and_scheduler(resume)
        trainer._load_rng_state(resume)
        trainer.state = TrainerState.load_from_json(os.path.join(output_dir, f"trainer_state.json"))

    trainer.train()

    if peft_method.endswith("mars"):

        output_dir = f"{output_dir}/mars_checkpoint"

        model.base_model.save_pretrained(output_dir)
        trainer.save_state()

        # Save optimizer and scheduler
        trainer._save_optimizer_and_scheduler(output_dir)

        # Save RNG state
        trainer._save_rng_state(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(trainer.args, os.path.join(output_dir, "training_args.bin"))

    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return output_dir

def train_joint_models(base_model, dataset_dict, datacol, lora_target, peft_method, num_cycles = 1, data_ratios = None, **peft_config):

    # TODO: Manual config
    # Create PEFT models with different ranks
    rank_a = 8
    rank_b = 4
    rank_c = rank_a + rank_b  # Target model has summed ranks

    peft_config["mixture"] = False
    peft_config["only_mixtures"] = False

    output_dir_a = "peft_model_a"
    output_dir_b = "peft_model_b"
    output_dir_c = "peft_model_c"

    total_data = len(dataset_dict["train"])
    cycle_data_size = total_data // num_cycles
    mixture_layers = {}

    # TODO: Define new learning rates for each of the cycles, or create a global learning rate
    # Learning rates for each of the models, the last learning rate should be remembered and then reused as the initial learning rate
    # In case of cosine decaying, the last learning rate always drops to 0.
    # Optimizer states are loaded from the last cycle checkpoint
    lrs = [
        3e-4,
        3e-4,
        3e-4
    ]

    for cycle in range(num_cycles):
        print(f"Starting cycle {cycle + 1}/{num_cycles}...")
        
        # Split dataset for this cycle
        start_idx = cycle * cycle_data_size
        end_idx = start_idx + cycle_data_size
        train_dataset = dataset_dict["train"].select(range(start_idx, end_idx))
        
        # Apply data ratios if provided
        if data_ratios:
            train_a_size = int(len(train_dataset) * data_ratios[0])
            train_b_size = int(len(train_dataset) * data_ratios[1])
        else:
            train_a_size = len(train_dataset) // 3
            train_b_size = len(train_dataset) // 3

        train_dataset_a = train_dataset.select(range(train_a_size))
        train_dataset_b = train_dataset.select(range(train_a_size, train_a_size + train_b_size))
        train_dataset_c = train_dataset.select(range(train_a_size + train_b_size, len(train_dataset)))
        test_dataset = dataset_dict["test"]
        
        # Create / load model A
        if peft_method == "joint_mars":
            peft_config["subspace"] = (rank_a, 1)
        
        # Update scheduler
        scheduler_args = get_cosine_scheduler(cycle, last_min_lr=lrs[0])
        lrs[0] = scheduler_args["eta_min"]

        print(scheduler_args)

        if cycle > 0:
            # Load from previous checkpoint
            if peft_method.endswith("lora"):
                checkpoint_path = get_latest_checkpoint(output_dir_a)
            else:
                checkpoint_path = f"{output_dir_a}/mars_checkpoint"
            model_a = load_peft_model(base_model, checkpoint_path, peft_method, **peft_config)
            training_args = get_training_args(output_dir_a, peft_method, scheduler_args, resume_from_checkpoint=checkpoint_path)

            # Update mixture layers if needed
            if peft_config["mixture"] and mixture_layers:
                print("Updating mixture layers...")
                model_a = update_layers(model_a, mixture_layers, [rank_a, rank_b], extract_index=0, only_mixture=peft_config["only_mixtures"])
            resume = checkpoint_path
        else:
            # Create a new PEFT model
            model_a = create_peft_model(base_model, lora_target, peft_method, rank_a, **peft_config)
            training_args = get_training_args(output_dir_a, peft_method, scheduler_args)
            resume = None

        print("Starting Model A training...")

        list_trainable_layers(model_a)
        print(count_trainable_parameters(model_a))
        # Train model A
        train_peft_model(model_a, train_dataset_a, test_dataset, datacol, output_dir_a, training_args, peft_method, resume)

        # Clear model from memory
        model_a.to("cpu")
        del model_a
        model_a = None
        torch.cuda.empty_cache()
        gc.collect()

        # Create / load model B
        if peft_method == "joint_mars":
            peft_config["subspace"] = (rank_b, 2)
        
        # Update scheduler
        scheduler_args = get_cosine_scheduler(cycle, last_min_lr=lrs[1])
        lrs[1] = scheduler_args["eta_min"]

        print(scheduler_args)

        if cycle > 0:
            # Load from previous checkpoint
            if peft_method.endswith("lora"):
                checkpoint_path = get_latest_checkpoint(output_dir_b)
            else:
                checkpoint_path = f"{output_dir_b}/mars_checkpoint"
            model_b = load_peft_model(base_model, checkpoint_path, peft_method, **peft_config)
            training_args = get_training_args(output_dir_b, peft_method, scheduler_args, resume_from_checkpoint=checkpoint_path)

            # Update mixture layers if needed
            if peft_config["mixture"] and mixture_layers:
                print("Updating mixture layers...")
                model_b = update_layers(model_b, mixture_layers, [rank_a, rank_b], extract_index=1, only_mixture=peft_config["only_mixtures"])
            resume = checkpoint_path
        else:
            # Create a new PEFT model
            model_b = create_peft_model(base_model, lora_target, peft_method, rank_b, **peft_config)
            training_args = get_training_args(output_dir_b, peft_method, scheduler_args)
            resume = None

        print("Starting Model B training...")

        # Train model B
        train_peft_model(model_b, train_dataset_b, test_dataset, datacol, output_dir_b, training_args, peft_method, resume)

        # Clear model from memory
        model_b.to("cpu")
        del model_b
        model_b = None
        del mixture_layers
        mixture_layers = None
        torch.cuda.empty_cache()
        gc.collect()

        # Create / load joint model C
        if peft_method == "joint_mars":
            peft_config["subspace"] = (rank_c, 3)

        # Update scheduler
        scheduler_args = get_cosine_scheduler(cycle, last_min_lr=lrs[2])
        lrs[2] = scheduler_args["eta_min"]

        print(scheduler_args)

        if cycle > 0:
            # Load from previous checkpoint
            if peft_method.endswith("lora"):
                checkpoint_path = get_latest_checkpoint(output_dir_c)
            else:
                checkpoint_path = f"{output_dir_c}/mars_checkpoint"
            model_c = create_joint_model(base_model, checkpoint_path_a, checkpoint_path_b, output_dir_c, lora_target, peft_method, ranks=[rank_a, rank_b], resume=True, **peft_config)
            training_args = get_training_args(output_dir_c, peft_method, scheduler_args, resume_from_checkpoint=checkpoint_path)
            resume = checkpoint_path
        else:
            # Create a new PEFT model
            # Load from previous checkpoints of model A and B
            if peft_method.endswith("lora"):
                checkpoint_path_a = get_latest_checkpoint(output_dir_a)
                checkpoint_path_b = get_latest_checkpoint(output_dir_b)
            else:
                checkpoint_path_a = f"{output_dir_a}/mars_checkpoint"
                checkpoint_path_b = f"{output_dir_b}/mars_checkpoint"
            model_c = create_joint_model(base_model, checkpoint_path_a, checkpoint_path_b, output_dir_c, lora_target, peft_method, ranks=[rank_a, rank_b], **peft_config)
            training_args = get_training_args(output_dir_c, peft_method, scheduler_args)
            resume = None
        
        print("Starting joint Model C training...")

        train_peft_model(model_c, train_dataset_c, test_dataset, datacol, output_dir_c, training_args, peft_method, resume=resume)

        # Offload the mixture layers after training the joint model
        if peft_config["mixture"]:
            # Make a new dict and delete old one
            mixture_layers = offload_layers(model_c, only_mixture=peft_config["only_mixtures"])
        
        # Clear model from memory
        model_c.to("cpu")
        del model_c
        model_c = None
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Finished cycle {cycle + 1}/{num_cycles}.")

def get_latest_checkpoint(output_dir):
    """Get the latest checkpoint path from the training output directory."""
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]

    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    return os.path.join(output_dir, latest_checkpoint)

def offload_layers(model, only_mixture=True):
    
    mixture_layers = {}
    
    for name, param in model.named_parameters():
        if only_mixture and "mixture" in name:
            #print(f"Offloading {name}")
            mixture_layers[name] = param.clone().detach().to("cpu")
        elif not only_mixture and param.requires_grad:
            #print(f"Offloading {name}")
            mixture_layers[name] = param.clone().detach().to("cpu")

    print(f"Extracted {len(mixture_layers)} mixture matrices from the model.")
    return mixture_layers

def update_layers(model, mixture_layers, ranks, extract_index=0, only_mixture=True):
    new_state_dict = {}

    for name, mixture_matrix in mixture_layers.items():
        #r_total = sum(ranks)  # Total rank should match the matrix size
        #assert mixture_matrix.shape == (r_total, r_total), f"Invalid shape for {name}: {mixture_matrix.shape}"

        start = 0
        for i, r in enumerate(ranks):
            if extract_index == i:

                if "mixture" in name:
                    new_state_dict[name] = mixture_matrix[start:start + r, start:start + r]

                # Replace only mixture layers
                if only_mixture:
                    continue
                
                # Replace all the B layers as well
                if "mixture" not in name:
                    new_state_dict[name] = mixture_matrix[:, start:start + r]

            start += r

    # Load extracted blocks into the model's state dictionary
    model_state = model.state_dict()
    model_state.update(new_state_dict)
    model.load_state_dict(model_state, strict=False)
    return model

def create_joint_model(base_model, model_a_dir, model_b_dir, model_c_dir, lora_target, peft_method, ranks, resume=False, **peft_config):
    """
    Creates Model C with LoRA rank (r3 = r1 + r2) and merges adapters from Model A and Model B.

    Args:
        model_a_dir (str): Path to Model A's saved adapter.
        model_b_dir (str): Path to Model B's saved adapter.
        model_c_dir (str): Output path for merged Model C.
        base_model (AutoModelForCausalLM): Base model (loaded once to avoid redundancy).
        r1 (int): Rank of Model A's LoRA adapter.
        r2 (int): Rank of Model B's LoRA adapter.

    Returns:
        None
    """
    
    # Define Model C with summed LoRA rank
    r = sum(ranks)

    if not resume:
        if peft_method.endswith("lora"):
            joint_peft_config = LoraConfig(
                r=r,
                lora_alpha=r,
                target_modules=lora_target,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(base_model, joint_peft_config)
            model.save_pretrained(model_c_dir)

        elif peft_method.endswith("mars"):
            joint_peft_config = MarsConfig(
                peft_type="MARS",
                r=r,
                mixture=peft_config.get("mixture", False),
                subspace=peft_config.get("subspace", (r, 3)),
                target_modules=lora_target,  # Target specific model layers
                task_type=None
            )

            model = get_peft_model(base_model, joint_peft_config, adapter_name="mars")
            model.base_model.save_pretrained(model_c_dir)
    else:
        model = load_peft_model(base_model, model_c_dir, peft_method, **peft_config)

    # Load LoRA adapters for Model A & Model B
    adapter_a = load_file(os.path.join(model_a_dir, "adapter_model.safetensors"))
    adapter_b = load_file(os.path.join(model_b_dir, "adapter_model.safetensors"))

    # Merge adapters by **concatenating along rank dimension**
    merged_adapters = {}
    for key in adapter_a.keys():
        # Ignore mixtures
        if key in adapter_b and "mixture" not in key:
            tensor_a = adapter_a[key]
            tensor_b = adapter_b[key]

            # Check which dimension matches
            if tensor_a.shape[1] == tensor_b.shape[1]:  # Match in dim=1
                concat_dim = 0
            elif tensor_a.shape[0] == tensor_b.shape[0]:  # Match in dim=0
                concat_dim = 1
            else:
                raise ValueError(f"Cannot combine {key}, incompatible shapes {tensor_a.shape} vs {tensor_b.shape}")
            
            # Both models have this key -> Concatenate tensors along rank dimension
            merged_adapters[key] = torch.cat([adapter_a[key], adapter_b[key]], dim=concat_dim)

    # Save merged adapter
    save_file(merged_adapters, os.path.join(model_c_dir, "adapter_model.safetensors"))

    # Load merged adapters
    if peft_method.endswith("lora"):
        model.model.load_state_dict(merged_adapters, strict=False)
    elif peft_method.endswith("mars"):
        model.base_model.model.load_state_dict(merged_adapters, strict=False)

    print(f"Successfully combined adapters into {model_c_dir}.")

    del adapter_a
    adapter_a = None
    del adapter_b
    adapter_b = None
    del merged_adapters
    merged_adapters = None
    gc.collect()

    return model

def get_cosine_scheduler(cycle, last_min_lr, initial_warmup_ratio=0.1, warmup_decay=0.8, eta_min_decay=0.2):
    """Dynamically adjust warmup steps and eta_min per cycle."""

    # Adjust warmup proportionally
    warmup_ratio = initial_warmup_ratio * (warmup_decay ** cycle)

    # Reduce min LR progressively
    eta_min = last_min_lr * eta_min_decay  # Decrease eta_min by 20% per cycle
    eta_min = max(eta_min, 1e-7)  # Ensure eta_min doesn’t go too low
    
    return  {
        "initial_lr": last_min_lr,
        "warmup_ratio": warmup_ratio,
        "eta_min": eta_min,  # Lower minimum LR in each cycle
    }