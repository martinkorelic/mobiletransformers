from dataclasses import dataclass
import inspect
from typing import Dict, List
from deepeval.benchmarks.hellaswag.template import HellaSwagTemplate
from deepeval.benchmarks.bool_q.template import BoolQTemplate
from deepeval.benchmarks.arc.template import ARCTemplate
from deepeval.benchmarks.logi_qa.template import LogiQATemplate
from deepeval.benchmarks.winogrande.template import WinograndeTemplate
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from peft.tuners.lora import LoraLayer

@dataclass
class DataCollatorForSupervisedDataset:
    """Dynamically pads input sequences for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: List[Dict], return_tensors="pt") -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        # Convert to tensors
        input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
        labels = [torch.tensor(x, dtype=torch.long) for x in labels]

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id  # Default to EOS if PAD is missing

        # Pad sequences dynamically
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Construct attention mask dynamically: 1 for non-pad tokens, 0 for pad tokens
        attention_mask = input_ids.ne(pad_token_id).long()

        print(input_ids)
        print(labels)
        print(attention_mask)

        raise ValueError("STOP")

        # Convert to requested tensor format
        if return_tensors == "np":
            return {
                "input_ids": input_ids.numpy(),
                "labels": labels.numpy(),
                "attention_mask": attention_mask.numpy()
            }
        elif return_tensors == "pt":
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
        else:
            raise ValueError(f"return_tensors must be 'pt' or 'np', got {return_tensors}")
        
    def numpy_call(self, instances: List[Dict]) -> Dict[str, np.ndarray]:
        """Convenience method that returns NumPy arrays."""
        return self.__call__(instances, return_tensors="np")
    
    def pytorch_call(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convenience method that returns PyTorch tensors."""
        return self.__call__(instances, return_tensors="pt")


def process_sample_minirecommendation(samples, tokenizer, batched=True):
    
    def format_question(data_point):
        """Format the recommendation prompt"""
        user_query = data_point["prompt"]
        category = data_point["category"]
        
        # Build the formatted question
        formatted = f"Recommend best actions based on this user query: {user_query}"
        
        return formatted
    
    def format_answer(data_point):
        """Format the recommendation answer"""
        return data_point["recommendation"]

    def generate_prompt(data_point):
        question = format_question(data_point) + "\n\nAnswer: "
        answer = format_answer(data_point)
        
        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]

        # Concatenate the token sequences
        input_ids = torch.cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(question_tokens)] = -100

        return (
            input_ids.squeeze(0),
            labels.squeeze(0)
        )
    
    if batched:
        batch = {
            "input_ids": [],
            "labels": []
        }
        for i in range(len(samples["prompt"])):
            sample = {
                "type": samples["type"][i],
                "category": samples["category"][i],
                "prompt": samples["prompt"][i],
                "recommendation": samples["recommendation"][i]
            }

            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
        
        return batch
    else:
        tk = generate_prompt(samples)

    return tk

def process_sample_minipersonalqa(samples, tokenizer, batched=True):
    
    def format_question(data_point):
        """Format the question with multiple choice options"""
        question_text = data_point["question"]
        choices = data_point["choices"]
        
        # Build the formatted question
        formatted = f"Question: {question_text}\n\n"
        for choice_key, choice_value in choices.items():
            formatted += f"{choice_key}: {choice_value}\n"
            
        return formatted
    
    def format_answer(data_point):
        """Format just the answer"""
        return data_point["correct_answer"]

    def generate_prompt(data_point):
        question = format_question(data_point) + "\n\nAnswer: "
        answer = format_answer(data_point)
        
        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]

        # Concatenate the token sequences
        input_ids = torch.cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(question_tokens)] = -100

        return (
            input_ids.squeeze(0),
            labels.squeeze(0)
        )
    
    if batched:
        batch = {
            "input_ids": [],
            "labels": []
        }
        for i in range(len(samples["question"])):
            sample = {
                "type": samples["type"][i],
                "category": samples["category"][i],
                "question": samples["question"][i],
                "choices": samples["choices"][i],
                "correct_answer": samples["correct_answer"][i]
            }

            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
        
        return batch
    else:
        tk = generate_prompt(samples)

    return tk

def process_sample_winogrande_deepeval(samples, tokenizer, batched=True):

    def generate_prompt(data_point):

        question = WinograndeTemplate.format_question(data_point, include_answer=False) + "\n\n "
        answer = WinograndeTemplate.format_answer(data_point)
        
        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question}, 
                {"role": "assistant", "content": " {}\n\n".format(WinograndeTemplate.format_answer(data_point))}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]

        # Concatenate the token sequences
        input_ids = torch.cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(question_tokens)] = -100

        return (
            input_ids.squeeze(0),
            labels.squeeze(0)
        )
    
    if batched:
        batch = {
            "input_ids": [],
            "labels": []
        }
        for i in range(len(samples["sentence"])):
            sample = {
                "sentence": samples["sentence"][i],
                "option1": samples["option1"][i],
                "option2": samples["option2"][i],
                "answer": samples["answer"][i]
            }

            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
        
        return batch
    else:
        tk = generate_prompt(samples)

    return tk

def process_sample_logiqa_deepeval(samples, tokenizer, batched=True):

    def generate_prompt(data_point):

        question = LogiQATemplate.format_question(data_point) + "\n\n "
        answer = LogiQATemplate.format_output(data_point)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question}, 
                {"role": "assistant", "content": " {}\n\n".format(answer)}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]

        # Concatenate the token sequences
        input_ids = torch.cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(question_tokens)] = -100

        return (
            input_ids.squeeze(0),
            labels.squeeze(0)
        )

    if batched:
        batch = {
            "input_ids": [],
            "labels": []
        }
        for i in range(len(samples["question"])):
            sample = {
                "question": samples["question"][i],
                "text": samples["text"][i],
                "options": samples["options"][i],
                "answer": samples["answer"][i]
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
        
        return batch
    else:
        tp = generate_prompt(samples)
        tk = {
            "input_ids": tp[0],
            "labels": tp[1]
        }

    return tk

def process_sample_arc_deepeval(samples, tokenizer, batched=True):

    def generate_prompt(data_point):

        question = ARCTemplate.format_question(data_point, include_answer=False) + "\n\n "
        answer = ARCTemplate.format_answer(data_point)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question}, 
                {"role": "assistant", "content": " {}\n\n".format(ARCTemplate.format_answer(data_point))}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]

        # Concatenate the token sequences
        input_ids = torch.cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(question_tokens)] = -100

        return (
            input_ids.squeeze(0),
            labels.squeeze(0)
        )
    
    if batched:
        batch = {
            "input_ids": [],
            "labels": []
        }
        for i in range(len(samples["question"])):
            sample = {
                "question": samples["question"][i],
                "choices": samples["choices"][i],
                "answerKey": samples["answerKey"][i]
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
        
        return batch
    else:
        tk = generate_prompt(samples)

    return tk

def process_sample_boolq_deepeval(samples, tokenizer, batched=True):

    def generate_prompt(data_point):

        question = BoolQTemplate.format_question(data_point) + "\n\n "
        answer = BoolQTemplate.format_answer(data_point)
        
        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question}, 
                {"role": "assistant", "content": " {}\n\n".format(answer)}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]

        # Concatenate the token sequences
        input_ids = torch.cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(question_tokens)] = -100

        return (
            input_ids.squeeze(0),
            labels.squeeze(0)
        )
    
    if batched:
        batch = {
            "input_ids": [],
            "labels": []
        }
        for i in range(len(samples["question"])):
            sample = {
                "question": samples["question"][i],
                "passage": samples["passage"][i],
                "answer": samples["answer"][i]
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
        
        return batch
    else:
        tk = generate_prompt(samples)

    return tk

def process_sample_hellaswag_deepeval(samples, tokenizer, batched=True):

    def generate_prompt(data_point):

        base_prompt = f'The following are multiple choice sentence completion problems about {data_point["activity_label"]}.\n\n'
        choices = ["A", "B", "C", "D"]
    
        if tokenizer.chat_template is not None:
            
            gen_output = HellaSwagTemplate.format_question(
                data_point,
                include_answer=False
            )
            messages = [
                {"role": "user", "content": base_prompt + gen_output}, 
                {"role": "assistant", "content": " {}\n\n".format(choices[int(data_point["label"])])}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)
        
        question = base_prompt + HellaSwagTemplate.format_question(
            data_point,
            include_answer=False
        ) + "\n\n "

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]

        answer = "{}".format(choices[int(data_point["label"])])
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)["input_ids"][0]

        input_ids = torch.cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[:len(question_tokens)] = -100

        return (
            input_ids.squeeze(0),
            labels.squeeze(0)
        )
    
    if batched:
        batch = {
            "input_ids": [],
            "labels": []
        }
        for i in range(len(samples["ctx"])):
            sample = {
                "ctx": samples["ctx"][i],
                "endings": samples["endings"][i],
                "label": samples["label"][i],
                "activity_label": samples["activity_label"][i]
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
        
        return batch
    else:
        tp = generate_prompt(samples)
        tk = {
            "input_ids": tp[0],
            "labels": tp[1]
        }

    return tk

def process_sample_dolly(sample, tokenizer):

    chat = [
        {"role": "user", "content": sample['instruction']},
        {"role": "assistant", "content": sample['response']}
    ]

    # TODO: This is wrong formatting
    if sample['context']:
        chat.insert(0, {"role": "system", "content": sample['context']})

    # Tokenize the prompt text
    text = tokenizer.apply_chat_template(chat, return_dict=True, tokenize=True, return_tensors="pt", padding=True, add_generation_prompt=False)
    return {
        "input_ids": text["input_ids"][0],
        "attention_mask": text["attention_mask"][0]
    }

def process_sample_alpaca(sample, tokenizer):

    def prompt_no_input(row):
        return ("Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)


    def prompt_input(row):
        return ("Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(row)
 
    chat = ""

    if len(sample['input']) == 0:
        chat = prompt_no_input(sample)
    else:
        chat = prompt_input(sample)

    # Tokenize the prompt text
    text = tokenizer(chat, return_tensors="pt", padding=True)
    return text


def process_sample_hellaswag(samples, tokenizer, batched=True):
    def generate_prompt(data_point):

        endings = "\n".join([ f'{i+1}. {e}' for i, e in enumerate(data_point["endings"])])

        return inspect.cleandoc(f"""
        Context: {data_point['ctx']}

        Options:
        {endings}

        Which option best completes the context?
        Answer: {data_point['label'] + 1}
        """).strip()
    
    if batched:
        text = [
            generate_prompt({
                "ctx": samples["ctx"][i],
                "endings": samples["endings"][i],
                "label": samples["label"][i]
            })
            for i in range(len(list(samples.values())[0]))
        ]
    else:
        text = generate_prompt(samples)
    
    tk = tokenizer(text, return_tensors="pt", padding=True)
    return tk

def taskname_to_deepeval_preprocess_function(preprocess_id):

    if preprocess_id == "hellaswag":
        return process_sample_hellaswag_deepeval
    elif preprocess_id == "boolq":
        return process_sample_boolq_deepeval
    elif preprocess_id == "arc_e" or preprocess_id == "arc_c":
        return process_sample_arc_deepeval
    elif preprocess_id == "logiqa":
        return process_sample_logiqa_deepeval
    elif preprocess_id == "winogrande":
        return process_sample_winogrande_deepeval
    elif preprocess_id == "mini_personalqa":
        return process_sample_minipersonalqa
    elif preprocess_id == 'mini_recommendation':
        print("USING RECOMMENDATRION")
        return process_sample_minirecommendation

    return None

def create_mars_adapter_mapping(model, shared_qkv=['q', 'k', 'v'], shared_mlp_enabled=True):
    """
    Create a JSON mapping of base layers to their corresponding adapters.
    Handles shared modules by their object identity and deduplicates them.

    NOTE: There could be problems with mapping with this function, depending on model architectures and position of the layers.
    
    Args:
        model: PyTorch model with PEFT adapters
        
    Returns:
        dict: Mapping of base layer names to their adapter configurations
    """
    mapping = {}
    
    # Track unique modules by their object id to handle shared modules
    module_id_to_name = {}
    
    def register_unique_module(module, full_name):
        """Register a module and return the canonical name for shared modules"""
        module_id = id(module)
        if module_id in module_id_to_name:
            # This module is shared, return the canonical name
            return module_id_to_name[module_id]
        else:
            # First time seeing this module
            module_id_to_name[module_id] = full_name
            return full_name
    
    # First pass: collect all modules and their paths
    all_modules = {}
    for name, module in model.named_modules():
        all_modules[name] = module
    
    # Find all base layers and their parent contexts
    base_layer_contexts = {}
    for name, module in all_modules.items():
        if "base_layer" in name:
            # Get parent path (everything before .base_layer)
            parent_path = name.rsplit('.base_layer', 1)[0]
            base_layer_contexts[name] = parent_path
    
    current_shared_mlp_name = None
    shared_mlp_counter = 0
    current_inter_mlp_name = None
    current_shared_qkv_name = None
    shared_qkv_counter = 0
    current_inter_qkv_name = None

    # For each base layer, find its adapters
    for base_layer_name, parent_path in base_layer_contexts.items():
        adapters = {}

        # Prefix renaming if needed
        if base_layer_name.startswith('base_model.model.model.'):
            base_layer_name = base_layer_name.replace('base_model.model.model.', 'backbone.model.')
        
        # Look for adapters in the parent context
        for module_name, module in all_modules.items():
            # Skip if not in the same parent context
            if not module_name.startswith(parent_path + "."):
                continue
            
            # Prefix renaming if needed
            if module_name.startswith('base_model.model.model.'):
                module_name.replace('base_model.model.model.', 'backbone.model.')

            # Get the relative path from parent
            relative_path = module_name[len(parent_path) + 1:]
            
            # Apply categorization rules based on path patterns
            # Rule 1: shared_*.mars_down_* -> "shared_A"

            if relative_path.startswith("shared_") and ".mars_down_" in relative_path:
                canonical_name = register_unique_module(module, module_name)
                adapters["shared_A"] = canonical_name

                if relative_path.startswith("shared_mlp"):
                    current_shared_mlp_name = module_name
                    
                elif relative_path.startswith("shared_qkv"):
                    current_shared_qkv_name = module_name
            
            # Rule 2: shared_*.mars -> "intermediate" (direct mars in shared)
            elif relative_path.startswith("shared_") and relative_path.endswith(".mars"):
                canonical_name = register_unique_module(module, module_name)
                adapters["intermediate"] = canonical_name

                if relative_path.startswith("shared_mlp"):
                    current_inter_mlp_name = module_name
                    shared_mlp_counter = 0
                    adapters["adapter_index"] = 0
                    shared_mlp_counter += 1
                elif relative_path.startswith("shared_qkv"):
                    current_inter_qkv_name = module_name
                    shared_qkv_counter = 0
                    adapters["adapter_index"] = 0
                    shared_qkv_counter += 1
            
            # Rule 3: up_project.mars -> "adapter_B"
            elif relative_path == "up_project.mars":
                canonical_name = register_unique_module(module, module_name)
                adapters["adapter_B"] = canonical_name

                if hasattr(module, 'rank'):
                    adapters["rank"] = int(module.rank)
                else:
                    print(f'[WARNING] Could not find rank in {module_name}')
                if hasattr(module, 'alpha'):
                    adapters["alpha"] = float(module.alpha)

            # Rule 4: down_project.mars -> "adapter_A"
            elif relative_path == "down_project.mars":
                canonical_name = register_unique_module(module, module_name)
                adapters["adapter_A"] = canonical_name
        
        # Check if we need to add pointer to shared or intermediate layer
        if "shared_A" not in adapters and "adapter_A" not in adapters:
            if ('q' in shared_qkv and 'q_proj' in base_layer_name) or ('v' in shared_qkv and 'v_proj' in base_layer_name) or ('k' in shared_qkv and 'k_proj' in base_layer_name):
                adapters["shared_A"] = current_shared_qkv_name
                adapters["intermediate"] = current_inter_qkv_name
                adapters["adapter_index"] = shared_qkv_counter
                shared_qkv_counter += 1
            elif shared_mlp_enabled and ('gate_proj' in base_layer_name or 'up_proj' in base_layer_name):
                adapters["shared_A"] = current_shared_mlp_name
                adapters["intermediate"] = current_inter_mlp_name
                adapters["adapter_index"] = shared_mlp_counter
                shared_mlp_counter += 1

        if adapters:
            mapping[base_layer_name] = adapters

    #with open('base_mapping.json', 'w') as f:
    #    json.dump(mapping, f)

    return mapping

def create_lora_mapping(peft_model) -> dict:
    """
    Creates a mapping from base layer names to their corresponding LoRA adapter layer names
    within a PEFT LoRA model.

    Args:
        peft_model (PeftModel): An instance of a PEFT LoRA model with applied LoRA adapters.

    Returns:
        dict: A dictionary where:
              - Keys are the full path names of the base layers with LoRA adapters.
              - Values are dictionaries containing the full path names to their
                corresponding 'lora_A' and 'lora_B' adapter modules.
    """

    peft_mapping = {}

    for module_path, module in peft_model.named_modules():
        # Identify modules that are LoRA-enabled layers
        if isinstance(module, LoraLayer):
            base_layer_name = module_path

            # Iterate through all adapter names for this LoRA layer (e.g., 'default')
            for adapter_name in module.lora_A.keys():
                lora_a_full_path = f"{base_layer_name}.lora_A.{adapter_name}"
                lora_b_full_path = f"{base_layer_name}.lora_B.{adapter_name}"

                # Prioritize 'default' adapter or use the first one found
                if adapter_name == 'default' or base_layer_name not in peft_mapping:
                    peft_mapping[base_layer_name] = {
                        "adapter_A": lora_a_full_path,
                        "adapter_B": lora_b_full_path
                    }

    return peft_mapping