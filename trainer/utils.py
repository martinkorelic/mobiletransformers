from dataclasses import dataclass
import textwrap, inspect
from typing import Dict, List
from deepeval.benchmarks.hellaswag.template import HellaSwagTemplate
from deepeval.benchmarks.bool_q.template import BoolQTemplate
from deepeval.benchmarks.arc.template import ARCTemplate
from deepeval.benchmarks.logi_qa.template import LogiQATemplate
import numpy as np
import torch
from transformers import PreTrainedTokenizer

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

def process_sample_logiqa_deepeval(samples, tokenizer, batched=True):

    def generate_prompt(data_point):

        question = LogiQATemplate.format_question(data_point)
        answer = LogiQATemplate.format_output(data_point)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question}, 
                {"role": "assistant", "content": " {}\n\n".format(answer)}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        base_prompt_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        inputs = tokenizer(question + answer, return_tensors="pt", padding=False)

        labels = inputs["input_ids"].clone()
        labels[0, :len(base_prompt_tokens)-1] = -100

        return (
            inputs["input_ids"].squeeze(0),
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

        question = ARCTemplate.format_question(data_point, include_answer=False)
        question_answer = ARCTemplate.format_question(data_point, include_answer=True)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question}, 
                {"role": "assistant", "content": " {}\n\n".format(ARCTemplate.format_answer(data_point))}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        base_prompt_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        inputs = tokenizer(question_answer, return_tensors="pt", padding=False)

        labels = inputs["input_ids"].clone()
        labels[0, :len(base_prompt_tokens)-1] = -100

        return (
            inputs["input_ids"].squeeze(0),
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

        question = BoolQTemplate.format_question(data_point)
        answer = BoolQTemplate.format_answer(data_point)

        
        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question}, 
                {"role": "assistant", "content": " {}\n\n".format(answer)}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        base_prompt_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        inputs = tokenizer(question + answer, return_tensors="pt", padding=False)

        labels = inputs["input_ids"].clone()
        labels[0, :len(base_prompt_tokens)-1] = -100

        return (
            inputs["input_ids"].squeeze(0),
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
        
        if tokenizer.chat_template is not None:
            choices = ["A", "B", "C", "D"]
            gen_output = HellaSwagTemplate.format_question(
                data_point,
                include_answer=False
            )
            messages = [
                {"role": "user", "content": base_prompt + gen_output}, 
                {"role": "assistant", "content": " {}\n\n".format(choices[int(data_point["label"])])}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)
        
        gen_output_base = base_prompt + HellaSwagTemplate.format_question(
            data_point,
            include_answer=False
        )

        base_prompt_tokens = tokenizer(gen_output_base, return_tensors="pt", padding=False)["input_ids"][0]

        gen_output_full = base_prompt + HellaSwagTemplate.format_question(
            data_point,
            include_answer=True
        )

        inputs = tokenizer(gen_output_full, return_tensors="pt", padding=False)
        labels = inputs["input_ids"].clone()
        labels[0, :len(base_prompt_tokens)] = -100

        return (
            inputs["input_ids"].squeeze(0),
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
    elif preprocess_id == "arc":
        return process_sample_arc_deepeval
    elif preprocess_id == "logiqa":
        return process_sample_logiqa_deepeval

    return None