"""Dataset preprocessing + the supervised data collator.

Migrated from ``trainer/utils.py`` (Migration Map S4).

The ``deepeval.benchmarks.*`` template imports are FUNCTION-LOCAL: deepeval is an eval-extra dependency,
and importing it at module level here would make the whole training package unimportable in the core
environment (it was a top-level import in the original).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only — `from __future__ import annotations` keeps them lazy
    import numpy as np
    import torch
    from transformers import PreTrainedTokenizer


def _torch():  # noqa: ANN202
    """Lazy torch handle — this module must import in the core env (no torch installed)."""
    import torch  # noqa: PLC0415

    return torch


@dataclass
class DataCollatorForSupervisedDataset:
    """Dynamically pads input sequences for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: list[dict], return_tensors="pt") -> dict[str, torch.Tensor]:

        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # Convert to tensors
        input_ids = [_torch().tensor(x, dtype=_torch().long) for x in input_ids]
        labels = [_torch().tensor(x, dtype=_torch().long) for x in labels]

        pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )  # Default to EOS if PAD is missing

        # Pad sequences dynamically
        input_ids = _torch().nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        labels = _torch().nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Construct attention mask dynamically: 1 for non-pad tokens, 0 for pad tokens
        attention_mask = input_ids.ne(pad_token_id).long()

        # Convert to requested tensor format
        if return_tensors == "np":
            return {
                "input_ids": input_ids.numpy(),
                "labels": labels.numpy(),
                "attention_mask": attention_mask.numpy(),
            }
        elif return_tensors == "pt":
            return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        else:
            raise ValueError(f"return_tensors must be 'pt' or 'np', got {return_tensors}")

    def numpy_call(self, instances: list[dict]) -> dict[str, np.ndarray]:
        """Convenience method that returns NumPy arrays."""
        return self.__call__(instances, return_tensors="np")

    def pytorch_call(self, instances: list[dict]) -> dict[str, torch.Tensor]:
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
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)[
            "input_ids"
        ][0]

        # Concatenate the token sequences
        input_ids = _torch().cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(question_tokens)] = -100

        return (input_ids.squeeze(0), labels.squeeze(0))

    if batched:
        batch = {"input_ids": [], "labels": []}
        for i in range(len(samples["prompt"])):
            sample = {
                "type": samples["type"][i],
                "category": samples["category"][i],
                "prompt": samples["prompt"][i],
                "recommendation": samples["recommendation"][i],
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
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)[
            "input_ids"
        ][0]

        # Concatenate the token sequences
        input_ids = _torch().cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(question_tokens)] = -100

        return (input_ids.squeeze(0), labels.squeeze(0))

    if batched:
        batch = {"input_ids": [], "labels": []}
        for i in range(len(samples["question"])):
            sample = {
                "type": samples["type"][i],
                "category": samples["category"][i],
                "question": samples["question"][i],
                "choices": samples["choices"][i],
                "correct_answer": samples["correct_answer"][i],
            }

            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

        return batch
    else:
        tk = generate_prompt(samples)

    return tk


def process_sample_winogrande_deepeval(samples, tokenizer, batched=True):
    from deepeval.benchmarks.winogrande.template import WinograndeTemplate  # noqa: PLC0415

    def generate_prompt(data_point):

        question = WinograndeTemplate.format_question(data_point, include_answer=False) + "\n\n "
        answer = WinograndeTemplate.format_answer(data_point)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f" {WinograndeTemplate.format_answer(data_point)}\n\n"},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)[
            "input_ids"
        ][0]

        # Concatenate the token sequences
        input_ids = _torch().cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(question_tokens)] = -100

        return (input_ids.squeeze(0), labels.squeeze(0))

    if batched:
        batch = {"input_ids": [], "labels": []}
        for i in range(len(samples["sentence"])):
            sample = {
                "sentence": samples["sentence"][i],
                "option1": samples["option1"][i],
                "option2": samples["option2"][i],
                "answer": samples["answer"][i],
            }

            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

        return batch
    else:
        tk = generate_prompt(samples)

    return tk


def process_sample_logiqa_deepeval(samples, tokenizer, batched=True):
    from deepeval.benchmarks.logi_qa.template import LogiQATemplate  # noqa: PLC0415

    def generate_prompt(data_point):

        question = LogiQATemplate.format_question(data_point) + "\n\n "
        answer = LogiQATemplate.format_output(data_point)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f" {answer}\n\n"},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)[
            "input_ids"
        ][0]

        # Concatenate the token sequences
        input_ids = _torch().cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(question_tokens)] = -100

        return (input_ids.squeeze(0), labels.squeeze(0))

    if batched:
        batch = {"input_ids": [], "labels": []}
        for i in range(len(samples["question"])):
            sample = {
                "question": samples["question"][i],
                "text": samples["text"][i],
                "options": samples["options"][i],
                "answer": samples["answer"][i],
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

        return batch
    else:
        tp = generate_prompt(samples)
        tk = {"input_ids": tp[0], "labels": tp[1]}

    return tk


def process_sample_arc_deepeval(samples, tokenizer, batched=True):
    from deepeval.benchmarks.arc.template import ARCTemplate  # noqa: PLC0415

    def generate_prompt(data_point):

        question = ARCTemplate.format_question(data_point, include_answer=False) + "\n\n "
        answer = ARCTemplate.format_answer(data_point)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f" {ARCTemplate.format_answer(data_point)}\n\n"},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)[
            "input_ids"
        ][0]

        # Concatenate the token sequences
        input_ids = _torch().cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(question_tokens)] = -100

        return (input_ids.squeeze(0), labels.squeeze(0))

    if batched:
        batch = {"input_ids": [], "labels": []}
        for i in range(len(samples["question"])):
            sample = {
                "question": samples["question"][i],
                "choices": samples["choices"][i],
                "answerKey": samples["answerKey"][i],
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

        return batch
    else:
        tk = generate_prompt(samples)

    return tk


def process_sample_boolq_deepeval(samples, tokenizer, batched=True):
    from deepeval.benchmarks.bool_q.template import BoolQTemplate  # noqa: PLC0415

    def generate_prompt(data_point):

        question = BoolQTemplate.format_question(data_point) + "\n\n "
        answer = BoolQTemplate.format_answer(data_point)

        if tokenizer.chat_template is not None:
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f" {answer}\n\n"},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)[
            "input_ids"
        ][0]

        # Concatenate the token sequences
        input_ids = _torch().cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(question_tokens)] = -100

        return (input_ids.squeeze(0), labels.squeeze(0))

    if batched:
        batch = {"input_ids": [], "labels": []}
        for i in range(len(samples["question"])):
            sample = {
                "question": samples["question"][i],
                "passage": samples["passage"][i],
                "answer": samples["answer"][i],
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

        return batch
    else:
        tk = generate_prompt(samples)

    return tk


def process_sample_hellaswag_deepeval(samples, tokenizer, batched=True):
    from deepeval.benchmarks.hellaswag.template import HellaSwagTemplate  # noqa: PLC0415

    def generate_prompt(data_point):

        base_prompt = f"The following are multiple choice sentence completion problems about {data_point['activity_label']}.\n\n"
        choices = ["A", "B", "C", "D"]

        if tokenizer.chat_template is not None:
            gen_output = HellaSwagTemplate.format_question(data_point, include_answer=False)
            messages = [
                {"role": "user", "content": base_prompt + gen_output},
                {"role": "assistant", "content": " {}\n\n".format(choices[int(data_point["label"])])},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)

        question = base_prompt + HellaSwagTemplate.format_question(data_point, include_answer=False) + "\n\n "

        question_tokens = tokenizer(question, return_tensors="pt", padding=False)["input_ids"][0]

        answer = "{}".format(choices[int(data_point["label"])])
        answer_tokens = tokenizer(answer, return_tensors="pt", padding=False, add_special_tokens=False)[
            "input_ids"
        ][0]

        input_ids = _torch().cat([question_tokens, answer_tokens], dim=0)
        labels = input_ids.clone()
        labels[: len(question_tokens)] = -100

        return (input_ids.squeeze(0), labels.squeeze(0))

    if batched:
        batch = {"input_ids": [], "labels": []}
        for i in range(len(samples["ctx"])):
            sample = {
                "ctx": samples["ctx"][i],
                "endings": samples["endings"][i],
                "label": samples["label"][i],
                "activity_label": samples["activity_label"][i],
            }
            input_ids, labels = generate_prompt(sample)
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)

        return batch
    else:
        tp = generate_prompt(samples)
        tk = {"input_ids": tp[0], "labels": tp[1]}

    return tk


def process_sample_dolly(sample, tokenizer):

    chat = [
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["response"]},
    ]

    # TODO: This is wrong formatting
    if sample["context"]:
        chat.insert(0, {"role": "system", "content": sample["context"]})

    # Tokenize the prompt text
    text = tokenizer.apply_chat_template(
        chat, return_dict=True, tokenize=True, return_tensors="pt", padding=True, add_generation_prompt=False
    )
    return {"input_ids": text["input_ids"][0], "attention_mask": text["attention_mask"][0]}


def process_sample_alpaca(sample, tokenizer):

    def prompt_no_input(row):
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}"
        ).format_map(row)

    def prompt_input(row):
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        ).format_map(row)

    chat = ""

    if len(sample["input"]) == 0:
        chat = prompt_no_input(sample)
    else:
        chat = prompt_input(sample)

    # Tokenize the prompt text
    text = tokenizer(chat, return_tensors="pt", padding=True)
    return text


def process_sample_hellaswag(samples, tokenizer, batched=True):
    def generate_prompt(data_point):

        endings = "\n".join([f"{i + 1}. {e}" for i, e in enumerate(data_point["endings"])])

        return inspect.cleandoc(f"""
        Context: {data_point["ctx"]}

        Options:
        {endings}

        Which option best completes the context?
        Answer: {data_point["label"] + 1}
        """).strip()

    if batched:
        text = [
            generate_prompt(
                {"ctx": samples["ctx"][i], "endings": samples["endings"][i], "label": samples["label"][i]}
            )
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
    elif preprocess_id == "mini_recommendation":
        print("USING RECOMMENDATRION")
        return process_sample_minirecommendation

    return None
