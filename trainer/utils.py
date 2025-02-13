import textwrap, inspect

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

def process_sample_commonsense(samples, tokenizer, batched=True):

    def generate_prompt(data_point):

        if tokenizer.chat_template is not None:
            messages = [
                    {"role": "user", "content": data_point["instruction"]},
                    {"role": "assistant", "content": data_point["output"]}
            ]
            return tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        else:
            return f"""
                    {data_point["instruction"]}
                    \n\n
                    {data_point["output"]}
                    """
    
    if batched:
        text = [
            generate_prompt({
                "instruction": samples["instruction"][i],
                "input": samples["input"][i],
                "output": samples["output"][i]
            })
            for i in range(len(list(samples.values())[0]))
        ]
    else:
        text = generate_prompt(samples)
    
    tk = tokenizer(text, return_tensors="pt", padding=True)
    return tk

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