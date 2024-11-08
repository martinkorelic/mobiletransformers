"""
Utility functions for the framework.
"""

import os
import shutil
from jinja2 import Template

def create_chat_input(query_prompt, config, add_generation_prompt=True):
    
    # Extract key parts of the config
    chat_template = config["chat_template"]
    eos_token = config["eos_token"]
    
    # Construct messages for template rendering
    # Simulating a simple conversation setup here with roles: system, user, assistant
    messages = [
        {"role": "user", "content": query_prompt}
    ]

    # Define a rendering context for the chat template
    rendering_context = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "eos_token": eos_token
    }

    # Render the chat template using the rendering context
    chat_input = render_template(chat_template, rendering_context)
    return chat_input

def render_template(template_str, context):
    """Render the chat template string with Jinja-style template logic"""
    
    template = Template(template_str)
    return template.render(context)

def move_files_excluding(source_dir, target_dir, exclude_files):
    os.makedirs(target_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)

        if os.path.isfile(source_file) and not any(ef in filename for ef in exclude_files):
            shutil.move(source_file, target_file)

def delete_directory(directory_path):
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        try:
            shutil.rmtree(directory_path)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Directory '{directory_path}' does not exist.")