"""Chat-template rendering (Jinja).

Migrated from ``tools/utils.py`` (Migration Map S1).
"""

from __future__ import annotations


def create_chat_input(query_prompt, config, add_generation_prompt=True):

    # Extract key parts of the config
    chat_template = config["chat_template"]
    eos_token = config["eos_token"]

    # Construct messages for template rendering
    # Simulating a simple conversation setup here with roles: system, user, assistant
    messages = [{"role": "user", "content": query_prompt}]

    # Define a rendering context for the chat template
    rendering_context = {
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "eos_token": eos_token,
    }

    # Render the chat template using the rendering context
    chat_input = render_template(chat_template, rendering_context)
    return chat_input


def render_template(template_str, context):
    """Render the chat template string with Jinja-style template logic"""
    from jinja2 import Template  # noqa: PLC0415

    template = Template(template_str)
    return template.render(context)
