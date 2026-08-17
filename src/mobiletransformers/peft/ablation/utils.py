"""Ablation target-module table.

DEDUPLICATED (#6): the table itself lives once in
``mobiletransformers.config.registry.peft.PEFT_TARGET_MODULES_BY_MODEL_TYPE``. This module and
``peft_models/mars/utils.py`` used to carry byte-identical copies under two different names, so a
new model type had to be added twice or the two silently drifted apart.
"""

from mobiletransformers.config.registry.peft import PEFT_TARGET_MODULES_BY_MODEL_TYPE

TRANSFORMERS_MODELS_TO_ABLATION_TARGET_MODULES_MAPPING = PEFT_TARGET_MODULES_BY_MODEL_TYPE
