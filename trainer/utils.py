"""DEPRECATED shim — moved into the package (Migration Map S4).

``create_lora_mapping`` -> ``mobiletransformers.peft.mapping``
``create_mars_adapter_mapping`` -> ``mobiletransformers.peft.mapping``
``DataCollatorForSupervisedDataset`` -> ``mobiletransformers.training.preprocessing``
``process_sample_alpaca`` -> ``mobiletransformers.training.preprocessing``
``process_sample_arc_deepeval`` -> ``mobiletransformers.training.preprocessing``
``process_sample_boolq_deepeval`` -> ``mobiletransformers.training.preprocessing``
``process_sample_dolly`` -> ``mobiletransformers.training.preprocessing``
``process_sample_hellaswag`` -> ``mobiletransformers.training.preprocessing``
``process_sample_hellaswag_deepeval`` -> ``mobiletransformers.training.preprocessing``
``process_sample_logiqa_deepeval`` -> ``mobiletransformers.training.preprocessing``
``process_sample_minipersonalqa`` -> ``mobiletransformers.training.preprocessing``
``process_sample_minirecommendation`` -> ``mobiletransformers.training.preprocessing``
``process_sample_winogrande_deepeval`` -> ``mobiletransformers.training.preprocessing``
``taskname_to_deepeval_preprocess_function`` -> ``mobiletransformers.training.preprocessing``
"""

import warnings

from mobiletransformers.peft.mapping import (  # noqa: F401
    create_lora_mapping,
    create_mars_adapter_mapping,
)
from mobiletransformers.training.preprocessing import (  # noqa: F401
    DataCollatorForSupervisedDataset,
    process_sample_alpaca,
    process_sample_arc_deepeval,
    process_sample_boolq_deepeval,
    process_sample_dolly,
    process_sample_hellaswag,
    process_sample_hellaswag_deepeval,
    process_sample_logiqa_deepeval,
    process_sample_minipersonalqa,
    process_sample_minirecommendation,
    process_sample_winogrande_deepeval,
    taskname_to_deepeval_preprocess_function,
)

warnings.warn(
    "trainer.utils moved into mobiletransformers.*; the shim will be removed.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DataCollatorForSupervisedDataset",
    "create_lora_mapping",
    "create_mars_adapter_mapping",
    "process_sample_alpaca",
    "process_sample_arc_deepeval",
    "process_sample_boolq_deepeval",
    "process_sample_dolly",
    "process_sample_hellaswag",
    "process_sample_hellaswag_deepeval",
    "process_sample_logiqa_deepeval",
    "process_sample_minipersonalqa",
    "process_sample_minirecommendation",
    "process_sample_winogrande_deepeval",
    "taskname_to_deepeval_preprocess_function",
]
