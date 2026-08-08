"""The PEFT benchmark dataset registry: task name -> (HF dataset id, preprocessor id[, config name]).

Migration Map S6b side-effect. This lived in ``research/offline_train_eval.py``, but
``training/validators.py`` — now a packaged module — depends on it. A packaged module importing
``research.`` works from a checkout and fails from an installed wheel, so the data moved rather than
the import being allow-listed. ``research/offline_train_eval.py`` re-exports both names.

Not to be confused with :data:`mobiletransformers.config.constants.TASK_NAME_TO_DATASET`, which maps
the *export/training CLI's* task names. This one is keyed by the PEFT benchmark suite's own task ids
and additionally carries the preprocessor id each dataset needs.
"""

from __future__ import annotations

from enum import Enum


class PEFTBenchmarkDataset(Enum):
    """Datasets the PEFT benchmark suite trains and evaluates against."""

    # Easy tasks
    BOOLQ = "boolq"
    ARC_E = "arc_e"
    LOGIQA = "logiqa"
    WINOGRANDE = "winogrande"

    # Complex tasks
    HELLASWAG = "hellaswag"
    ARC_C = "arc_c"

    # Mobile tasks
    MINI_PERSONALQA = "mini_personalqa"
    MINI_RECOMMENDATION = "mini_recommendation"


#: ``task -> (dataset_id, preprocess_id)`` or ``(dataset_id, preprocess_id, dataset_config_name)``.
#: The 3-tuple form selects a named config within a multi-config HF dataset (ai2_arc's Easy/Challenge
#: splits, winogrande's size variants); consumers must handle both arities.
DATASET_MAPPING: dict[str, tuple[str, ...]] = {
    # Easy tasks
    PEFTBenchmarkDataset.BOOLQ.value: ("google/boolq", "boolq_train_deepeval"),
    PEFTBenchmarkDataset.WINOGRANDE.value: (
        "allenai/winogrande",
        "winogrande_train_deepeval",
        "winogrande_l",
    ),
    PEFTBenchmarkDataset.ARC_E.value: ("allenai/ai2_arc", "arc_train_deepeval", "ARC-Easy"),
    PEFTBenchmarkDataset.LOGIQA.value: ("data/logiqa_train", "logiqa_train_deepeval"),
    # Complex tasks
    PEFTBenchmarkDataset.HELLASWAG.value: ("Rowan/hellaswag", "hellaswag_train_deepeval"),
    PEFTBenchmarkDataset.ARC_C.value: ("allenai/ai2_arc", "arc_train_deepeval", "ARC-Challenge"),
    # Mobile tasks
    PEFTBenchmarkDataset.MINI_PERSONALQA.value: ("data/MiniPersonalQA_train", "mini_personalqa"),
    PEFTBenchmarkDataset.MINI_RECOMMENDATION.value: (
        "data/MiniRecommendation_train",
        "mini_recommendation",
    ),
}
