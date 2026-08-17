"""Typed config models: round-trip stability, discriminated union, and fail-closed parsing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mobiletransformers.config.models import (
    CROSS_BOUNDARY_MODELS,
    CosineScheduler,
    GenerationConfig,
    LinearScheduler,
    TrainingConfig,
)


@pytest.mark.parametrize("name,model", list(CROSS_BOUNDARY_MODELS.items()))
def test_round_trip_is_byte_stable(name, model):
    dumped = model().model_dump(by_alias=True, mode="json")
    reparsed = model.model_validate(dumped).model_dump(by_alias=True, mode="json")
    assert reparsed == dumped


@pytest.mark.parametrize("name,model", list(CROSS_BOUNDARY_MODELS.items()))
def test_version_block_present(name, model):
    dumped = model().model_dump(by_alias=True, mode="json")
    assert dumped["schemaVersion"] == "1.0"
    assert dumped["minReaderVersion"] == "1.0"


def test_scheduler_union_selects_by_wire_tag():
    linear = TrainingConfig(scheduler=LinearScheduler()).model_dump(by_alias=True, mode="json")
    cosine = TrainingConfig(scheduler=CosineScheduler()).model_dump(by_alias=True, mode="json")
    assert linear["scheduler"]["schedulerType"] == "linear"
    assert cosine["scheduler"]["schedulerType"] == "cosine"
    assert isinstance(TrainingConfig.model_validate(cosine).scheduler, CosineScheduler)
    assert isinstance(TrainingConfig.model_validate(linear).scheduler, LinearScheduler)


def test_unknown_field_tolerated():
    dumped = GenerationConfig().model_dump(by_alias=True, mode="json")
    # additive minor bumps must not break older readers (extra="ignore")
    GenerationConfig.model_validate({**dumped, "someFutureField": 123})


def test_unknown_enum_value_fails_closed():
    dumped = GenerationConfig().model_dump(by_alias=True, mode="json")
    dumped["sampling"]["method"] = "definitely-not-a-method"
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(dumped)


def test_camelcase_aliases_on_wire():
    dumped = GenerationConfig().model_dump(by_alias=True, mode="json")
    assert "maxSequenceLength" in dumped
    assert "deviceOptions" in dumped
    assert dumped["sampling"]["topK"] == 10  # alias, not top_k
