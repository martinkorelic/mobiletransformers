"""The classification head's label names, and when the export records them.

A classification graph predicts an **index**, and an index is not an answer. ``_read_id2label`` copies
the HF config's ``id2label`` into ``inference/optimum_config.json`` so the device can say ``spam``
rather than ``LABEL_3``; ``classify()`` fails closed without it, by design.

It shipped with no test at all, which matters more than the size of the function suggests: it is
best-effort by construction — every failure path returns ``{}`` rather than raising — so a mistake
here is silent. An export would succeed, the package would look complete, and the labels would simply
be missing on device. These tests pin each of those quiet paths.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mobiletransformers.config.constants import PEFTMethod
from mobiletransformers.export.pipeline import ExportPlan, _read_id2label


class FakeConfig:
    """A stand-in for a transformers config: attributes only, nothing else is used."""

    def __init__(self, **fields):
        for key, value in fields.items():
            setattr(self, key, value)


class RecordingLog:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)


def plan(task: str) -> ExportPlan:
    return ExportPlan(
        model_id="acme/sentiment",
        task=task,
        peft_method=PEFTMethod.LORA,
        optimization_level=None,
        rank=8,
        quant="int4",
        variant_id="cpu-int4",
        output_dir=Path("/tmp/does-not-matter"),
        features=("inference",),
        supported_engines=("native",),
    )


@pytest.fixture
def patch_autoconfig(monkeypatch):
    """Stand a fake ``transformers`` module up in ``sys.modules``.

    ``transformers`` is deliberately absent from the dev profile — it lives in ``export`` and
    ``ort-training-local``, which cannot co-install — so there is no real ``AutoConfig`` to patch here.
    ``_read_id2label`` imports it *inside* the function, which is exactly what makes injection work:
    the import resolves against ``sys.modules`` at call time. Same approach as
    ``tests/fixtures/gen_merger_golden.py``'s ``onnxruntime`` stub.

    This keeps the test in the profile ``make check`` actually runs, rather than stranding it behind
    an ``importorskip`` that would skip in CI and never be seen to fail.
    """

    def apply(result):
        def fake_from_pretrained(model_id, **kwargs):
            if isinstance(result, Exception):
                raise result
            return result

        module = types.ModuleType("transformers")
        module.AutoConfig = types.SimpleNamespace(from_pretrained=fake_from_pretrained)
        monkeypatch.setitem(sys.modules, "transformers", module)

    return apply


def test_records_labels_for_a_classification_task(patch_autoconfig):
    patch_autoconfig(FakeConfig(id2label={0: "negative", 1: "positive"}))

    assert _read_id2label(plan("text-classification"), token=None, log=RecordingLog()) == {
        "0": "negative",
        "1": "positive",
    }


def test_keys_and_values_are_stringified(patch_autoconfig):
    """HF hands back int keys; JSON carries strings, and ``PackageTask.kt`` parses them back to ints.

    Leaving ints here would still serialise (json coerces dict keys), but the contract is stated in
    the return type and the Kotlin reader depends on it, so it is asserted rather than assumed.
    """
    patch_autoconfig(FakeConfig(id2label={0: 1, 2: "x"}))

    result = _read_id2label(plan("text-classification"), token=None, log=RecordingLog())

    assert result == {"0": "1", "2": "x"}
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in result.items())


def test_an_absent_transformers_records_nothing_instead_of_raising(monkeypatch):
    """The dev profile's real state: no ``transformers`` at all.

    The function-local import means an export run without the export extra hits ``ImportError`` here,
    and the same fail-open path must swallow it. This is why the tests above have to inject a module
    rather than patch one.
    """
    monkeypatch.setitem(sys.modules, "transformers", None)
    log = RecordingLog()

    assert _read_id2label(plan("text-classification"), token=None, log=log) == {}
    assert len(log.warnings) == 1


def test_a_decoder_task_records_nothing(patch_autoconfig):
    """A decoder's ``id2label`` is a leftover from some other head.

    Recording it would tell the device this package classifies — and `Tasks.resolve` would then offer
    a Classify path on a model with no classification head.
    """
    patch_autoconfig(FakeConfig(id2label={0: "LABEL_0"}))

    assert _read_id2label(plan("text-generation"), token=None, log=RecordingLog()) == {}


def test_feature_extraction_records_nothing(patch_autoconfig):
    patch_autoconfig(FakeConfig(id2label={0: "LABEL_0"}))

    assert _read_id2label(plan("feature-extraction"), token=None, log=RecordingLog()) == {}


def test_an_unknown_task_records_nothing_instead_of_raising():
    """``get_task_spec`` fails closed on an unknown task; the label read must not turn that into a
    failed export, because it is a convenience for one task and irrelevant to every other."""
    assert _read_id2label(plan("not-a-real-task"), token=None, log=RecordingLog()) == {}


def test_an_unreachable_config_warns_and_records_nothing(patch_autoconfig):
    """A private or offline repo must not fail an otherwise-good export — but must say so."""
    patch_autoconfig(OSError("401 Unauthorized"))
    log = RecordingLog()

    assert _read_id2label(plan("text-classification"), token=None, log=log) == {}
    assert len(log.warnings) == 1
    assert "401 Unauthorized" in log.warnings[0]


@pytest.mark.parametrize(
    "mapping",
    [None, {}, "LABEL_0", ["a", "b"], 3],
    ids=["absent", "empty", "string", "list", "int"],
)
def test_a_missing_or_malformed_mapping_records_nothing(patch_autoconfig, mapping):
    """Anything that is not a non-empty dict is not a label map.

    Without the ``isinstance`` guard a string ``id2label`` would be enumerated character by character
    and written to the package as a label set.
    """
    patch_autoconfig(FakeConfig(id2label=mapping))

    assert _read_id2label(plan("text-classification"), token=None, log=RecordingLog()) == {}


def test_a_config_without_the_attribute_at_all_records_nothing(patch_autoconfig):
    patch_autoconfig(FakeConfig())

    assert _read_id2label(plan("text-classification"), token=None, log=RecordingLog()) == {}
