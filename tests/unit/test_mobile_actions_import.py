"""#37: importing a real function-calling corpus into the tool-call training shape.

Runs offline against `tests/fixtures/agent/mobile_actions_sample.jsonl` — five records excerpted from
`google/mobile-actions` (CC-BY-4.0), chosen to cover single-call, multi-call and eval-split records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mobiletransformers.agent.mobile_actions_import import (
    ANDROID_INTENT_BY_ACTION,
    extract_allowlist,
    read_records,
    resolve_source,
    to_training_rows,
    write_action_schema,
)
from mobiletransformers.exceptions import ConfigValidationError

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "agent" / "mobile_actions_sample.jsonl"


@pytest.fixture
def records() -> list[dict]:
    return list(read_records(FIXTURE))


# --- source resolution ------------------------------------------------------


def test_local_path_is_used_as_is():
    assert resolve_source(FIXTURE) == FIXTURE


def test_repo_id_is_downloaded_through_the_injected_downloader():
    calls = []

    def fake(repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return str(FIXTURE)

    assert resolve_source("google/mobile-actions", downloader=fake) == FIXTURE
    assert calls == [("google/mobile-actions", "dataset.jsonl")]


def test_a_missing_local_file_fails_closed_rather_than_hitting_the_network():
    with pytest.raises(ConfigValidationError, match="no such file"):
        resolve_source("nope.jsonl")


# --- allowlist derivation ---------------------------------------------------


def test_allowlist_is_derived_from_the_corpus_tool_declarations(records):
    specs = {s.action_name: s for s in extract_allowlist(records)}
    assert set(specs) == {
        "create_calendar_event",
        "create_contact",
        "open_wifi_settings",
        "send_email",
        "show_map",
        "turn_off_flashlight",
        "turn_on_flashlight",
    }
    assert [s.action_name for s in extract_allowlist(records)] == sorted(specs)


def test_optional_parameters_are_not_treated_as_required(records):
    """The case `ActionSpec.required_parameters` exists for — see the class docstring."""
    specs = {s.action_name: s for s in extract_allowlist(records)}

    email = specs["send_email"]
    assert set(email.parameters) == {"to", "subject", "body"}
    assert email.required == {"to", "subject"}, "body is optional in the corpus"

    contact = specs["create_contact"]
    assert contact.required == {"first_name", "last_name"}
    assert "phone_number" in contact.parameters and "phone_number" not in contact.required


def test_an_unmapped_action_gets_no_intent_rather_than_an_invented_one(records):
    specs = {s.action_name: s for s in extract_allowlist(records)}
    assert specs["show_map"].allowed_intent == ANDROID_INTENT_BY_ACTION["show_map"]
    # The flashlight is a CameraManager torch call, not an intent. Empty, never guessed.
    assert specs["turn_on_flashlight"].allowed_intent == ""


def test_a_corpus_that_declares_one_action_two_ways_fails_closed():
    a = {"tools": [{"function": {"name": "x", "parameters": {"properties": {"p": {"type": "STRING"}}}}}]}
    b = {"tools": [{"function": {"name": "x", "parameters": {"properties": {}}}}]}
    with pytest.raises(ConfigValidationError, match="declared two different ways"):
        extract_allowlist([a, b])


def test_a_corpus_with_no_tools_fails_closed():
    with pytest.raises(ConfigValidationError, match="declares no tools"):
        extract_allowlist([{"messages": []}])


# --- training rows ----------------------------------------------------------


def test_completions_are_exactly_what_the_validator_parses(records):
    """The design property: training target and validation boundary are one object."""
    specs = {s.action_name: s for s in extract_allowlist(records)}
    rows = to_training_rows(records)
    assert rows

    for row in rows:
        call = json.loads(row["completion"])
        assert set(call) == {"actionName", "parameters"}
        spec = specs[call["actionName"]]
        supplied = call["parameters"]
        assert all(isinstance(v, str) for v in supplied.values())
        # Exactly the checks FunctionCallValidator.validate performs.
        assert not (set(supplied) - set(spec.parameters)), "no undeclared parameter"
        assert not (spec.required - set(supplied)), "every required parameter present"


def test_multi_call_records_are_skipped_by_default(records):
    default = to_training_rows(records)
    kept = to_training_rows(records, multi_call="first")
    assert len(kept) > len(default), "the fixture carries a multi-call record"


def test_multi_call_policy_is_validated(records):
    with pytest.raises(ConfigValidationError, match="unknown multi_call"):
        to_training_rows(records, multi_call="explode")


def test_split_filtering(records):
    train = to_training_rows(records, split="train")
    evaluation = to_training_rows(records, split="eval")
    everything = to_training_rows(records, split=None)
    assert train and evaluation
    assert len(everything) == len(train) + len(evaluation)


def test_context_prompt_carries_the_date_so_relative_targets_are_learnable(records):
    """A third of the corpus asks for a calendar event in relative terms ("this Friday")."""
    with_context = to_training_rows(records, prompt_style="context")
    bare = to_training_rows(records, prompt_style="user")
    assert any("Current date and time" in r["prompt"] for r in with_context)
    assert not any("Current date and time" in r["prompt"] for r in bare)
    assert len(with_context) == len(bare)


def test_unknown_prompt_style_fails_closed(records):
    with pytest.raises(ConfigValidationError, match="unknown prompt style"):
        to_training_rows(records, prompt_style="freeform")


def test_stringified_arguments_are_accepted_too():
    """The dataset card documents `arguments` as a JSON string; the data ships objects. Both work."""
    record = {
        "metadata": "train",
        "tools": [{"function": {"name": "show_map", "parameters": {"properties": {"query": {}}}}}],
        "messages": [
            {"role": "user", "content": "where is the station"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "show_map", "arguments": '{"query": "station"}'}}],
            },
        ],
    }
    (row,) = to_training_rows([record])
    assert json.loads(row["completion"])["parameters"] == {"query": "station"}


def test_non_string_argument_values_are_rendered_not_dropped():
    record = {
        "metadata": "train",
        "tools": [{"function": {"name": "t", "parameters": {"properties": {"n": {}}}}}],
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "t", "arguments": {"n": 5}}}]},
        ],
    }
    (row,) = to_training_rows([record])
    # Crosses into Kotlin as Map<String, String>; converting here keeps the failure near its cause.
    assert json.loads(row["completion"])["parameters"] == {"n": "5"}


# --- action schema ----------------------------------------------------------


def test_action_schema_round_trips_through_the_wire_names(tmp_path, records):
    specs = extract_allowlist(records)
    path = write_action_schema(specs, tmp_path / "action_schema.json")
    payload = json.loads(path.read_text())

    assert {row["actionName"] for row in payload} == {s.action_name for s in specs}
    for row in payload:
        # camelCase, and exactly the keys the Kotlin ActionSpec declares.
        assert set(row) == {
            "actionName",
            "parameters",
            "allowedIntent",
            "requiredParameters",
            "validationRules",
            "privacyClass",
        }

    email = next(row for row in payload if row["actionName"] == "send_email")
    assert email["requiredParameters"] == ["subject", "to"]


def test_datetime_rule_is_one_the_kotlin_validator_understands(tmp_path, records):
    """Only `HH:mm` and `/regex/` are supported there, and an unrecognised rule rejects everything."""
    import re

    specs = {s.action_name: s for s in extract_allowlist(records)}
    rule = specs["create_calendar_event"].validation_rules["datetime"]
    assert rule.startswith("/") and rule.endswith("/")
    assert re.compile(rule[1:-1]).match("2025-06-06T14:00:00")

    # And every emitted datetime actually satisfies it, so training targets pass their own gate.
    for row in to_training_rows(records):
        params = json.loads(row["completion"])["parameters"]
        if "datetime" in params:
            assert re.compile(rule[1:-1]).fullmatch(params["datetime"]), params["datetime"]


def test_schema_round_trips_back_into_action_specs(tmp_path, records):
    """write_action_schema -> load_allowlist must preserve the required set, or the generator would
    treat every optional parameter as mandatory and produce rows the corpus itself contradicts."""
    from mobiletransformers.agent.mobile_actions import generate_examples, load_allowlist

    path = write_action_schema(extract_allowlist(records), tmp_path / "action_schema.json")
    reloaded = {s.action_name: s for s in load_allowlist(path)}
    assert reloaded["send_email"].required == {"to", "subject"}
    assert reloaded["create_contact"].required == {"first_name", "last_name"}

    # And the synthetic generator runs off the imported schema — the per-user layer on the same
    # boundary as the corpus.
    rows = generate_examples(list(reloaded.values()), per_action=2, seed=1)
    assert len(rows) == 2 * len(reloaded)
    for row in rows:
        call = json.loads(row["completion"])
        assert not (reloaded[call["actionName"]].required - set(call["parameters"]))


def test_a_schema_without_required_parameters_keeps_the_stricter_default(tmp_path):
    from mobiletransformers.agent.mobile_actions import load_allowlist

    path = tmp_path / "old_schema.json"
    path.write_text(json.dumps([{"actionName": "a", "parameters": {"p": "string", "q": "string"}}]))
    (spec,) = load_allowlist(path)
    assert spec.required_parameters is None
    assert spec.required == {"p", "q"}
