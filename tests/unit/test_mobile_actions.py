"""#37: the per-user action dataset is generated FROM the app's allowlist, so it teaches the boundary.

The property worth pinning is not "rows are produced" but that every completion is a call the Kotlin
`FunctionCallValidator` would accept: declared action, exactly the declared parameter keys, and values
satisfying the same `validationRules`. A generator that drifted from the validator would train the model
to emit calls the app then rejects — the two halves each fine alone, disagreeing at the seam.
"""

from __future__ import annotations

import json
import re

import pytest

from mobiletransformers.agent.mobile_actions import (
    ActionSpec,
    generate_examples,
    load_allowlist,
    write_jsonl,
)
from mobiletransformers.exceptions import ConfigValidationError

ALARM = ActionSpec(
    action_name="set_alarm",
    parameters={"time": "string", "label": "string"},
    allowed_intent="android.intent.action.SET_ALARM",
    validation_rules={"time": "HH:mm"},
    privacy_class="harmless-demo",
)
TIMER = ActionSpec(
    action_name="set_timer",
    parameters={"seconds": "string"},
    allowed_intent="android.intent.action.SET_TIMER",
    validation_rules={"seconds": "/[0-9]{1,4}/"},
)

HH_MM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def test_every_completion_is_a_call_the_validator_would_accept() -> None:
    rows = generate_examples([ALARM, TIMER], per_action=6, seed=1)

    specs = {s.action_name: s for s in (ALARM, TIMER)}
    for row in rows:
        call = json.loads(str(row["completion"]))
        spec = specs[call["actionName"]]
        # Exactly the declared keys: the validator rejects both unknown and missing parameters.
        assert set(call["parameters"]) == set(spec.parameters)
        for param, rule in spec.validation_rules.items():
            value = call["parameters"][param]
            if rule == "HH:mm":
                assert HH_MM.match(value), f"{value!r} would be rejected by the HH:mm rule"
            elif rule.startswith("/") and rule.endswith("/"):
                assert re.fullmatch(rule[1:-1], value), f"{value!r} would be rejected by {rule}"


def test_the_dataset_covers_every_declared_action() -> None:
    # A user who declared an action must have training data for it; silently skipping one would teach
    # the model that part of their own vocabulary does not exist.
    rows = generate_examples([ALARM, TIMER], per_action=4, seed=0)
    seen = {json.loads(str(r["completion"]))["actionName"] for r in rows}

    assert seen == {"set_alarm", "set_timer"}
    assert len(rows) == 8


def test_generation_is_deterministic_for_a_seed() -> None:
    # A per-user dataset that changed between runs makes "the model learned this user's actions"
    # unfalsifiable.
    assert generate_examples([ALARM], per_action=5, seed=7) == generate_examples(
        [ALARM], per_action=5, seed=7
    )


def test_an_action_with_no_template_still_gets_data() -> None:
    custom = ActionSpec(
        action_name="water_plants",
        parameters={"room": "string"},
        allowed_intent="com.example.WATER",
    )
    rows = generate_examples([custom], per_action=3, seed=0)

    assert len(rows) == 3
    assert all("water plants" in str(r["prompt"]) for r in rows)


def test_a_template_naming_an_undeclared_parameter_fails_closed() -> None:
    with pytest.raises(ConfigValidationError, match="does not.*declare"):
        generate_examples([ALARM], per_action=1, seed=0, templates={"set_alarm": ("ring at {nonexistent}",)})


def test_an_empty_allowlist_is_rejected_rather_than_producing_nothing() -> None:
    with pytest.raises(ConfigValidationError, match="empty allowlist"):
        generate_examples([], per_action=4)


def test_round_trips_through_the_action_schema_json(tmp_path) -> None:
    schema = tmp_path / "actions.json"
    schema.write_text(
        json.dumps(
            [
                {
                    "actionName": "set_alarm",
                    "parameters": {"time": "string", "label": "string"},
                    "allowedIntent": "android.intent.action.SET_ALARM",
                    "validationRules": {"time": "HH:mm"},
                    "privacyClass": "harmless-demo",
                }
            ]
        ),
        encoding="utf-8",
    )

    specs = load_allowlist(schema)
    assert specs[0].action_name == "set_alarm"
    assert specs[0].allowed_intent == "android.intent.action.SET_ALARM"

    out = write_jsonl(generate_examples(specs, per_action=2, seed=3), tmp_path / "d.jsonl")
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert all(set(json.loads(line)) == {"prompt", "completion"} for line in lines)


def test_a_malformed_action_schema_fails_closed(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"parameters": {}}]), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="actionName"):
        load_allowlist(bad)
