"""Synthetic **per-user** mobile-action datasets for the #37 tool-call demo.

## Why synthetic, and why per-user

The differentiation gate in ``04_code_plans/05`` is explicit that running a vendor's off-device tutorial
is not a contribution. What this project can show that a hosted assistant cannot is a model fine-tuned
**on one person's own action vocabulary, on their device, from data that never leaves it**. That needs a
per-user dataset, and a per-user dataset cannot be downloaded — it has to be generated from the action
set that user's app actually declares.

So the generator takes the **allowlist** as input. The training targets it emits are exactly the calls
``FunctionCallValidator`` would accept: same action names, same parameter keys, values that satisfy the
same ``validationRules``. A model trained on this is being taught the app's real boundary rather than a
generic function-calling format that then has to be validated into shape.

## What it deliberately does not do

No natural-language *diversity* modelling: the prompt templates are simple and few. The dataset exists
to demonstrate the loop (per-user actions → on-device fine-tune → validated call → dry-run intent), not
to be a benchmark, and pretending otherwise by generating thousands of near-duplicates would overstate
what it shows.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from mobiletransformers.exceptions import ConfigValidationError

#: Prompt templates per action, with `{...}` slots naming the action's own parameters.
#:
#: Keyed by action name so an app's allowlist drives what is generated. An action with no template
#: falls back to a generic phrasing rather than being skipped — a silently missing action would mean a
#: user's dataset lacked the very action they declared.
DEFAULT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "set_alarm": (
        "wake me at {time}",
        "set an alarm for {time} called {label}",
        "alarm {time} for {label}",
    ),
    "set_timer": (
        "timer for {seconds} seconds",
        "count down {seconds} seconds",
    ),
    "send_message": (
        "text {recipient} saying {body}",
        "message {recipient}: {body}",
    ),
}

#: Value pools per rule, so generated values satisfy the SAME rules the validator enforces.
_VALUES_BY_RULE: dict[str, tuple[str, ...]] = {
    "HH:mm": ("06:15", "07:30", "08:00", "12:45", "18:20", "22:05"),
}

_GENERIC_VALUES: tuple[str, ...] = ("gym", "work", "school", "run", "call mum", "groceries")


@dataclass
class ActionSpec:
    """Mirror of the Kotlin ``ActionSpec``. The app's declaration, not the model's."""

    action_name: str
    parameters: dict[str, str] = field(default_factory=dict)
    allowed_intent: str = ""
    validation_rules: dict[str, str] = field(default_factory=dict)
    privacy_class: str = "unspecified"
    #: Parameters that MUST be present. ``None`` means "all of them", which is what a hand-written
    #: allowlist means and what the validator enforced before optional parameters existed.
    #:
    #: Real tool schemas distinguish the two: in `google/mobile-actions`, `send_email` declares
    #: `subject`/`body`/`to` but requires only `to`/`subject`, and `create_contact` requires 2 of 4.
    #: Treating every declared parameter as required would reject calls the dataset itself considers
    #: correct — the model would be trained on targets its own validator refuses.
    required_parameters: set[str] | None = None

    @property
    def required(self) -> set[str]:
        """The effective required set (all declared parameters unless narrowed)."""
        return set(self.parameters) if self.required_parameters is None else set(self.required_parameters)


def _value_for(param: str, rule: str | None, rng: random.Random) -> str:
    if rule is not None and rule in _VALUES_BY_RULE:
        return rng.choice(_VALUES_BY_RULE[rule])
    if rule is not None and rule.startswith("/") and rule.endswith("/") and "0-9" in rule:
        return str(rng.randint(5, 900))
    if param in ("seconds", "minutes"):
        return str(rng.randint(5, 900))
    return rng.choice(_GENERIC_VALUES)


def generate_examples(
    allowlist: list[ActionSpec],
    *,
    per_action: int = 8,
    seed: int = 0,
    templates: dict[str, tuple[str, ...]] | None = None,
) -> list[dict[str, str]]:
    """One user's dataset: ``{"prompt": ..., "completion": <tool-call JSON>}`` rows.

    The completion is the exact JSON shape ``FunctionCallValidator.validate`` parses, so what the model
    is trained to emit and what the app will accept are the same object by construction rather than by
    a later mapping step.

    Deterministic for a given ``seed`` — a per-user dataset that changed between runs would make "the
    model learned this user's actions" unfalsifiable.
    """
    if not allowlist:
        raise ConfigValidationError("cannot generate a dataset from an empty allowlist")
    if per_action < 1:
        raise ConfigValidationError(f"per_action must be >= 1, got {per_action}")

    table = {**DEFAULT_TEMPLATES, **(templates or {})}
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for spec in allowlist:
        forms = table.get(spec.action_name) or (
            f"{spec.action_name.replace('_', ' ')} " + " ".join(f"{{{p}}}" for p in spec.parameters),
        )
        for _ in range(per_action):
            params = {
                param: _value_for(param, spec.validation_rules.get(param), rng) for param in spec.parameters
            }
            form = rng.choice(forms)
            try:
                prompt = form.format(**params)
            except KeyError as exc:
                # A template naming a parameter the action does not declare would produce a prompt the
                # completion cannot satisfy. Fail closed naming both, rather than emitting the pair.
                raise ConfigValidationError(
                    f"template for {spec.action_name!r} references {exc} which the action does not "
                    f"declare (declared: {sorted(spec.parameters)})"
                ) from exc
            rows.append(
                {
                    "prompt": prompt,
                    "completion": json.dumps(
                        {"actionName": spec.action_name, "parameters": params},
                        sort_keys=True,
                    ),
                }
            )

    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[dict[str, str]], path: str | Path) -> Path:
    """Write ``rows`` as JSONL, the format ``ORTDataCurator`` reads on device."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def load_allowlist(path: str | Path) -> list[ActionSpec]:
    """Read an app's action-schema JSON — the same records the Kotlin validator is built from."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ConfigValidationError("action schema must be a JSON array of action records")
    specs = []
    for row in data:
        if "actionName" not in row:
            raise ConfigValidationError(f"action record has no 'actionName': {row}")
        specs.append(
            ActionSpec(
                action_name=row["actionName"],
                parameters=row.get("parameters", {}),
                allowed_intent=row.get("allowedIntent", ""),
                validation_rules=row.get("validationRules", {}),
                privacy_class=row.get("privacyClass", "unspecified"),
                # Absent means "all declared parameters are required" — the same default the Kotlin
                # ActionSpec applies, so a schema written before optional parameters existed keeps
                # its original, stricter meaning on both sides.
                required_parameters=(set(row["requiredParameters"]) if "requiredParameters" in row else None),
            )
        )
    return specs


__all__ = [
    "ActionSpec",
    "DEFAULT_TEMPLATES",
    "generate_examples",
    "load_allowlist",
    "write_jsonl",
]
