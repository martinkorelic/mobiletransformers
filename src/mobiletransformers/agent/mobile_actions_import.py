"""Import a real function-calling dataset into the #37 tool-call training shape.

Companion to ``mobile_actions.py``: that module *generates* a per-user set from an app's allowlist,
this one *imports* an existing corpus. Both emit the same two things, so the demo can be driven by
either or by both in sequence:

* **training rows** — ``{"prompt", "completion"}`` JSONL, where the completion is exactly the JSON
  ``FunctionCallValidator.validate`` parses. What the model is trained to emit and what the app will
  accept are the same object by construction, not by a later mapping step;
* **an action schema** — the ``ActionSpec`` allowlist, derived from the corpus's own tool
  declarations, so the validator's boundary and the training targets cannot drift apart.

## The source format

Written against ``google/mobile-actions`` (CC-BY-4.0), whose records are::

    {"metadata": "train"|"eval",
     "tools":    [{"function": {"name", "description", "parameters": {...JSON-Schema-ish...}}}],
     "messages": [{"role": "developer", "content": ...},
                  {"role": "user",      "content": ...},
                  {"role": "assistant", "tool_calls": [{"function": {"name", "arguments": {...}}}]}]}

Any corpus in that shape imports — bring your own file and it is the same code path. Three details
were taken from the **data**, not from its README, because they disagree:

1. the README says ``arguments`` is "a stringified JSON object"; in all 9,654 records it is a real
   object. Both are accepted here, because a corpus that follows the README is equally valid;
2. ``parameters`` uses Gemini-style upper-case type names (``OBJECT``/``STRING``), not JSON Schema's
   lower-case ones. Both are lower-cased on the way in;
3. ``required`` is a genuine subset of ``properties`` (``send_email`` requires 2 of 3), which is why
   :class:`ActionSpec` grew ``required_parameters``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from mobiletransformers.agent.mobile_actions import ActionSpec
from mobiletransformers.exceptions import ConfigValidationError
from mobiletransformers.utils.logging import get_logger

logger = get_logger(__name__)

#: The file inside a Hub dataset repo to read when ``source`` is a repo id.
DEFAULT_DATASET_FILE = "dataset.jsonl"

#: Function name -> the Android intent action an accepted call may produce.
#:
#: **The corpus does not carry this, and must not.** `google/mobile-actions` describes function calls;
#: which intent a call is permitted to fire is the *app's* decision, and #37's safety contract is that
#: the intent string never comes from anything the model touched. So the mapping lives here, in the
#: repo, reviewable in one place — and an action absent from it gets an empty ``allowedIntent``,
#: meaning it can be trained on and validated but **never bound**. Failing that way round keeps an
#: unmapped action from silently acquiring an intent.
ANDROID_INTENT_BY_ACTION: dict[str, str] = {
    "create_calendar_event": "android.intent.action.INSERT",
    "create_contact": "android.intent.action.INSERT",
    "send_email": "android.intent.action.SENDTO",
    "show_map": "android.intent.action.VIEW",
    "open_wifi_settings": "android.settings.WIFI_SETTINGS",
    # The flashlight has no public intent action — it is a CameraManager torch call. Deliberately
    # left unmapped rather than invented, so the demo cannot claim a binding that does not exist.
}

#: Parameter name -> a `validationRules` entry, for values whose shape the corpus states in prose.
#: Only rules the Kotlin validator understands (`HH:mm`, or `/regex/`) may appear here.
VALIDATION_RULES_BY_PARAM: dict[str, str] = {
    # "The date and time of the event in the format YYYY-MM-DDTHH:MM:SS" — asserted, not assumed.
    "datetime": r"/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/",
}


def _download(repo_id: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download  # core dep, imported lazily to keep import light

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")  # type: ignore[no-any-return]


def resolve_source(
    source: str | Path,
    *,
    filename: str = DEFAULT_DATASET_FILE,
    downloader: Callable[[str, str], str] | None = None,
) -> Path:
    """A local JSONL path for ``source``, downloading it from the Hub if it is a repo id.

    ``downloader`` is injectable (defaults to ``huggingface_hub.hf_hub_download``) for the same reason
    ``hub/pull.py`` does it: the tests must run offline.
    """
    local = Path(source)
    if local.exists():
        return local
    text = str(source)
    if "/" not in text or text.endswith(".jsonl"):
        raise ConfigValidationError(
            f"no such file: {source!r} (a Hub dataset id looks like 'google/mobile-actions')"
        )
    logger.info("fetching %s:%s from the Hub", text, filename)
    return Path((downloader or _download)(text, filename))


def read_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream the JSONL, skipping blank lines. Malformed lines fail closed naming the line number."""
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigValidationError(f"{path}:{lineno} is not valid JSON: {exc}") from exc


def _arguments(raw: Any) -> dict[str, str]:
    """Tool-call arguments as a flat string map, accepting both the object and stringified forms."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"tool-call arguments are not valid JSON: {raw!r}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"tool-call arguments must be an object, got {type(raw).__name__}")
    # Every value crosses into `parameters: Map<String, String>` on the Kotlin side, so non-strings are
    # rendered here rather than at the JNI boundary where the failure would be far from its cause.
    return {k: v if isinstance(v, str) else json.dumps(v, sort_keys=True) for k, v in raw.items()}


def extract_allowlist(records: Iterable[dict[str, Any]]) -> list[ActionSpec]:
    """The union of every tool the corpus declares, as the validator's allowlist.

    Deriving the allowlist from the corpus is what keeps the two halves honest: the model is trained on
    calls to these actions, and the validator accepts exactly these actions. A hand-written allowlist
    beside a downloaded corpus is a drift waiting to happen.

    Fails closed if two records declare the same action with different parameters — that is the corpus
    disagreeing with itself, and picking one silently would make the validator's verdict depend on
    record order.
    """
    specs: dict[str, ActionSpec] = {}
    for record in records:
        for tool in record.get("tools") or []:
            function = tool.get("function") or {}
            name = function.get("name")
            if not name:
                raise ConfigValidationError(f"tool declaration has no function name: {tool}")
            schema = function.get("parameters") or {}
            properties = schema.get("properties") or {}
            spec = ActionSpec(
                action_name=name,
                parameters={p: str(v.get("type", "STRING")).lower() for p, v in properties.items()},
                allowed_intent=ANDROID_INTENT_BY_ACTION.get(name, ""),
                validation_rules={
                    p: VALIDATION_RULES_BY_PARAM[p] for p in properties if p in VALIDATION_RULES_BY_PARAM
                },
                privacy_class="imported-corpus",
                required_parameters=set(schema.get("required") or ()),
            )
            existing = specs.get(name)
            if existing is not None and existing != spec:
                raise ConfigValidationError(
                    f"action {name!r} is declared two different ways in the corpus: "
                    f"{existing.parameters} / required={sorted(existing.required)} vs "
                    f"{spec.parameters} / required={sorted(spec.required)}"
                )
            specs[name] = spec
    if not specs:
        raise ConfigValidationError("corpus declares no tools — nothing to build an allowlist from")
    return [specs[name] for name in sorted(specs)]


def _prompt(messages: list[dict[str, Any]], style: str) -> str:
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
    if not user:
        return ""
    if style == "user":
        return str(user)
    if style != "context":
        raise ConfigValidationError(f"unknown prompt style {style!r} (expected 'context' or 'user')")
    # The developer turn carries the current date and day of week, and a third of the corpus asks for
    # a calendar event in relative terms ("this Friday"). Drop it and those targets become unlearnable
    # — the model would be supervised toward a datetime nothing in its input determines.
    developer = next((m.get("content", "") for m in messages if m.get("role") == "developer"), "")
    return f"{developer}\n{user}".strip() if developer else str(user)


def to_training_rows(
    records: Iterable[dict[str, Any]],
    *,
    split: str | None = "train",
    prompt_style: str = "context",
    multi_call: str = "skip",
) -> list[dict[str, str]]:
    """Convert corpus records into ``{"prompt", "completion"}`` rows.

    ``split`` filters on the record's ``metadata`` field (``"train"`` / ``"eval"``); ``None`` keeps all.

    ``multi_call`` decides what to do with the ~33% of `google/mobile-actions` records whose assistant
    turn emits two or three calls, which the single-call ``ValidatedCall`` contract cannot express:

    * ``"skip"`` (default) drops them. **This is deliberate.** Splitting one prompt into several rows
      would train the model to answer a two-action request with one action and call that correct —
      supervision that is actively wrong, and invisible in the loss. Dropping loses examples; splitting
      loses the truth.
    * ``"first"`` keeps the first call, for when volume matters more than fidelity. Say so if you use
      it: the resulting model is being taught to under-answer.
    """
    if multi_call not in ("skip", "first"):
        raise ConfigValidationError(f"unknown multi_call policy {multi_call!r} (expected 'skip'/'first')")

    rows: list[dict[str, str]] = []
    dropped = 0
    for record in records:
        if split is not None and record.get("metadata") != split:
            continue
        messages = record.get("messages") or []
        calls = next((m.get("tool_calls") for m in messages if m.get("role") == "assistant"), None) or []
        if not calls:
            continue
        if len(calls) > 1 and multi_call == "skip":
            dropped += 1
            continue
        prompt = _prompt(messages, prompt_style)
        if not prompt:
            continue
        function = calls[0].get("function") or {}
        name = function.get("name")
        if not name:
            raise ConfigValidationError(f"tool call has no function name: {calls[0]}")
        rows.append(
            {
                "prompt": prompt,
                "completion": json.dumps(
                    {"actionName": name, "parameters": _arguments(function.get("arguments"))},
                    sort_keys=True,
                ),
            }
        )
    if dropped:
        logger.info("skipped %d multi-call record(s); kept %d single-call row(s)", dropped, len(rows))
    return rows


def write_action_schema(specs: list[ActionSpec], path: str | Path) -> Path:
    """Write the allowlist as the action-schema JSON both languages read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "actionName": spec.action_name,
            "parameters": spec.parameters,
            "allowedIntent": spec.allowed_intent,
            "requiredParameters": sorted(spec.required),
            "validationRules": spec.validation_rules,
            "privacyClass": spec.privacy_class,
        }
        for spec in specs
    ]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ANDROID_INTENT_BY_ACTION",
    "DEFAULT_DATASET_FILE",
    "VALIDATION_RULES_BY_PARAM",
    "extract_allowlist",
    "read_records",
    "resolve_source",
    "to_training_rows",
    "write_action_schema",
]
