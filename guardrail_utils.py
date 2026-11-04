"""Shared payload handling for non-mutating Obot guardrails."""

from typing import Any


class InspectionLimitError(ValueError):
    """The complete payload cannot be inspected within the configured limits."""


def message_text(message: dict[str, Any], max_chars: int = 100_000) -> str:
    """Collect payload text without changing its JSON structure.

    Include dictionary keys because tool results can expose model-visible text
    in keys too. JSON-RPC routing fields (method and id) are not prompt text.
    Bound both text and structural work; never silently truncate a message.
    """
    parts: list[str] = []
    total = 0
    visited = 0
    stack = [(message.get(field), 0) for field in ("error", "result", "params")]
    while stack:
        value, depth = stack.pop()
        visited += 1
        if visited > 20_000 or depth > 64:
            raise InspectionLimitError("Payload structure exceeds the inspection limit.")
        if isinstance(value, str):
            if value.strip():
                total += len(value) + 1
                if total > max_chars:
                    raise InspectionLimitError("Payload text exceeds the inspection limit.")
                parts.append(value)
        elif isinstance(value, dict):
            if len(value) > 10_000:
                raise InspectionLimitError("Payload structure exceeds the inspection limit.")
            for key, item in reversed(list(value.items())):
                stack.append((item, depth + 1))
                stack.append((key, depth + 1))
        elif isinstance(value, list):
            if len(value) > 20_000:
                raise InspectionLimitError("Payload structure exceeds the inspection limit.")
            stack.extend((item, depth + 1) for item in reversed(value))
    return "\n".join(parts)


def decision(message: dict[str, Any], *, accept: bool, reason: str = "") -> dict[str, Any]:
    return {
        "accept": accept,
        "mutated": False,
        "message": message,
        "reason": reason,
    }
