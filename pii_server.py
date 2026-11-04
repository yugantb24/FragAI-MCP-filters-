import logging
import os
from typing import Any

from filters import PIIFilter
from mcp_common import MCPMessage, coerce_message, create_server

logger = logging.getLogger(__name__)


def _configured_types(env_var_name: str) -> dict[str, float | None]:
    configured_types: dict[str, float | None] = {}

    for item in os.getenv(env_var_name, "").split(","):
        configured_type = item.strip()
        if not configured_type:
            continue

        entity_type, has_threshold, threshold_text = configured_type.partition("=")
        entity_type = entity_type.strip()
        if not entity_type:
            continue

        if not has_threshold:
            configured_types[entity_type] = None
            continue

        try:
            threshold = float(threshold_text)
        except ValueError:
            logger.warning(
                "Invalid confidence threshold for %s: %s. Ignoring threshold override.",
                entity_type,
                threshold_text,
            )
            configured_types[entity_type] = None
            continue

        if not 0 <= threshold <= 1:
            logger.warning(
                "Out-of-range confidence threshold for %s: %s. Expected a value between 0 and 1.",
                entity_type,
                threshold_text,
            )
            configured_types[entity_type] = None
            continue

        configured_types[entity_type] = threshold

    return configured_types


server = create_server("pii-filter", "PII", 8080)
pii_filter = PIIFilter(
    blocked_thresholds=_configured_types("PII_BLOCK_TYPES"),
    redacted_thresholds=_configured_types("PII_REDACT_TYPES"),
)


def _humanize_entity_type(entity_type: str) -> str:
    return entity_type.replace("_", " ").lower()


def _format_entity_types(entity_types: list[str]) -> tuple[str, str]:
    entity_labels = [_humanize_entity_type(entity_type) for entity_type in entity_types]
    if len(entity_labels) == 1:
        return entity_labels[0], "was"
    if len(entity_labels) == 2:
        return " and ".join(entity_labels), "were"
    return (
        ", ".join(entity_labels[:-1]) + f", and {entity_labels[-1]}",
        "were",
    )


def _filter_message(message: MCPMessage) -> dict[str, Any]:
    params_analysis = pii_filter.analyze_payload(message.params)
    result_analysis = pii_filter.analyze_payload(message.result)
    blocked_entities = pii_filter.sensitive_entities(
        params_analysis["entities"] + result_analysis["entities"],
        pii_filter.blocked_entity_types,
        entity_thresholds=pii_filter.blocked_types,
    )

    if blocked_entities:
        blocked_entity_types = sorted(
            {entity["entity_type"] for entity in blocked_entities}
        )
        blocked_description, verb = _format_entity_types(blocked_entity_types)
        return {
            "accept": False,
            "reason": (
                f"The message contains {blocked_description}, which {verb} blocked."
            ),
        }

    redacted_message = message.model_dump()
    params_result = pii_filter.redact_payload(message.params)
    result_result = pii_filter.redact_payload(message.result)

    redacted_message["params"] = params_result["redacted_payload"]
    redacted_message["result"] = result_result["redacted_payload"]

    sensitive_entities = (
        params_result["sensitive_entities"] + result_result["sensitive_entities"]
    )
    modified = params_result["modified"] or result_result["modified"]

    if modified:
        redacted_entity_types = sorted(
            {entity["entity_type"] for entity in sensitive_entities}
        )
        redacted_description, verb = _format_entity_types(redacted_entity_types)
        reason = f"The message contains {redacted_description}, which {verb} redacted."
    else:
        reason = ""

    return {
        "accept": True,
        "mutated": modified,
        "message": redacted_message,
        "reason": reason,
    }


@server.tool(
    name="filter_pii",
    title="PII Filter",
    description="Obot-compatible MCP hook that detects PII and can return a redacted message when content is not accepted as-is.",
)
def filter_pii(
    message: MCPMessage | dict[str, Any],
) -> dict[str, Any]:
    print("Filtering message")
    return _filter_message(coerce_message(message))


def main() -> None:
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
