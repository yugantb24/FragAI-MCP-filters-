import json
import logging
from typing import Any, Dict, List, Tuple

from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)

FALLBACK_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_CONFIDENCE_THRESHOLDS = {
    "EMAIL_ADDRESS": 0.8,
    "PHONE_NUMBER": 0.6,
    "CREDIT_CARD": 0.8,
    "CRYPTO": 0.8,
    "IBAN_CODE": 0.8,
    "IP_ADDRESS": 0.6,
    "US_SSN": 0.5,
    "US_BANK_NUMBER": 0.4,
    "US_PASSPORT": 0.4,
    "US_DRIVER_LICENSE": 0.6,
    "MEDICAL_LICENSE": 0.6,
}


def default_confidence_threshold(entity_type: str) -> float:
    return DEFAULT_CONFIDENCE_THRESHOLDS.get(entity_type, FALLBACK_CONFIDENCE_THRESHOLD)


def configured_entity_types(
    configured_types: dict[str, float | None], allowed_entity_types: set[str]
) -> dict[str, float | None]:
    entity_types = set(configured_types)

    unknown_entity_types = entity_types - allowed_entity_types
    if unknown_entity_types:
        logger.warning(
            f"Unknown values: "
            f"{', '.join(sorted(unknown_entity_types))}. "
            "Ignoring unknown values. Allowed values: "
            f"{', '.join(sorted(allowed_entity_types))}"
        )

    return {
        entity_type: configured_types[entity_type]
        for entity_type in entity_types & allowed_entity_types
    }


class PIIFilter:
    """Filter to detect and reject PII using Microsoft Presidio."""

    def __init__(
        self,
        blocked_thresholds: dict[str, float | None] | None = None,
        redacted_thresholds: dict[str, float | None] | None = None,
    ) -> None:
        self.analyzer = AnalyzerEngine()
        self.supported_entity_types = set(
            self.analyzer.get_supported_entities(language="en")
        )
        self.blocked_types = configured_entity_types(
            blocked_thresholds or {}, self.supported_entity_types
        )
        self.redacted_types = configured_entity_types(
            redacted_thresholds or {}, self.supported_entity_types
        )
        self.blocked_entity_types = set(self.blocked_types)
        self.redacted_entity_types = set(self.redacted_types)
        self.analyzed_entity_types = (
            self.blocked_entity_types | self.redacted_entity_types
        )
        logger.info(
            "PII filter initialized with %s supported types, block types: %s, redact types: %s",
            len(self.supported_entity_types),
            sorted(self.blocked_entity_types),
            sorted(self.redacted_entity_types),
        )

    def analyze_text(
        self, text: str, entity_types: set[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Analyze text for PII entities."""
        if entity_types is None:
            entity_types = self.analyzed_entity_types
        if not entity_types:
            return []

        results = self.analyzer.analyze(
            text=text,
            language="en",
            entities=list(entity_types),
        )
        return [
            {
                "entity_type": result.entity_type,
                "start": result.start,
                "end": result.end,
                "confidence": result.score,
                "text": text[result.start : result.end],
            }
            for result in results
        ]

    def analyze_payload(self, payload: Any) -> Dict[str, Any]:
        """Analyze an arbitrary JSON-serializable payload for PII."""
        if payload is None:
            return {"has_pii": False, "entities": []}

        payload_text = json.dumps(payload, ensure_ascii=False)
        entities = self.analyze_text(payload_text)

        logger.info("PII analysis found %s entities", len(entities))
        for entity in entities:
            logger.warning(
                "PII detected: %s (confidence: %.2f)",
                entity["entity_type"],
                entity["confidence"],
            )

        return {
            "has_pii": len(entities) > 0,
            "entities": entities,
        }

    def sensitive_entities(
        self,
        entities: List[Dict[str, Any]],
        entity_types: set[str],
        entity_thresholds: Dict[str, float | None] | None = None,
    ) -> List[Dict[str, Any]]:
        return [
            entity
            for entity in entities
            if entity["entity_type"] in entity_types
            and entity["confidence"]
            >= self._confidence_threshold(entity["entity_type"], entity_thresholds)
        ]

    def _confidence_threshold(
        self,
        entity_type: str,
        entity_thresholds: Dict[str, float | None] | None = None,
    ) -> float:
        configured_threshold = (
            entity_thresholds.get(entity_type) if entity_thresholds is not None else None
        )
        if configured_threshold is not None:
            return configured_threshold

        return default_confidence_threshold(entity_type)

    def redact_text(
        self, text: str
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        entities = self.analyze_text(text)
        sensitive_entities = self.sensitive_entities(
            entities,
            self.redacted_entity_types,
            self.redacted_types,
        )

        if not sensitive_entities:
            return text, entities, []

        redacted = text
        for entity in sorted(
            sensitive_entities, key=lambda item: item["start"], reverse=True
        ):
            replacement = f"[REDACTED_{entity['entity_type']}]"
            redacted = (
                redacted[: entity["start"]] + replacement + redacted[entity["end"] :]
            )

        return redacted, entities, sensitive_entities

    def redact_payload(self, payload: Any) -> Dict[str, Any]:
        all_entities: List[Dict[str, Any]] = []
        all_sensitive_entities: List[Dict[str, Any]] = []

        def walk(value: Any) -> Any:
            if isinstance(value, str):
                redacted, entities, sensitive_entities = self.redact_text(value)
                all_entities.extend(entities)
                all_sensitive_entities.extend(sensitive_entities)
                return redacted

            if isinstance(value, list):
                return [walk(item) for item in value]

            if isinstance(value, dict):
                return {key: walk(item) for key, item in value.items()}

            return value

        redacted_payload = walk(payload)
        return {
            "redacted_payload": redacted_payload,
            "entities": all_entities,
            "sensitive_entities": all_sensitive_entities,
            "modified": redacted_payload != payload,
        }