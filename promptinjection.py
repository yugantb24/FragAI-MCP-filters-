"""Prompt-injection rules using NeMo's compile/match/reject YARA approach.

The rules below are project-specific. NeMo's bundled injection rules target
SQL, code, template and XSS injection, not natural-language prompt attacks.
Reference: NVIDIA-NeMo/Guardrails, library/injection_detection (see README).
"""

import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any
import unicodedata

from guardrail_utils import InspectionLimitError, decision, message_text

logger = logging.getLogger(__name__)

# Bounded expressions keep scanning predictable on large MCP tool results.
PROMPT_INJECTION_RULES = r"""
rule instruction_override {
    strings:
        $override = /\b(ignore|disregard|forget|override|discard)[ \t\r\n]+(all[ \t\r\n]+)?(the[ \t\r\n]+|your[ \t\r\n]+)?(previous|prior|earlier|above|system|developer)[ \t\r\n]+(instructions?|prompts?|rules?|messages?)\b/ nocase
        $user = /\b(ignore|disregard)[ \t\r\n]+(the[ \t\r\n]+)?user('s)?[ \t\r\n]+(request|instructions?|task)\b/ nocase
    condition: any of them
}

rule instruction_replacement {
    strings:
        $replace = /\b(follow|obey)[ \t\r\n]+(only[ \t\r\n]+)?(these|my|the following)[ \t\r\n]+instructions?[ \t\r\n]+instead\b/ nocase
        $authority = /\b(this|the following)[ \t\r\n]+(message|text|document)[ \t\r\n]+(overrides|supersedes)[ \t\r\n]+(all[ \t\r\n]+)?(previous|system|developer)[ \t\r\n]+instructions?\b/ nocase
    condition: any of them
}

rule system_prompt_extraction {
    strings:
        $extract = /\b(reveal|print|show|repeat|disclose|output)[ \t\r\n]+(me[ \t\r\n]+)?(the[ \t\r\n]+|your[ \t\r\n]+)?(hidden[ \t\r\n]+|original[ \t\r\n]+|full[ \t\r\n]+)?(system|developer)[ \t\r\n]+(prompt|instructions?|message)\b/ nocase
    condition: $extract
}

rule role_spoofing {
    strings:
        $role1 = "<|im_start|>system" nocase
        $role2 = "<|im_start|>developer" nocase
        $role3 = "<|start_header_id|>system<|end_header_id|>" nocase
        $role4 = "[SYSTEM]" nocase
        $directive = /\b(ignore|override|obey|disregard|reveal|execute)\b/ nocase
    condition: any of ($role*) and $directive
}
"""


class PromptInjectionFilter:
    """Block known prompt-attack patterns; allow/block only, never redact."""

    def __init__(self, rules_path: str | None = None, timeout_seconds: int = 2):
        if timeout_seconds < 1:
            raise ValueError("YARA timeout must be positive.")
        self.rules_path = rules_path
        self.timeout_seconds = timeout_seconds
        self._rules: Any = None
        self._lock = Lock()

    @classmethod
    def from_environment(cls) -> "PromptInjectionFilter":
        return cls(rules_path=os.getenv("PROMPT_INJECTION_RULES_PATH") or None)

    def _load_rules(self) -> Any:
        with self._lock:
            if self._rules is None:
                import yara

                sources = {"builtin": PROMPT_INJECTION_RULES}
                if self.rules_path:
                    root = Path(self.rules_path)
                    paths = sorted(root.glob("*.yara")) if root.is_dir() else [root]
                    if not paths:
                        raise ValueError("No custom YARA rule files found.")
                    for index, path in enumerate(paths):
                        sources[f"custom_{index}"] = path.read_text(encoding="utf-8")
                # Compile once. Custom rules supplement the defaults.
                self._rules = yara.compile(sources=sources, includes=False)
            return self._rules

    def filter_message(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            text = message_text(message)
            if not text:
                return decision(message, accept=True)
            # Normalize a scanning copy only; preserve original MCP content.
            normalized = unicodedata.normalize("NFKC", text)
            normalized = "".join(c for c in normalized if unicodedata.category(c) != "Cf")
            matches = self._load_rules().match(
                data=normalized, timeout=self.timeout_seconds
            )
            if matches:
                rules = ", ".join(sorted({match.rule for match in matches}))
                return decision(
                    message, accept=False,
                    reason=f"Blocked: suspected prompt injection ({rules}).",
                )
            return decision(message, accept=True)
        except InspectionLimitError:
            return decision(message, accept=False, reason="Prompt injection inspection limit exceeded.")
        except Exception as exc:
            # Avoid logging submitted text, custom rules or sensitive paths.
            logger.error("Prompt injection detection failed (%s)", type(exc).__name__)
            return decision(message, accept=False, reason="Prompt injection detection unavailable; message blocked.")
