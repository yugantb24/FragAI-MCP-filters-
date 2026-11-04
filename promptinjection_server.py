"""Standalone MCP server exposing only prompt-injection detection."""

import asyncio
from typing import Any

from mcp_common import GuardrailMessage, create_server, guardrail_payload
from promptinjection import PromptInjectionFilter

server = create_server("prompt-injection-filter", "PROMPT_INJECTION", 8081)
prompt_injection_filter = PromptInjectionFilter.from_environment()


@server.tool(
    name="filter_prompt_injection",
    title="Prompt Injection Filter",
    description="Obot-compatible MCP hook that blocks known prompt-injection patterns without mutating the message.",
)
async def filter_prompt_injection(message: GuardrailMessage) -> dict[str, Any]:
    return await asyncio.to_thread(
        prompt_injection_filter.filter_message, guardrail_payload(message)
    )


def main() -> None:
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
