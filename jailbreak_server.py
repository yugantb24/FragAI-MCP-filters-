"""Standalone MCP server exposing only local jailbreak detection."""

import asyncio
from typing import Any

from jailbreak import JailbreakFilter
from mcp_common import GuardrailMessage, create_server, guardrail_payload

server = create_server("jailbreak-filter", "JAILBREAK", 8082)
jailbreak_filter = JailbreakFilter.from_environment()


@server.tool(
    name="filter_jailbreak",
    title="Jailbreak Filter",
    description="Obot-compatible MCP hook that uses local NVIDIA NemoGuard models to block jailbreak attempts without mutating the message.",
)
async def filter_jailbreak(message: GuardrailMessage) -> dict[str, Any]:
    return await asyncio.to_thread(
        jailbreak_filter.filter_message, guardrail_payload(message)
    )


def main() -> None:
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
