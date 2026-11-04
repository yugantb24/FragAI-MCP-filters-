"""MCP contract and server configuration shared without importing detectors."""

import logging
import os
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, WrapValidator


class MCPMessage(BaseModel):
    jsonrpc: str = "2.0"
    id: Any | None = None
    method: str | None = None
    params: Any | None = None
    result: Any | None = None
    error: Any | None = None


# Preserve the dictionary branch, including extension fields, without changing
# the advertised JSON schema or the original PII tool's validation behavior.
GuardrailMessage = Annotated[
    MCPMessage | dict[str, Any],
    WrapValidator(lambda value, handler: value if isinstance(value, dict) else handler(value)),
]


def coerce_message(message: MCPMessage | dict[str, Any]) -> MCPMessage:
    if isinstance(message, MCPMessage):
        return message
    return MCPMessage.model_validate(message)


def guardrail_payload(message: MCPMessage | dict[str, Any]) -> dict[str, Any]:
    validated = coerce_message(message)
    return dict(message) if isinstance(message, dict) else validated.model_dump(exclude_unset=True)


def create_server(name: str, env_prefix: str, default_port: int) -> FastMCP:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    host = os.getenv(f"{env_prefix}_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv(f"{env_prefix}_PORT", os.getenv("PORT", str(default_port))))
    return FastMCP(name, host=host, port=port)
