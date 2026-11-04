# FragAI MCP filters

Three independent MCP servers use the existing Obot hook contract. Each process
exposes exactly one tool and imports only its own detector:

| Command | Entry point | Tool | Default endpoint |
| --- | --- | --- | --- |
| `pii-filter-mcp` | `pii_server.py` | `filter_pii` | `http://localhost:8080/mcp` |
| `prompt-injection-filter-mcp` | `promptinjection_server.py` | `filter_prompt_injection` | `http://localhost:8081/mcp` |
| `jailbreak-filter-mcp` | `jailbreak_server.py` | `filter_jailbreak` | `http://localhost:8082/mcp` |

Detection remains in `filters.py`, `promptinjection.py`, and `jailbreak.py`.
`mcp_common.py` shares only the message contract and server configuration; it
does not import any detector. `guardrail_utils.py` contains shared payload
extraction and decision formatting. No server calls either of the other servers.

All tools accept the existing `message` argument and return decisions using
`accept`, `mutated`, `message`, and `reason`. The new tools always return all
four fields, set `mutated` to `false`, and preserve the message. No new MCP
message fields or Obot gateway changes are needed. The existing PII logic and
its configuration are unchanged.

## Run all three locally

Install all optional dependencies into the shared development environment:

```powershell
uv sync --locked --all-extras
```

Run each command in a separate terminal:

```powershell
uv run --locked --all-extras pii-filter-mcp
```

```powershell
uv run --locked --all-extras prompt-injection-filter-mcp
```

```powershell
$env:JAILBREAK_MODEL_CACHE = "$PWD/.models"
uv run --locked --all-extras jailbreak-filter-mcp
```

Use `--all-extras` when sharing one environment across the three processes so
`uv` does not remove dependencies needed by another running guardrail.
Alternatively, use the matching Python entry-point files directly from an
already-installed environment. `python main.py` now starts **PII only**, as a
compatibility entry point. The previous combined three-tool endpoint has been
replaced by the three endpoints above.

## Install only one guardrail

For separate deployments/environments, install only the corresponding extra:

| Guardrail | Install | Start |
| --- | --- | --- |
| PII | `uv sync --locked --extra pii` | `uv run --locked --extra pii pii-filter-mcp` |
| Prompt injection | `uv sync --locked --extra prompt-injection` | `uv run --locked --extra prompt-injection prompt-injection-filter-mcp` |
| Jailbreak | `uv sync --locked --extra jailbreak` | `uv run --locked --extra jailbreak jailbreak-filter-mcp` |

The base package contains only MCP/Pydantic dependencies. PII adds Presidio
and spaCy's English model; prompt injection adds YARA; jailbreak adds ONNX
Runtime, NumPy, tokenizers and model-download dependencies. For example, an
injection-only deployment does not need the PII NLP model or jailbreak models.
The full test suite requires all extras.

## Prepare the local jailbreak models

```powershell
$env:JAILBREAK_MODEL_CACHE = "$PWD/.models"
uv run --locked --all-extras python jailbreak.py
```

In a jailbreak-only environment, use `--extra jailbreak` instead of
`--all-extras`. This preparation command downloads pinned model artifacts into the cache
and loads them locally. The full-precision embedding ONNX file is about
548 MB, plus the classifier and tokenizer. Allow additional RAM for inference.
There is no inference API or API key. Messages are never sent to Hugging Face;
network access is only for obtaining model artifacts. Once cached, set
`JAILBREAK_LOCAL_FILES_ONLY=true` to require offline loading. Preparation
avoids downloading files on the first MCP call and verifies loading in a
separate process. The jailbreak server still loads its own models into memory
once, on its first jailbreak check. PII and injection never load these models.

All servers use streamable HTTP. Each has its own host/port overrides below.

## Obot configuration

Create a separate MCP filter configuration for each desired tool, pointing
to that tool's own server endpoint and selecting its exact tool name above. Attach the filters
to the required target servers or catalog entries. Use method selectors for
`tools/call`, `resources/read`, `prompts/get`, and optionally `tools/list` for
tool-description inspection. Selectors determine which traffic reaches the
filters; simply exposing the tools does not enable them in the gateway.

Leave mutation permission off for prompt injection and jailbreak. Retain the
existing mutation permission for PII if redaction is desired. Both new filters
scan text recursively in `params`, `result` (including `structuredContent`),
and `error`, including object keys. They do not infer direction from `method`:
Obot also supplies the request method to response hooks.

## Configuration

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `PII_HOST`, `PII_PORT` | `0.0.0.0`, `8080` | PII server address. |
| `PROMPT_INJECTION_HOST`, `PROMPT_INJECTION_PORT` | `0.0.0.0`, `8081` | Prompt-injection server address. |
| `JAILBREAK_HOST`, `JAILBREAK_PORT` | `0.0.0.0`, `8082` | Jailbreak server address. |
| `HOST`, `PORT` | unset | Backward-compatible fallbacks for the process being started. Per-guardrail overrides take precedence. |
| `PROMPT_INJECTION_RULES_PATH` | unset | Optional `.yara` file or directory of `.yara` files, added to the built-in prompt rules. Includes are disabled. |
| `JAILBREAK_THRESHOLD` | `0.5` | Block if any window's class-1 probability meets this threshold; must be strictly between 0 and 1. Calibrate on your MCP traffic. |
| `JAILBREAK_MODEL_CACHE` | Hugging Face default cache | Local model cache directory. |
| `JAILBREAK_LOCAL_FILES_ONLY` | `false` | `true`/`1` disallows model downloads; missing files cause rejection. |

Existing `PII_BLOCK_TYPES` and `PII_REDACT_TYPES` apply only to the PII filter.
Avoid setting the same global `PORT` in all three local terminals unless each
also has a different per-guardrail port override. Settings for one detector
are not read by either other server.

Both new tools fail closed: missing dependencies/models, invalid custom YARA
rules, inference errors, and inspection limits return `accept: false` with a
generic reason. Reasons and logs do not echo submitted content. Invalid
jailbreak configuration fails at startup instead of disabling protection.
Text is bounded to 100,000 characters (including separators), with structural
limits of 20,000 visited nodes and depth 64. Oversized input is rejected,
not truncated. Textless messages are allowed without loading a detector.

## Detection logic and references

References were reviewed at NVIDIA-NeMo/Guardrails commit
`9a051dcc06780b70d547264dd7e6dde798335d28`:

- [Injection detection](https://github.com/NVIDIA-NeMo/Guardrails/tree/9a051dcc06780b70d547264dd7e6dde798335d28/nemoguardrails/library/injection_detection):
  compile configurable YARA rules, match content, reject detections. The
  upstream rules target SQL, code, template and XSS injection. This project's
  original prompt-specific rules detect instruction overrides, instruction
  replacement, system-prompt extraction and role spoofing. This is a pattern
  detector, not NeMo's semantic prompt-safety model. Unicode normalization
  and removal of formatting characters apply only to a scanning copy.
- [Jailbreak detection](https://github.com/NVIDIA-NeMo/Guardrails/tree/9a051dcc06780b70d547264dd7e6dde798335d28/nemoguardrails/library/jailbreak_detection):
  follow the model-based path, using unnormalized 768-dimensional CLS
  embeddings from `Snowflake/snowflake-arctic-embed-m-long` as input to
  `nvidia/NemoGuard-JailbreakDetect`. Both models run using ONNX Runtime on CPU.
  The official full-precision embedding export replaces the upstream PyTorch
  loader; no remote Python code or pickle weights are executed. Long inputs
  use 2048-token windows with 128-token overlap, with any flagged window
  blocking the message. The optional NeMo GPT-2 perplexity heuristics and
  remote NIM endpoint are not used.

Model revisions are pinned in `jailbreak.py`. The
[Snowflake model](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long)
is Apache-2.0 licensed; the
[NVIDIA classifier](https://huggingface.co/nvidia/NemoGuard-JailbreakDetect)
uses the NVIDIA Open Model License. Consult their model cards for terms and
evaluation limitations. This implementation references NeMo's algorithms;
the NeMo Guardrails package is not a runtime dependency.

Pattern and model checks can produce false positives and false negatives.
Quoted attack examples can trigger the pattern filter. Jailbreak classification
was trained on prompts, so validate thresholds on tool results before broad
deployment. This service sees only selected MCP traffic, not chat history or
user prompts that never enter MCP. Binary/image/audio content and encoded
instructions require additional extraction; text scanning does not decode them.

## Verification

```powershell
uv run --locked --all-extras python -m unittest discover -s tests -v
```

Unit/integration tests cover the decision contract, MCP registration and
serialization, nested request/response/error payloads, actual YARA matching,
model failure handling, and token-window coverage using controlled model
outputs. Fresh-interpreter tests block imports of unrelated detectors and
check that each server starts and lists only its own tool. Tool schemas are
compared with the pre-split input schema fixture. To exercise downloaded
models through the jailbreak server locally as well:

```powershell
$env:JAILBREAK_MODEL_CACHE = "$PWD/.models"
$env:JAILBREAK_LOCAL_FILES_ONLY = "true"
$env:RUN_LOCAL_MODEL_TESTS = "1"
uv run --locked --all-extras python -m unittest discover -s tests -v
```

Set `RUN_HTTP_TESTS=1` to also start all three installed commands on temporary
loopback ports, initialize MCP sessions, list tools and make filter calls.
These smoke-test processes are shut down by the tests.
