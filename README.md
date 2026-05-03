# ContextFuse

ContextFuse is a context fuse / safety proxy for local `llama-server`.

It sits between Claude Code, OpenClaw, or any Anthropic Messages compatible
client and a local llama.cpp server. Its job is to prevent long agent sessions
from crashing into the model context limit, while preserving as much useful
conversation intelligence as possible.

```text
Claude Code / OpenClaw / client
        |
        v
ContextFuse :8001
        |
        v
llama-server :8000
```

Keywords for people looking for this problem: llama-server context overflow,
Claude Code local model 400 error, OpenClaw local LLM context limit,
Anthropic Messages proxy, llama.cpp long context compaction, multimodal
llama-server image proxy.

## Why This Exists

`llama-server` is a strong local inference engine, but it does not provide
production-grade automatic context management. When a request exceeds
`--ctx-size`, it can return HTTP 400 and the agent session may be interrupted.

ContextFuse adds a safety layer:

- estimates or asks the backend for token counts;
- strips old `tool_use`, `tool_result`, and `thinking` blocks when needed;
- truncates huge retained tool outputs;
- drops oldest turns only as a final fallback;
- injects a visible long-context reminder when pure conversation text becomes large;
- supports image-aware estimation, resizing, and old-image compaction for multimodal models.

Small requests pass through with minimal overhead. Large requests are compacted
before they hit `llama-server`.

## Features

- Anthropic Messages compatible `POST /v1/messages`
- `GET /health`
- `GET /v1/models`
- `POST /v1/messages/count_tokens` passthrough with local fallback
- Structured backend errors: invalid JSON, backend unreachable, backend timeout
- Streaming SSE warning injection that only targets text blocks
- Log rotation when `PROXY_LOG_FILE` is set
- OpenClaw custom provider friendly
- Multimodal image handling:
  - Anthropic `image` blocks
  - URL image blocks
  - OpenAI-style `image_url` blocks
  - images nested inside `tool_result.content[]`
  - optional base64 image resizing with Pillow

## Quick Start

```bash
python -m venv proxy-venv
source proxy-venv/bin/activate
pip install -r requirements.txt

python context_proxy.py
```

Point your client at:

```text
ANTHROPIC_BASE_URL=http://localhost:8001
ANTHROPIC_AUTH_TOKEN=local-dummy-key
ANTHROPIC_MODEL=your-local-model-id
```

Example with custom settings:

```bash
PROXY_BACKEND_URL=http://localhost:8000 \
PROXY_MODEL_ID=qwen36-27b-q3km-96k \
PROXY_CONTEXT_LIMIT=98304 \
PROXY_TRIGGER=75000 \
PROXY_LOG_FILE=./contextfuse.log \
python context_proxy.py
```

Production deployment examples are in
[DEPLOYMENT.md](DEPLOYMENT.md) and
[systemd/contextfuse.service.example](systemd/contextfuse.service.example).

## Compaction Strategy

Compaction only activates when estimated input tokens exceed
`PROXY_TRIGGER`.

```text
Step 0  Resize base64 images when Pillow is available.
Step 1  Strip old tool_use / tool_result / thinking / image blocks.
Step 2  Truncate huge retained tool_result blocks, including nested images.
Step 3  Drop oldest user/assistant turns.
Step 4  Shrink the recent-message protection window and strip more aggressively.
Step 5  Drop oldest turns again as the final fallback.
```

ContextFuse keeps normal user and assistant text intact until the final
drop-oldest fallback becomes unavoidable.

## Vision / Multimodal

ContextFuse can forward image inputs for multimodal `llama-server` backends.
It does not add vision ability by itself: the backend must be started with a
compatible multimodal model and projector, for example:

```bash
llama-server \
  -m /opt/models/Qwen3.6-27B-Q3_K_M.gguf \
  --mmproj /opt/models/mmproj-Qwen-Qwen3.6-27B-Q6_K.gguf
```

Supported Anthropic image block:

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/png",
    "data": "..."
  }
}
```

Supported URL image block:

```json
{
  "type": "image",
  "source": {
    "type": "url",
    "url": "https://example.com/image.png"
  }
}
```

`image_url` blocks with `data:image/...;base64,...` are also recognized.
Remote image URLs are never downloaded by ContextFuse.

Image behavior:

- each image counts as `PROXY_IMAGE_TOKEN_ESTIMATE` local estimated tokens;
- old image blocks are replaced with placeholders during compaction;
- images inside `tool_result.content[]` are counted and compacted;
- base64 images can be resized before token counting and forwarding when Pillow is installed;
- base64 payloads and full remote URLs are not written to logs.

Pillow is optional. Without Pillow, image resizing is disabled but image
estimation and compaction still work.

```bash
pip install Pillow
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `PROXY_BACKEND_URL` | `http://localhost:8000` | Backend llama-server URL |
| `PROXY_MODEL_ID` | `gemma4-26b-iq4xs-128k` | Static model id returned when backend models are unavailable |
| `PROXY_CONTEXT_LIMIT` | `131072` | Informational context window used in warnings and health output |
| `PROXY_TRIGGER` | `75000` | Compaction trigger |
| `PROXY_CD_POINT` | `30000` | Pure-text long-context warning threshold |
| `PROXY_RESERVE_TOKENS` | `8192` | Output reserve |
| `PROXY_RECENT_SKIP` | `6` | Recent message count protected from old-history stripping |
| `PROXY_USE_COUNT_TOKENS` | `1` | Use backend `/v1/messages/count_tokens` when near trigger |
| `PROXY_COUNT_TOKENS_PRECISE_THRESHOLD` | `0.7` | Call precise count only after local estimate reaches trigger times this ratio |
| `PROXY_INJECT_STREAM_WARNINGS` | `1` | Inject long-context warning into streaming text output |
| `PROXY_TOOL_RESULT_TRUNCATE_TRIGGER_CHARS` | `20000` | Tool result truncation threshold |
| `PROXY_TOOL_RESULT_TRUNCATE_KEEP_CHARS` | `400` | Prefix/suffix chars kept when truncating text tool results |
| `PROXY_IMAGE_TOKEN_ESTIMATE` | `1024` | Estimated tokens per image |
| `PROXY_RESIZE_IMAGES` | `1` | Enable base64 image resizing when Pillow is available |
| `PROXY_IMAGE_MAX_DIM` | `1024` | Maximum width or height after resizing |
| `PROXY_IMAGE_RESIZE_MIN_BYTES` | `51200` | Skip resizing below this estimated raw image size |
| `PROXY_IMAGE_MAX_INPUT_BYTES` | `5242880` | Do not decode images above this estimated raw size |
| `PROXY_IMAGE_JPEG_QUALITY` | `85` | JPEG quality for non-transparent images |
| `PROXY_IMAGE_RESIZE_WORKERS` | `2` | Thread-pool workers for image resizing |
| `PROXY_LOG_FILE` | empty | Write logs to this file when set |
| `PROXY_LOG_MAX_BYTES` | `10485760` | Rotating log size |
| `PROXY_LOG_BACKUP_COUNT` | `3` | Rotating log backup count |

## Client Setup

### Claude Code

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8001",
    "ANTHROPIC_AUTH_TOKEN": "local-dummy-key",
    "ANTHROPIC_MODEL": "qwen36-27b-q3km-96k"
  }
}
```

### OpenClaw

Use a dedicated custom provider instead of overriding OpenClaw's built-in
Anthropic provider. See [OPENCLAW.md](OPENCLAW.md) and
[openclaw.models.json](openclaw.models.json).

## Tests

```bash
python -m unittest discover -s tests -v
```

The multimodal tests cover image token estimation, old-image compaction,
`tool_result` nested images, optional Pillow resizing, and list-style `system`
image resizing.

## Files

| File | Purpose |
| --- | --- |
| `context_proxy.py` | Main proxy |
| `requirements.txt` | Required Python dependencies |
| `tests/test_multimodal.py` | Multimodal estimation, resizing, and compaction tests |
| `DEPLOYMENT.md` | Local, remote, and systemd deployment notes |
| `systemd/contextfuse.service.example` | Example systemd unit |
| `OPENCLAW.md` | OpenClaw integration guide |
| `openclaw.models.json` | OpenClaw provider example |
| `DEV_ROADMAP.md` | Development notes |

## Not A Fit When

- Your backend already has reliable automatic context management.
- Your conversations are short and never approach the context limit.
- You need image generation, audio, or video support. ContextFuse currently
  handles image input only.
