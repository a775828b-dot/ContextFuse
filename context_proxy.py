"""
Context compaction proxy for llama-server (v4).
Sits between any client and llama-server, prevents 128K context overflow.

Strategy: Strip non-essential content from old messages.
  - Remove tool_use blocks (function call inputs)
  - Remove tool_result blocks (function call outputs)
  - Remove thinking blocks
  - Keep ALL user text and assistant text intact
  - When pure conversation text reaches CD point (~30K tokens), warn user

This is a safety-net proxy — it only activates when context is about to overflow.
Pure conversation text is never modified. Intelligence loss is zero until the
conversation itself is so long that /new is the only option.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.environ.get("PROXY_BACKEND_URL", "http://localhost:8000")
MODEL_ID = os.environ.get("PROXY_MODEL_ID", "gemma4-26b-iq4xs-128k")
# Informational backend context window. llama-server enforces the real limit;
# the proxy uses this value for user-facing warnings and health output.
CONTEXT_LIMIT = int(os.environ.get("PROXY_CONTEXT_LIMIT", "131072"))
COMPACTION_TRIGGER = int(os.environ.get("PROXY_TRIGGER", "75000"))
COUNT_TOKENS_PRECISE_THRESHOLD = float(os.environ.get("PROXY_COUNT_TOKENS_PRECISE_THRESHOLD", "0.7"))
CD_POINT = int(os.environ.get("PROXY_CD_POINT", "30000"))
RESERVE_TOKENS = int(os.environ.get("PROXY_RESERVE_TOKENS", "8192"))
RECENT_SKIP = int(os.environ.get("PROXY_RECENT_SKIP", "6"))
USE_COUNT_TOKENS = os.environ.get("PROXY_USE_COUNT_TOKENS", "1").lower() not in ("0", "false", "no")
INJECT_STREAM_WARNINGS = os.environ.get("PROXY_INJECT_STREAM_WARNINGS", "1").lower() not in ("0", "false", "no")
TOOL_RESULT_TRUNCATE_TRIGGER_CHARS = int(os.environ.get("PROXY_TOOL_RESULT_TRUNCATE_TRIGGER_CHARS", "20000"))
TOOL_RESULT_TRUNCATE_KEEP_CHARS = int(os.environ.get("PROXY_TOOL_RESULT_TRUNCATE_KEEP_CHARS", "400"))
LOG_FILE = os.environ.get("PROXY_LOG_FILE", "")
LOG_MAX_BYTES = int(os.environ.get("PROXY_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("PROXY_LOG_BACKUP_COUNT", "3"))

log_handlers: list[logging.Handler] = [logging.StreamHandler()]
if LOG_FILE:
    log_handlers.append(
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
    )
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger("context_proxy")

client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(
        base_url=BACKEND_URL,
        timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
    )
    logger.info(
        "Proxy v4 started — backend=%s model=%s trigger=%d cd_point=%d count_tokens=%s precise_threshold=%.2f stream_warnings=%s log_file=%s",
        BACKEND_URL, MODEL_ID, COMPACTION_TRIGGER, CD_POINT,
        USE_COUNT_TOKENS, COUNT_TOKENS_PRECISE_THRESHOLD, INJECT_STREAM_WARNINGS, LOG_FILE or "stdout",
    )
    yield
    await client.aclose()


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Token estimation (chars / 4)
# ---------------------------------------------------------------------------

def _content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            t = block.get("type", "")
            if t == "text":
                total += len(block.get("text", ""))
            elif t == "thinking":
                total += len(block.get("thinking", ""))
            elif t == "tool_use":
                total += len(json.dumps(block.get("input", {}), ensure_ascii=False))
            elif t == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, str):
                    total += len(rc)
                elif isinstance(rc, list):
                    for rb in rc:
                        if isinstance(rb, dict) and rb.get("type") == "text":
                            total += len(rb.get("text", ""))
            else:
                total += 50
        return total
    return 0


def _text_only_chars(content) -> int:
    """Count only user/assistant text chars (no tool/thinking)."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                total += len(block.get("text", ""))
        return total
    return 0


def _tools_chars(tools) -> int:
    """Estimate chars consumed by tools schema definitions."""
    if not tools:
        return 0
    return len(json.dumps(tools, ensure_ascii=False))


def estimate_tokens(body: dict) -> int:
    total = 0
    sys = body.get("system", "")
    if isinstance(sys, str):
        total += len(sys) // 4
    elif isinstance(sys, list):
        total += _content_chars(sys) // 4
    total += _tools_chars(body.get("tools")) // 4
    for msg in body.get("messages", []):
        total += _content_chars(msg.get("content", "")) // 4 + 4
    return total


def estimate_message_tokens(msg: dict) -> int:
    return _content_chars(msg.get("content", "")) // 4 + 4


def estimate_pure_text_tokens(body: dict) -> int:
    """Estimate tokens from text-only content (what remains after stripping)."""
    total = 0
    sys = body.get("system", "")
    if isinstance(sys, str):
        total += len(sys) // 4
    elif isinstance(sys, list):
        total += _text_only_chars(sys) // 4
    for msg in body.get("messages", []):
        total += _text_only_chars(msg.get("content", "")) // 4 + 4
    return total


def _extract_token_count(data: dict) -> int | None:
    """Read token count from Anthropic/llama.cpp compatible responses."""
    for key in ("input_tokens", "count", "tokens", "total_tokens"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    usage = data.get("usage")
    if isinstance(usage, dict):
        value = usage.get("input_tokens")
        if isinstance(value, int):
            return value
    return None


async def count_tokens_precise(body: dict) -> tuple[int | None, str]:
    """Return (token_count, source). Falls back at caller if unavailable."""
    if not USE_COUNT_TOKENS:
        return None, "disabled"
    try:
        resp = await client.post("/v1/messages/count_tokens", json=body, timeout=30.0)
        if resp.status_code != 200:
            return None, f"backend_http_{resp.status_code}"
        count = _extract_token_count(resp.json())
        if count is None:
            return None, "backend_unrecognized"
        return count, "backend_count_tokens"
    except Exception as exc:
        logger.debug("count_tokens unavailable: %r", exc)
        return None, "backend_error"


async def request_token_count(body: dict) -> tuple[int, str]:
    """Use cheap estimation first; call backend only near compaction pressure."""
    estimated = estimate_tokens(body)
    if estimated <= int(COMPACTION_TRIGGER * COUNT_TOKENS_PRECISE_THRESHOLD):
        return estimated, "estimate_chars4"

    precise, source = await count_tokens_precise(body)
    if precise is not None:
        return precise, source
    return estimated, "estimate_chars4"


# ---------------------------------------------------------------------------
# Core compaction: strip tool_use, tool_result, thinking from old messages
# ---------------------------------------------------------------------------

def _strip_with_skip(body: dict, skip: int) -> dict:
    """Strip non-text with a custom skip count (for adaptive compaction)."""
    messages = body["messages"]
    if len(messages) <= skip:
        return body

    cutoff = len(messages) - skip
    new_messages = []

    for idx, msg in enumerate(messages):
        if idx >= cutoff:
            new_messages.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, str) or not isinstance(content, list):
            new_messages.append(msg)
            continue

        new_blocks = []
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue
            block_type = block.get("type", "")
            if block_type in ("tool_use", "tool_result", "thinking"):
                continue
            new_blocks.append(block)

        if new_blocks:
            new_messages.append({**msg, "content": new_blocks})
        else:
            new_messages.append({**msg, "content": [{"type": "text", "text": "[context compacted]"}]})

    final_messages = _sanitize_tool_links(_fix_role_alternation(new_messages))
    return {**body, "messages": final_messages}


def strip_non_text(body: dict) -> dict:
    """Remove tool_use, tool_result, and thinking blocks from old messages.
    Keep all text blocks intact. Skip the most recent RECENT_SKIP messages."""
    messages = body["messages"]
    if len(messages) <= RECENT_SKIP:
        return body

    stripped_tools = 0
    stripped_results = 0
    stripped_thinking = 0
    saved_chars = 0
    new_messages = []

    cutoff = len(messages) - RECENT_SKIP

    for idx, msg in enumerate(messages):
        if idx >= cutoff:
            new_messages.append(msg)
            continue

        content = msg.get("content")

        # String content = pure text, keep as-is
        if isinstance(content, str):
            new_messages.append(msg)
            continue

        if not isinstance(content, list):
            new_messages.append(msg)
            continue

        # Filter content blocks
        new_blocks = []
        msg_changed = False

        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue

            block_type = block.get("type", "")

            if block_type == "tool_use":
                chars = len(json.dumps(block.get("input", {}), ensure_ascii=False))
                saved_chars += chars + 50  # block overhead
                stripped_tools += 1
                msg_changed = True
                # Skip this block entirely

            elif block_type == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, str):
                    saved_chars += len(rc)
                elif isinstance(rc, list):
                    for rb in rc:
                        if isinstance(rb, dict):
                            saved_chars += len(rb.get("text", ""))
                saved_chars += 50
                stripped_results += 1
                msg_changed = True
                # Skip this block entirely

            elif block_type == "thinking":
                saved_chars += len(block.get("thinking", ""))
                stripped_thinking += 1
                msg_changed = True
                # Skip this block entirely

            else:
                # Keep text blocks and anything else
                new_blocks.append(block)

        if msg_changed:
            if new_blocks:
                new_messages.append({**msg, "content": new_blocks})
            else:
                # Message has no remaining content — insert placeholder
                new_messages.append({**msg, "content": [{"type": "text", "text": "[context compacted]"}]})
        else:
            new_messages.append(msg)

    if stripped_tools or stripped_results or stripped_thinking:
        logger.info(
            "Compaction: stripped %d tool_use, %d tool_result, %d thinking blocks; saved ~%d chars (~%d tokens)",
            stripped_tools, stripped_results, stripped_thinking,
            saved_chars, saved_chars // 4,
        )

    # Drop empty message pairs that break role alternation
    final_messages = _sanitize_tool_links(_fix_role_alternation(new_messages))
    return {**body, "messages": final_messages}


def _truncate_middle(text: str) -> tuple[str, int]:
    """Keep the beginning and end of a very large tool result."""
    if len(text) <= TOOL_RESULT_TRUNCATE_TRIGGER_CHARS:
        return text, 0

    keep = max(0, TOOL_RESULT_TRUNCATE_KEEP_CHARS)
    if keep == 0:
        omitted = len(text)
        return f"[tool_result truncated: omitted {omitted} chars]", omitted

    omitted = max(0, len(text) - (keep * 2))
    marker = f"\n\n[tool_result truncated: omitted {omitted} chars]\n\n"
    return text[:keep] + marker + text[-keep:], omitted


def truncate_large_tool_results(body: dict) -> dict:
    """Truncate huge retained tool_result blocks before they can blow the context."""
    messages = body.get("messages", [])
    changed_messages = []
    truncated_blocks = 0
    saved_chars = 0

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            changed_messages.append(msg)
            continue

        new_blocks = []
        msg_changed = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                new_blocks.append(block)
                continue

            result_content = block.get("content", "")
            if isinstance(result_content, str):
                new_text, omitted = _truncate_middle(result_content)
                if omitted:
                    block = {**block, "content": new_text}
                    truncated_blocks += 1
                    saved_chars += omitted
                    msg_changed = True
            elif isinstance(result_content, list):
                new_result_content = []
                block_changed = False
                for rb in result_content:
                    if isinstance(rb, dict) and rb.get("type") == "text":
                        new_text, omitted = _truncate_middle(rb.get("text", ""))
                        if omitted:
                            rb = {**rb, "text": new_text}
                            truncated_blocks += 1
                            saved_chars += omitted
                            block_changed = True
                    new_result_content.append(rb)
                if block_changed:
                    block = {**block, "content": new_result_content}
                    msg_changed = True

            new_blocks.append(block)

        if msg_changed:
            changed_messages.append({**msg, "content": new_blocks})
        else:
            changed_messages.append(msg)

    if truncated_blocks:
        logger.info(
            "Compaction: truncated %d large retained tool_result blocks; saved ~%d chars (~%d tokens)",
            truncated_blocks, saved_chars, saved_chars // 4,
        )
    return {**body, "messages": changed_messages}


def _sanitize_tool_links(messages: list) -> list:
    """Drop tool_result blocks whose matching tool_use was compacted away."""
    if not messages:
        return messages

    available_tool_uses = set()
    removed_orphan_results = 0
    fixed = []

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            fixed.append(msg)
            continue

        new_blocks = []
        changed = False
        for block in content:
            if not isinstance(block, dict):
                new_blocks.append(block)
                continue

            block_type = block.get("type", "")
            if block_type == "tool_use":
                tool_id = block.get("id")
                if tool_id:
                    available_tool_uses.add(tool_id)
                new_blocks.append(block)
            elif block_type == "tool_result":
                tool_use_id = block.get("tool_use_id")
                if tool_use_id and tool_use_id in available_tool_uses:
                    new_blocks.append(block)
                    available_tool_uses.discard(tool_use_id)
                else:
                    removed_orphan_results += 1
                    changed = True
            else:
                new_blocks.append(block)

        if changed:
            if new_blocks:
                fixed.append({**msg, "content": new_blocks})
            else:
                fixed.append({**msg, "content": [{"type": "text", "text": "[tool result compacted]"}]})
        else:
            fixed.append(msg)

    if removed_orphan_results:
        logger.info("Compaction: removed %d orphan tool_result blocks", removed_orphan_results)
    return fixed


def _fix_role_alternation(messages: list) -> list:
    """Ensure valid role alternation (user/assistant/user/assistant...).
    Messages API requires strict alternation starting with user."""
    if not messages:
        return messages

    fixed = []
    expected_role = "user"

    for msg in messages:
        role = msg.get("role", "")
        if role == expected_role:
            fixed.append(msg)
            expected_role = "assistant" if role == "user" else "user"
        elif role == "assistant" and expected_role == "user":
            # Missing user message — inject placeholder
            fixed.append({"role": "user", "content": [{"type": "text", "text": "[continued]"}]})
            fixed.append(msg)
            expected_role = "user"
        elif role == "user" and expected_role == "assistant":
            # Missing assistant message — inject placeholder
            fixed.append({"role": "assistant", "content": [{"type": "text", "text": "[continued]"}]})
            fixed.append(msg)
            expected_role = "assistant"

    return fixed


# ---------------------------------------------------------------------------
# Fallback: if still over limit after stripping, drop oldest messages
# ---------------------------------------------------------------------------

def drop_oldest(body: dict, target_tokens: int) -> dict:
    messages = body["messages"]
    msg_tokens = [estimate_message_tokens(msg) for msg in messages]
    current_tokens = estimate_tokens({**body, "messages": []}) + sum(msg_tokens)
    start = 0
    dropped = 0

    while len(messages) - start > 2 and current_tokens > target_tokens:
        if (
            len(messages) - start >= 2
            and messages[start].get("role") == "user"
            and messages[start + 1].get("role") == "assistant"
        ):
            current_tokens -= msg_tokens[start] + msg_tokens[start + 1]
            start += 2
            dropped += 2
        else:
            current_tokens -= msg_tokens[start]
            start += 1
            dropped += 1

    if dropped:
        logger.info("Fallback: dropped %d oldest messages; estimated remaining=%d tokens", dropped, current_tokens)
    return {**body, "messages": messages[start:]}


# ---------------------------------------------------------------------------
# Main compaction orchestrator
# ---------------------------------------------------------------------------

def compact_request(body: dict, trigger_tokens: int | None = None, trigger_source: str = "unknown") -> tuple[dict, bool]:
    """Returns (compacted_body, should_warn_cd_point)."""
    original_est = estimate_tokens(body)
    if trigger_tokens is not None:
        logger.info(
            "Compaction baseline: trigger_count=%d (%s), local_estimate_before=%d (estimate_chars4)",
            trigger_tokens, trigger_source, original_est,
        )

    # Step 1: Strip non-text content from old messages (with default RECENT_SKIP)
    body = strip_non_text(body)
    est_after = estimate_tokens(body)
    logger.info("After stripping (estimate_chars4): %d -> %d tokens", original_est, est_after)

    # Step 2: If still over limit, preserve but truncate huge retained tool outputs.
    if est_after > COMPACTION_TRIGGER:
        before_truncate = est_after
        body = truncate_large_tool_results(body)
        est_after = estimate_tokens(body)
        if est_after != before_truncate:
            logger.info("After truncating large tool_results: %d -> %d estimated tokens", before_truncate, est_after)

    # Step 3: If still over limit, drop oldest messages
    if est_after > COMPACTION_TRIGGER:
        target = COMPACTION_TRIGGER - RESERVE_TOKENS
        body = drop_oldest(body, target)
        est_after = estimate_tokens(body)
        logger.info("After dropping oldest: %d estimated tokens", est_after)

    # Step 4: If STILL over limit (recent messages themselves are huge),
    # progressively strip recent messages too
    if est_after > COMPACTION_TRIGGER:
        messages = body["messages"]
        skip = max(0, RECENT_SKIP - 2)
        while est_after > COMPACTION_TRIGGER and skip >= 0:
            logger.info("Adaptive: reducing protected messages to %d", skip)
            temp_body = {**body, "messages": messages}
            # Re-strip with smaller skip
            temp_body = _strip_with_skip(temp_body, skip)
            est_after = estimate_tokens(temp_body)
            if est_after <= COMPACTION_TRIGGER:
                body = temp_body
                break
            skip -= 2
        if est_after > COMPACTION_TRIGGER:
            # Last resort: drop oldest again after aggressive stripping
            body = drop_oldest(body, COMPACTION_TRIGGER - RESERVE_TOKENS)
            est_after = estimate_tokens(body)
            logger.info("After aggressive compaction: %d estimated tokens", est_after)

    # Check CD point
    pure_text_est = estimate_pure_text_tokens(body)
    should_warn = pure_text_est >= CD_POINT
    if should_warn:
        logger.info("CD POINT reached: ~%d tokens after compaction (threshold=%d)", pure_text_est, CD_POINT)

    return body, should_warn


# ---------------------------------------------------------------------------
# Warning injection (post-generation, appended to response)
# ---------------------------------------------------------------------------

CD_WARNING = (
    "\n\n---\n"
    "[Context Proxy 提醒] 当前会话已变长：纯文本约 {text_k}K tokens，"
    "后端窗口约 {limit_k}K tokens。建议完成当前小任务后新开会话，"
    "并让代理带一份任务摘要过去，以保持速度和准确性。"
)


def format_cd_warning(pure_text_tokens: int) -> str:
    return CD_WARNING.format(
        text_k=pure_text_tokens // 1000,
        limit_k=CONTEXT_LIMIT // 1000,
    )


def inject_warning_streaming(index: int, pure_text_tokens: int) -> bytes:
    """Return a text_delta SSE event for an active text content block."""
    warning_text = format_cd_warning(pure_text_tokens)
    event_data = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": warning_text},
    }
    return f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n".encode("utf-8")


def parse_sse_json(frame: bytes) -> dict | None:
    """Parse a complete SSE frame and return its JSON data payload, if any."""
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError:
        return None

    data_lines = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None

    data_text = "\n".join(data_lines)
    if not data_text or data_text == "[DONE]":
        return None
    try:
        payload = json.loads(data_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def inject_warning_non_streaming(response_body: dict, pure_text_tokens: int) -> dict:
    """Append warning to the last text block in a non-streaming response."""
    warning_text = format_cd_warning(pure_text_tokens)
    content = response_body.get("content", [])
    for block in reversed(content):
        if isinstance(block, dict) and block.get("type") == "text":
            block["text"] += warning_text
            break
    return response_body


# ---------------------------------------------------------------------------
# Proxy routes
# ---------------------------------------------------------------------------

def forward_headers(headers) -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": headers.get("accept", "application/json"),
        "X-Api-Key": headers.get("x-api-key", ""),
        "Authorization": headers.get("authorization", ""),
        "anthropic-version": headers.get("anthropic-version", ""),
    }


def static_models_response() -> dict:
    return {
        "data": [
            {
                "id": MODEL_ID,
                "type": "model",
                "display_name": MODEL_ID,
            }
        ]
    }


def json_response(payload: dict, status_code: int) -> Response:
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        status_code=status_code,
        media_type="application/json",
    )


async def parse_json_body(request: Request) -> tuple[dict | None, Response | None]:
    try:
        body = await request.json()
    except Exception:
        return None, json_response({"error": "invalid JSON body"}, 400)
    if not isinstance(body, dict):
        return None, json_response({"error": "JSON body must be an object"}, 400)
    return body, None


def backend_error_response(exc: Exception) -> Response:
    if isinstance(exc, httpx.ConnectError):
        return json_response(
            {"error": "backend unreachable", "backend_url": BACKEND_URL},
            502,
        )
    if isinstance(exc, httpx.ReadTimeout):
        return json_response(
            {"error": "backend timeout", "timeout_seconds": 600, "backend_url": BACKEND_URL},
            504,
        )
    if isinstance(exc, httpx.WriteTimeout):
        return json_response(
            {"error": "backend write timeout", "backend_url": BACKEND_URL},
            504,
        )
    if isinstance(exc, httpx.HTTPError):
        return json_response(
            {"error": "backend request failed", "detail": type(exc).__name__, "backend_url": BACKEND_URL},
            502,
        )
    return json_response(
        {"error": "proxy request failed", "detail": type(exc).__name__},
        500,
    )


@app.get("/health")
async def health():
    backend = "disconnected"
    backend_status = None
    try:
        resp = await client.get("/health", timeout=5.0)
        backend_status = resp.status_code
        if resp.status_code < 500:
            backend = "connected"
    except Exception as exc:
        backend_status = type(exc).__name__
        try:
            resp = await client.get("/v1/models", timeout=5.0)
            backend_status = resp.status_code
            if resp.status_code < 500:
                backend = "connected"
        except Exception as exc2:
            backend_status = type(exc2).__name__

    return {
        "status": "ok",
        "backend": backend,
        "backend_status": backend_status,
        "backend_url": BACKEND_URL,
        "model": MODEL_ID,
        "context_limit": CONTEXT_LIMIT,
        "compaction_trigger": COMPACTION_TRIGGER,
        "cd_point": CD_POINT,
        "reserve_tokens": RESERVE_TOKENS,
        "recent_skip": RECENT_SKIP,
        "count_tokens": USE_COUNT_TOKENS,
        "count_tokens_precise_threshold": COUNT_TOKENS_PRECISE_THRESHOLD,
        "stream_warnings": INJECT_STREAM_WARNINGS,
        "tool_result_truncate_trigger_chars": TOOL_RESULT_TRUNCATE_TRIGGER_CHARS,
        "tool_result_truncate_keep_chars": TOOL_RESULT_TRUNCATE_KEEP_CHARS,
        "log_file": LOG_FILE or "stdout",
        "log_max_bytes": LOG_MAX_BYTES if LOG_FILE else None,
        "log_backup_count": LOG_BACKUP_COUNT if LOG_FILE else None,
    }


@app.get("/v1/models")
async def proxy_models(request: Request):
    try:
        resp = await client.get("/v1/models", headers=forward_headers(request.headers), timeout=10.0)
        if resp.status_code == 200:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        logger.info("Backend /v1/models returned %d; using static model list", resp.status_code)
    except Exception as exc:
        logger.info("Backend /v1/models unavailable (%r); using static model list", exc)

    return Response(
        content=json.dumps(static_models_response(), ensure_ascii=False),
        status_code=200,
        media_type="application/json",
    )


@app.post("/v1/messages/count_tokens")
async def proxy_count_tokens(request: Request):
    body, error = await parse_json_body(request)
    if error:
        return error
    try:
        resp = await client.post(
            "/v1/messages/count_tokens",
            json=body,
            headers=forward_headers(request.headers),
            timeout=30.0,
        )
        if resp.status_code == 200:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        logger.info("Backend count_tokens returned %d; using estimate", resp.status_code)
        backend_status = resp.status_code
    except Exception as exc:
        logger.info("Backend count_tokens unavailable (%r); using estimate", exc)
        backend_status = type(exc).__name__

    estimated = estimate_tokens(body)
    payload = {
        "input_tokens": estimated,
        "estimated": True,
        "source": "estimate_chars4",
        "backend_status": backend_status,
    }
    return Response(
        content=json.dumps(payload, ensure_ascii=False),
        status_code=200,
        media_type="application/json",
    )


@app.post("/v1/messages")
async def proxy_messages(request: Request):
    body, error = await parse_json_body(request)
    if error:
        return error
    if "messages" not in body or not isinstance(body["messages"], list):
        return json_response({"error": "missing or invalid 'messages' field"}, 400)

    est, token_source = await request_token_count(body)
    num_msgs = len(body.get("messages", []))
    is_streaming = body.get("stream", False)
    should_warn = False
    pure_text_est = 0

    logger.info(
        "Request: ~%d tokens (%s), %d messages, stream=%s",
        est, token_source, num_msgs, is_streaming,
    )

    if est > COMPACTION_TRIGGER:
        logger.info("Trigger hit (%d > %d) — compacting", est, COMPACTION_TRIGGER)
        body, should_warn = compact_request(body, est, token_source)
        pure_text_est = estimate_pure_text_tokens(body)
    else:
        # Still check CD point even without compaction
        pure_text_est = estimate_pure_text_tokens(body)
        if pure_text_est >= CD_POINT:
            should_warn = True
            logger.info("CD POINT (no compaction): pure text ~%d tokens", pure_text_est)

    if is_streaming:
        req = client.build_request(
            "POST", "/v1/messages",
            json=body,
            headers=forward_headers(request.headers),
        )
        try:
            resp = await client.send(req, stream=True)
        except Exception as exc:
            logger.warning("Backend streaming request failed: %r", exc)
            return backend_error_response(exc)
        if should_warn and not INJECT_STREAM_WARNINGS:
            logger.info("CD warning suppressed for streaming response; sent as headers only")

        async def stream_with_warning():
            if not should_warn or not INJECT_STREAM_WARNINGS:
                async for chunk in resp.aiter_raw():
                    yield chunk
                return

            injected = False
            active_text_indexes: set[int] = set()
            pending_text_stop: tuple[int, bytes] | None = None
            buffer = b""
            async for chunk in resp.aiter_raw():
                buffer += chunk
                while b"\n\n" in buffer:
                    frame, buffer = buffer.split(b"\n\n", 1)
                    frame_bytes = frame + b"\n\n"
                    payload = parse_sse_json(frame)

                    if payload:
                        event_type = payload.get("type")
                        index = payload.get("index")

                        if pending_text_stop is not None:
                            pending_index, pending_frame = pending_text_stop
                            if event_type in ("message_delta", "message_stop"):
                                if not injected:
                                    yield inject_warning_streaming(pending_index, pure_text_est)
                                    injected = True
                                yield pending_frame
                                pending_text_stop = None
                            elif event_type == "content_block_start":
                                yield pending_frame
                                pending_text_stop = None
                            elif event_type != "content_block_stop":
                                yield pending_frame
                                pending_text_stop = None

                        if (
                            event_type == "content_block_start"
                            and isinstance(index, int)
                            and isinstance(payload.get("content_block"), dict)
                            and payload["content_block"].get("type") == "text"
                        ):
                            active_text_indexes.add(index)

                        if (
                            not injected
                            and event_type == "content_block_stop"
                            and isinstance(index, int)
                            and index in active_text_indexes
                        ):
                            pending_text_stop = (index, frame_bytes)
                            active_text_indexes.discard(index)
                            continue

                        if event_type == "content_block_stop" and isinstance(index, int):
                            active_text_indexes.discard(index)

                    yield frame_bytes

            if buffer:
                yield buffer
            if pending_text_stop is not None:
                pending_index, pending_frame = pending_text_stop
                if not injected:
                    yield inject_warning_streaming(pending_index, pure_text_est)
                    injected = True
                yield pending_frame
            if not injected:
                logger.info("CD warning was not injected because no active text block was found")

        stream_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        if should_warn:
            stream_headers["X-Context-Proxy-Warning"] = "cd_point"
            stream_headers["X-Context-Proxy-Pure-Text-Tokens"] = str(pure_text_est)

        return StreamingResponse(
            stream_with_warning(),
            status_code=resp.status_code,
            headers=stream_headers,
            background=BackgroundTask(resp.aclose),
        )
    else:
        try:
            resp = await client.post(
                "/v1/messages",
                json=body,
                headers=forward_headers(request.headers),
            )
        except Exception as exc:
            logger.warning("Backend non-streaming request failed: %r", exc)
            return backend_error_response(exc)
        if should_warn and resp.status_code == 200:
            try:
                resp_body = resp.json()
                resp_body = inject_warning_non_streaming(resp_body, pure_text_est)
                return Response(
                    content=json.dumps(resp_body, ensure_ascii=False),
                    status_code=resp.status_code,
                    media_type="application/json",
                )
            except Exception:
                pass
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def catch_all(request: Request, path: str):
    url = f"/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    request_kwargs = {
        "method": request.method,
        "url": url,
        "headers": forward_headers(request.headers),
    }
    if request.method.upper() not in ("GET", "DELETE", "OPTIONS"):
        request_kwargs["content"] = await request.body()

    try:
        resp = await client.request(**request_kwargs)
    except Exception as exc:
        logger.warning("Backend catch-all request failed: %r", exc)
        return backend_error_response(exc)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
