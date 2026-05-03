"""
ContextFuse: context compaction proxy for llama-server.

It sits between Anthropic Messages compatible clients and llama-server. The
proxy keeps normal requests cheap, but compacts long agent histories before
llama-server returns a context overflow error. It also understands image blocks
well enough to resize base64 images, estimate image cost, and compact old image
content safely for multimodal models.
"""
import asyncio
import base64
import binascii
import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

try:
    from PIL import Image, ImageOps
    HAS_PILLOW = True
except ImportError:
    Image = None
    ImageOps = None
    HAS_PILLOW = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BACKEND_URL = os.environ.get("PROXY_BACKEND_URL", "http://localhost:8000")
MODEL_ID = os.environ.get("PROXY_MODEL_ID", "gemma4-26b-iq4xs-128k")
# Informational backend context window. llama-server enforces the real limit;
# this value is used for warnings and health output.
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
IMAGE_TOKEN_ESTIMATE = int(os.environ.get("PROXY_IMAGE_TOKEN_ESTIMATE", "1024"))
RESIZE_IMAGES = os.environ.get("PROXY_RESIZE_IMAGES", "1").lower() not in ("0", "false", "no")
IMAGE_MAX_DIM = int(os.environ.get("PROXY_IMAGE_MAX_DIM", "1024"))
IMAGE_RESIZE_MIN_BYTES = int(os.environ.get("PROXY_IMAGE_RESIZE_MIN_BYTES", "51200"))
IMAGE_MAX_INPUT_BYTES = int(os.environ.get("PROXY_IMAGE_MAX_INPUT_BYTES", "5242880"))
IMAGE_JPEG_QUALITY = int(os.environ.get("PROXY_IMAGE_JPEG_QUALITY", "85"))
IMAGE_RESIZE_WORKERS = int(os.environ.get("PROXY_IMAGE_RESIZE_WORKERS", "2"))
LOG_FILE = os.environ.get("PROXY_LOG_FILE", "")
LOG_MAX_BYTES = int(os.environ.get("PROXY_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("PROXY_LOG_BACKUP_COUNT", "3"))

log_handlers: list[logging.Handler] = [logging.StreamHandler()]
if LOG_FILE:
    log_handlers.append(
        RotatingFileHandler(LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8")
    )
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=log_handlers)
logger = logging.getLogger("context_proxy")

client: httpx.AsyncClient | None = None
image_executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, image_executor
    client = httpx.AsyncClient(
        base_url=BACKEND_URL,
        timeout=httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0),
    )
    if RESIZE_IMAGES and HAS_PILLOW and IMAGE_RESIZE_WORKERS > 0:
        image_executor = ThreadPoolExecutor(max_workers=IMAGE_RESIZE_WORKERS)
    elif RESIZE_IMAGES and not HAS_PILLOW:
        logger.warning("Pillow not installed - image resizing disabled; estimation and stripping remain active")
    logger.info(
        "ContextFuse started backend=%s model=%s trigger=%d cd_point=%d count_tokens=%s image_resize=%s",
        BACKEND_URL,
        MODEL_ID,
        COMPACTION_TRIGGER,
        CD_POINT,
        USE_COUNT_TOKENS,
        bool(image_executor),
    )
    yield
    await client.aclose()
    if image_executor:
        image_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _image_token_chars() -> int:
    return max(0, IMAGE_TOKEN_ESTIMATE) * 4


def _estimated_base64_bytes(data: str) -> int:
    if not isinstance(data, str) or not data:
        return 0
    length = len(data) - sum(data.count(ch) for ch in " \t\n\r")
    stripped = data.rstrip()
    padding = 2 if stripped.endswith("==") else 1 if stripped.endswith("=") else 0
    return max(0, (length * 3) // 4 - padding)


def _format_bytes(size: int | None) -> str:
    if not isinstance(size, int) or size < 0:
        return ""
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{round(size / 1024)}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    header, sep, data = url.partition(",")
    if not sep or ";base64" not in header:
        return None
    media_type = header[5:].split(";", 1)[0] or "image/jpeg"
    return media_type, data


def _image_block_info(block: dict) -> dict | None:
    if not isinstance(block, dict):
        return None
    block_type = block.get("type")
    if block_type == "image":
        source = block.get("source")
        if not isinstance(source, dict):
            return {"media_type": "image", "bytes": None, "base64": None, "url": False}
        source_type = source.get("type")
        media_type = source.get("media_type") or "image"
        if source_type == "base64" and isinstance(source.get("data"), str):
            data = source["data"]
            return {"media_type": media_type, "bytes": _estimated_base64_bytes(data), "base64": data, "url": False}
        if source_type == "url":
            return {"media_type": media_type, "bytes": None, "base64": None, "url": True}
        return {"media_type": media_type, "bytes": None, "base64": None, "url": False}
    if block_type == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        parsed = _parse_data_url(url)
        if parsed:
            media_type, data = parsed
            return {"media_type": media_type, "bytes": _estimated_base64_bytes(data), "base64": data, "url": False}
        return {"media_type": "image", "bytes": None, "base64": None, "url": bool(url)}
    return None


def _is_image_block(block: dict) -> bool:
    return _image_block_info(block) is not None


def _image_placeholder(block: dict) -> dict:
    info = _image_block_info(block) or {}
    media_type = info.get("media_type") or "image"
    size_text = _format_bytes(info.get("bytes"))
    width = block.get("width")
    height = block.get("height")
    dims = f", {width}x{height}" if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0 else ""
    size = f", ~{size_text}" if size_text else ""
    return {"type": "text", "text": f"[image compacted: {media_type}{size}{dims}]"}


def _image_saved_chars(block: dict) -> int:
    info = _image_block_info(block) or {}
    size = info.get("bytes")
    if isinstance(size, int) and size > 0:
        return max(_image_token_chars(), size)
    return _image_token_chars()


def _has_transparency(img) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    return img.mode == "P" and "transparency" in img.info


def _resize_image_sync(data_b64: str, media_type: str) -> dict:
    try:
        compact_data = "".join(data_b64.split())
        raw = base64.b64decode(compact_data, validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            img = ImageOps.exif_transpose(opened)
            original_size = img.size
            longest = max(original_size) if original_size else 0
            if longest > IMAGE_MAX_DIM > 0:
                scale = IMAGE_MAX_DIM / longest
                new_size = (
                    max(1, int(round(original_size[0] * scale))),
                    max(1, int(round(original_size[1] * scale))),
                )
                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                img = img.resize(new_size, resampling)
            else:
                new_size = original_size
            output = io.BytesIO()
            if _has_transparency(img):
                out_media = "image/png"
                if img.mode not in ("RGBA", "LA"):
                    img = img.convert("RGBA")
                img.save(output, format="PNG", optimize=True)
            else:
                out_media = "image/jpeg"
                img = img.convert("RGB")
                img.save(output, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)
        resized = output.getvalue()
        return {
            "ok": True,
            "data": base64.b64encode(resized).decode("ascii"),
            "media_type": out_media,
            "original_bytes": len(raw),
            "new_bytes": len(resized),
            "original_size": original_size,
            "new_size": new_size,
            "changed": resized != raw or out_media != media_type,
        }
    except Exception as exc:
        logger.debug("Image resize failed (%r), keeping original", exc)
        return {"ok": False, "data": data_b64, "media_type": media_type, "error": type(exc).__name__}


async def _resize_image_data(data_b64: str, media_type: str) -> dict:
    if image_executor is None:
        return {"ok": False, "data": data_b64, "media_type": media_type, "error": "resize_executor_unavailable"}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(image_executor, _resize_image_sync, data_b64, media_type)


async def _resize_image_block(block: dict, stats: dict) -> tuple[dict, bool]:
    info = _image_block_info(block)
    if not info:
        return block, False
    stats["images"] += 1
    data_b64 = info.get("base64")
    if not data_b64:
        stats["remote_or_unknown"] += 1
        return block, False
    stats["base64"] += 1
    estimated_bytes = info.get("bytes") or 0
    if not RESIZE_IMAGES or not HAS_PILLOW:
        stats["resize_disabled"] += 1
        return block, False
    if estimated_bytes < IMAGE_RESIZE_MIN_BYTES:
        stats["skipped_small"] += 1
        return block, False
    if estimated_bytes > IMAGE_MAX_INPUT_BYTES:
        stats["skipped_too_large"] += 1
        return block, False
    result = await _resize_image_data(data_b64, info.get("media_type") or "image/jpeg")
    if not result.get("ok"):
        stats["failed"] += 1
        return block, False
    if not result.get("changed"):
        stats["unchanged"] += 1
        return block, False
    stats["resized"] += 1
    stats["bytes_saved"] += max(0, result.get("original_bytes", 0) - result.get("new_bytes", 0))
    if block.get("type") == "image":
        source = {**block.get("source", {})}
        source["media_type"] = result["media_type"]
        source["data"] = result["data"]
        return {**block, "source": source}, True
    image_url = block.get("image_url")
    new_url = f"data:{result['media_type']};base64,{result['data']}"
    if isinstance(image_url, dict):
        return {**block, "image_url": {**image_url, "url": new_url}}, True
    return {**block, "image_url": new_url}, True


async def _resize_content_images(content, stats: dict):
    if not isinstance(content, list):
        return content, False
    changed = False
    new_content = []
    for block in content:
        if not isinstance(block, dict):
            new_content.append(block)
            continue
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            nested, nested_changed = await _resize_content_images(block["content"], stats)
            if nested_changed:
                block = {**block, "content": nested}
                changed = True
            new_content.append(block)
            continue
        new_block, block_changed = await _resize_image_block(block, stats)
        changed = changed or block_changed
        new_content.append(new_block)
    return new_content, changed


async def resize_all_images(body: dict) -> dict:
    stats = {
        "images": 0,
        "base64": 0,
        "resized": 0,
        "bytes_saved": 0,
        "remote_or_unknown": 0,
        "resize_disabled": 0,
        "skipped_small": 0,
        "skipped_too_large": 0,
        "failed": 0,
        "unchanged": 0,
    }
    changed = False
    system = body.get("system")
    new_system = system
    if isinstance(system, list):
        new_system, system_changed = await _resize_content_images(system, stats)
        changed = changed or system_changed
    new_messages = []
    for msg in body.get("messages", []):
        content, content_changed = await _resize_content_images(msg.get("content"), stats)
        if content_changed:
            new_messages.append({**msg, "content": content})
            changed = True
        else:
            new_messages.append(msg)
    if stats["images"]:
        logger.info(
            "Image processing: images=%d base64=%d resized=%d saved=%s skipped_small=%d skipped_too_large=%d remote_or_unknown=%d failed=%d",
            stats["images"],
            stats["base64"],
            stats["resized"],
            _format_bytes(stats["bytes_saved"]),
            stats["skipped_small"],
            stats["skipped_too_large"],
            stats["remote_or_unknown"],
            stats["failed"],
        )
    return {**body, "system": new_system, "messages": new_messages} if changed else body

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type == "text":
                total += len(block.get("text", ""))
            elif block_type == "thinking":
                total += len(block.get("thinking", ""))
            elif block_type == "tool_use":
                total += len(json.dumps(block.get("input", {}), ensure_ascii=False))
            elif block_type == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, str):
                    total += len(rc)
                elif isinstance(rc, list):
                    for rb in rc:
                        if isinstance(rb, dict):
                            if rb.get("type") == "text":
                                total += len(rb.get("text", ""))
                            elif _is_image_block(rb):
                                total += _image_token_chars()
            elif _is_image_block(block):
                total += _image_token_chars()
            else:
                total += 50
        return total
    return 0


def _text_only_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text")
    return 0


def _tools_chars(tools) -> int:
    return len(json.dumps(tools, ensure_ascii=False)) if tools else 0


def estimate_tokens(body: dict) -> int:
    total = 0
    system = body.get("system", "")
    if isinstance(system, str):
        total += len(system) // 4
    elif isinstance(system, list):
        total += _content_chars(system) // 4
    total += _tools_chars(body.get("tools")) // 4
    for msg in body.get("messages", []):
        total += _content_chars(msg.get("content", "")) // 4 + 4
    return total


def estimate_message_tokens(msg: dict) -> int:
    return _content_chars(msg.get("content", "")) // 4 + 4


def estimate_pure_text_tokens(body: dict) -> int:
    total = 0
    system = body.get("system", "")
    if isinstance(system, str):
        total += len(system) // 4
    elif isinstance(system, list):
        total += _text_only_chars(system) // 4
    for msg in body.get("messages", []):
        total += _text_only_chars(msg.get("content", "")) // 4 + 4
    return total


def _extract_token_count(data: dict) -> int | None:
    for key in ("input_tokens", "count", "tokens", "total_tokens"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    usage = data.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("input_tokens"), int):
        return usage["input_tokens"]
    return None


async def count_tokens_precise(body: dict) -> tuple[int | None, str]:
    if not USE_COUNT_TOKENS:
        return None, "disabled"
    try:
        resp = await client.post("/v1/messages/count_tokens", json=body, timeout=30.0)
        if resp.status_code != 200:
            return None, f"backend_http_{resp.status_code}"
        count = _extract_token_count(resp.json())
        return (count, "backend_count_tokens") if count is not None else (None, "backend_unrecognized")
    except Exception as exc:
        logger.debug("count_tokens unavailable: %r", exc)
        return None, "backend_error"


async def request_token_count(body: dict) -> tuple[int, str]:
    estimated = estimate_tokens(body)
    if estimated <= int(COMPACTION_TRIGGER * COUNT_TOKENS_PRECISE_THRESHOLD):
        return estimated, "estimate_chars4"
    precise, source = await count_tokens_precise(body)
    if precise is not None:
        return precise, source
    return estimated, "estimate_chars4"

# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def _sanitize_tool_links(messages: list) -> list:
    available_tool_uses = set()
    fixed = []
    removed = 0
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
                    removed += 1
                    changed = True
            else:
                new_blocks.append(block)
        fixed.append({**msg, "content": new_blocks or [{"type": "text", "text": "[tool result compacted]"}]} if changed else msg)
    if removed:
        logger.info("Compaction: removed %d orphan tool_result blocks", removed)
    return fixed


def _fix_role_alternation(messages: list) -> list:
    if not messages:
        return messages
    fixed = []
    expected = "user"
    for msg in messages:
        role = msg.get("role", "")
        if role == expected:
            fixed.append(msg)
            expected = "assistant" if role == "user" else "user"
        elif role == "assistant" and expected == "user":
            fixed.append({"role": "user", "content": [{"type": "text", "text": "[continued]"}]})
            fixed.append(msg)
            expected = "user"
        elif role == "user" and expected == "assistant":
            fixed.append({"role": "assistant", "content": [{"type": "text", "text": "[continued]"}]})
            fixed.append(msg)
            expected = "assistant"
    return fixed


def _strip_message_content(content, strip_images: bool = True) -> tuple[list | str, dict]:
    stats = {"tool_use": 0, "tool_result": 0, "thinking": 0, "image": 0, "saved_chars": 0, "changed": False}
    if isinstance(content, str) or not isinstance(content, list):
        return content, stats
    new_blocks = []
    for block in content:
        if not isinstance(block, dict):
            new_blocks.append(block)
            continue
        block_type = block.get("type", "")
        if block_type == "tool_use":
            stats["tool_use"] += 1
            stats["saved_chars"] += len(json.dumps(block.get("input", {}), ensure_ascii=False)) + 50
            stats["changed"] = True
            continue
        if block_type == "tool_result":
            stats["tool_result"] += 1
            stats["saved_chars"] += _content_chars(block.get("content", "")) + 50
            stats["changed"] = True
            continue
        if block_type == "thinking":
            stats["thinking"] += 1
            stats["saved_chars"] += len(block.get("thinking", ""))
            stats["changed"] = True
            continue
        if strip_images and _is_image_block(block):
            stats["image"] += 1
            stats["saved_chars"] += _image_saved_chars(block)
            stats["changed"] = True
            new_blocks.append(_image_placeholder(block))
            continue
        new_blocks.append(block)
    return new_blocks or [{"type": "text", "text": "[context compacted]"}], stats


def _strip_with_skip(body: dict, skip: int) -> dict:
    messages = body["messages"]
    if len(messages) <= skip:
        return body
    cutoff = len(messages) - skip
    new_messages = []
    for idx, msg in enumerate(messages):
        if idx >= cutoff:
            new_messages.append(msg)
            continue
        new_content, stats = _strip_message_content(msg.get("content"), strip_images=True)
        new_messages.append({**msg, "content": new_content} if stats["changed"] else msg)
    return {**body, "messages": _sanitize_tool_links(_fix_role_alternation(new_messages))}


def strip_non_text(body: dict) -> dict:
    messages = body["messages"]
    if len(messages) <= RECENT_SKIP:
        return body
    cutoff = len(messages) - RECENT_SKIP
    totals = {"tool_use": 0, "tool_result": 0, "thinking": 0, "image": 0, "saved_chars": 0}
    new_messages = []
    for idx, msg in enumerate(messages):
        if idx >= cutoff:
            new_messages.append(msg)
            continue
        new_content, stats = _strip_message_content(msg.get("content"), strip_images=True)
        for key in totals:
            totals[key] += stats.get(key, 0)
        new_messages.append({**msg, "content": new_content} if stats["changed"] else msg)
    if any(totals[k] for k in ("tool_use", "tool_result", "thinking", "image")):
        logger.info(
            "Compaction: stripped %d tool_use, %d tool_result, %d thinking, %d image blocks; saved ~%d chars (~%d tokens)",
            totals["tool_use"],
            totals["tool_result"],
            totals["thinking"],
            totals["image"],
            totals["saved_chars"],
            totals["saved_chars"] // 4,
        )
    return {**body, "messages": _sanitize_tool_links(_fix_role_alternation(new_messages))}


def _truncate_middle(text: str) -> tuple[str, int]:
    if len(text) <= TOOL_RESULT_TRUNCATE_TRIGGER_CHARS:
        return text, 0
    keep = max(0, TOOL_RESULT_TRUNCATE_KEEP_CHARS)
    if keep == 0:
        return f"[tool_result truncated: omitted {len(text)} chars]", len(text)
    omitted = max(0, len(text) - keep * 2)
    marker = f"\n\n[tool_result truncated: omitted {omitted} chars]\n\n"
    return text[:keep] + marker + text[-keep:], omitted


def truncate_large_tool_results(body: dict) -> dict:
    changed_messages = []
    truncated_blocks = 0
    saved_chars = 0
    for msg in body.get("messages", []):
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
            rc = block.get("content", "")
            if isinstance(rc, str):
                new_text, omitted = _truncate_middle(rc)
                if omitted:
                    block = {**block, "content": new_text}
                    truncated_blocks += 1
                    saved_chars += omitted
                    msg_changed = True
            elif isinstance(rc, list):
                nested = []
                block_changed = False
                for rb in rc:
                    if isinstance(rb, dict) and rb.get("type") == "text":
                        new_text, omitted = _truncate_middle(rb.get("text", ""))
                        if omitted:
                            rb = {**rb, "text": new_text}
                            truncated_blocks += 1
                            saved_chars += omitted
                            block_changed = True
                    elif isinstance(rb, dict) and _is_image_block(rb):
                        saved_chars += _image_saved_chars(rb)
                        rb = _image_placeholder(rb)
                        truncated_blocks += 1
                        block_changed = True
                    nested.append(rb)
                if block_changed:
                    block = {**block, "content": nested}
                    msg_changed = True
            new_blocks.append(block)
        changed_messages.append({**msg, "content": new_blocks} if msg_changed else msg)
    if truncated_blocks:
        logger.info(
            "Compaction: truncated %d large retained tool_result blocks; saved ~%d chars (~%d tokens)",
            truncated_blocks,
            saved_chars,
            saved_chars // 4,
        )
    return {**body, "messages": changed_messages}


def drop_oldest(body: dict, target_tokens: int) -> dict:
    messages = body["messages"]
    msg_tokens = [estimate_message_tokens(msg) for msg in messages]
    current_tokens = estimate_tokens({**body, "messages": []}) + sum(msg_tokens)
    start = 0
    dropped = 0
    while len(messages) - start > 2 and current_tokens > target_tokens:
        if len(messages) - start >= 2 and messages[start].get("role") == "user" and messages[start + 1].get("role") == "assistant":
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


def compact_request(body: dict, trigger_tokens: int | None = None, trigger_source: str = "unknown") -> tuple[dict, bool]:
    original_est = estimate_tokens(body)
    if trigger_tokens is not None:
        logger.info(
            "Compaction baseline: trigger_count=%d (%s), local_estimate_before=%d (estimate_chars4)",
            trigger_tokens,
            trigger_source,
            original_est,
        )
    body = strip_non_text(body)
    est_after = estimate_tokens(body)
    logger.info("After stripping (estimate_chars4): %d -> %d tokens", original_est, est_after)
    if est_after > COMPACTION_TRIGGER:
        before = est_after
        body = truncate_large_tool_results(body)
        est_after = estimate_tokens(body)
        if est_after != before:
            logger.info("After truncating large tool_results: %d -> %d estimated tokens", before, est_after)
    if est_after > COMPACTION_TRIGGER:
        body = drop_oldest(body, COMPACTION_TRIGGER - RESERVE_TOKENS)
        est_after = estimate_tokens(body)
        logger.info("After dropping oldest: %d estimated tokens", est_after)
    if est_after > COMPACTION_TRIGGER:
        messages = body["messages"]
        skip = max(0, RECENT_SKIP - 2)
        while est_after > COMPACTION_TRIGGER and skip >= 0:
            logger.info("Adaptive: reducing protected messages to %d", skip)
            temp_body = _strip_with_skip({**body, "messages": messages}, skip)
            est_after = estimate_tokens(temp_body)
            if est_after <= COMPACTION_TRIGGER:
                body = temp_body
                break
            skip -= 2
        if est_after > COMPACTION_TRIGGER:
            body = drop_oldest(body, COMPACTION_TRIGGER - RESERVE_TOKENS)
            est_after = estimate_tokens(body)
            logger.info("After aggressive compaction: %d estimated tokens", est_after)
    pure_text_est = estimate_pure_text_tokens(body)
    should_warn = pure_text_est >= CD_POINT
    if should_warn:
        logger.info("CD POINT reached: ~%d tokens after compaction (threshold=%d)", pure_text_est, CD_POINT)
    return body, should_warn

# ---------------------------------------------------------------------------
# Warning injection
# ---------------------------------------------------------------------------

CD_WARNING = (
    "\n\n---\n"
    "[Context Proxy 提醒] 当前会话已变长：纯文本约 {text_k}K tokens，"
    "后端窗口约 {limit_k}K tokens。建议完成当前小任务后新开会话，"
    "并让代理带一份任务摘要过去，以保持速度和准确性。"
)


def format_cd_warning(pure_text_tokens: int) -> str:
    return CD_WARNING.format(text_k=pure_text_tokens // 1000, limit_k=CONTEXT_LIMIT // 1000)


def inject_warning_streaming(index: int, pure_text_tokens: int) -> bytes:
    event_data = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": format_cd_warning(pure_text_tokens)},
    }
    return f"event: content_block_delta\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode("utf-8")


def parse_sse_json(frame: bytes) -> dict | None:
    try:
        text = frame.decode("utf-8")
    except UnicodeDecodeError:
        return None
    data_lines = [line[5:].lstrip() for line in text.splitlines() if line.startswith("data:")]
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
    for block in reversed(response_body.get("content", [])):
        if isinstance(block, dict) and block.get("type") == "text":
            block["text"] += format_cd_warning(pure_text_tokens)
            break
    return response_body

# ---------------------------------------------------------------------------
# Routes
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
    return {"data": [{"id": MODEL_ID, "type": "model", "display_name": MODEL_ID}]}


def json_response(payload: dict, status_code: int) -> Response:
    return Response(content=json.dumps(payload, ensure_ascii=False), status_code=status_code, media_type="application/json")


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
        return json_response({"error": "backend unreachable", "backend_url": BACKEND_URL}, 502)
    if isinstance(exc, httpx.ReadTimeout):
        return json_response({"error": "backend timeout", "timeout_seconds": 600, "backend_url": BACKEND_URL}, 504)
    if isinstance(exc, httpx.WriteTimeout):
        return json_response({"error": "backend write timeout", "backend_url": BACKEND_URL}, 504)
    if isinstance(exc, httpx.HTTPError):
        return json_response({"error": "backend request failed", "detail": type(exc).__name__, "backend_url": BACKEND_URL}, 502)
    return json_response({"error": "proxy request failed", "detail": type(exc).__name__}, 500)


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
        "image_token_estimate": IMAGE_TOKEN_ESTIMATE,
        "resize_images": RESIZE_IMAGES and HAS_PILLOW and IMAGE_RESIZE_WORKERS > 0,
        "pillow_available": HAS_PILLOW,
        "image_max_dim": IMAGE_MAX_DIM,
        "image_resize_min_bytes": IMAGE_RESIZE_MIN_BYTES,
        "image_max_input_bytes": IMAGE_MAX_INPUT_BYTES,
        "image_jpeg_quality": IMAGE_JPEG_QUALITY,
        "image_resize_workers": IMAGE_RESIZE_WORKERS,
        "log_file": LOG_FILE or "stdout",
        "log_max_bytes": LOG_MAX_BYTES if LOG_FILE else None,
        "log_backup_count": LOG_BACKUP_COUNT if LOG_FILE else None,
    }


@app.get("/v1/models")
async def proxy_models(request: Request):
    try:
        resp = await client.get("/v1/models", headers=forward_headers(request.headers), timeout=10.0)
        if resp.status_code == 200:
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))
        logger.info("Backend /v1/models returned %d; using static model list", resp.status_code)
    except Exception as exc:
        logger.info("Backend /v1/models unavailable (%r); using static model list", exc)
    return Response(content=json.dumps(static_models_response(), ensure_ascii=False), status_code=200, media_type="application/json")


@app.post("/v1/messages/count_tokens")
async def proxy_count_tokens(request: Request):
    body, error = await parse_json_body(request)
    if error:
        return error
    try:
        resp = await client.post("/v1/messages/count_tokens", json=body, headers=forward_headers(request.headers), timeout=30.0)
        if resp.status_code == 200:
            return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))
        logger.info("Backend count_tokens returned %d; using estimate", resp.status_code)
        backend_status = resp.status_code
    except Exception as exc:
        logger.info("Backend count_tokens unavailable (%r); using estimate", exc)
        backend_status = type(exc).__name__
    return json_response({"input_tokens": estimate_tokens(body), "estimated": True, "source": "estimate_chars4", "backend_status": backend_status}, 200)


@app.post("/v1/messages")
async def proxy_messages(request: Request):
    body, error = await parse_json_body(request)
    if error:
        return error
    if "messages" not in body or not isinstance(body["messages"], list):
        return json_response({"error": "missing or invalid 'messages' field"}, 400)
    body = await resize_all_images(body)
    est, token_source = await request_token_count(body)
    num_msgs = len(body.get("messages", []))
    is_streaming = body.get("stream", False)
    should_warn = False
    pure_text_est = 0
    logger.info("Request: ~%d tokens (%s), %d messages, stream=%s", est, token_source, num_msgs, is_streaming)
    if est > COMPACTION_TRIGGER:
        logger.info("Trigger hit (%d > %d) - compacting", est, COMPACTION_TRIGGER)
        body, should_warn = compact_request(body, est, token_source)
        pure_text_est = estimate_pure_text_tokens(body)
    else:
        pure_text_est = estimate_pure_text_tokens(body)
        if pure_text_est >= CD_POINT:
            should_warn = True
            logger.info("CD POINT (no compaction): pure text ~%d tokens", pure_text_est)
    if is_streaming:
        req = client.build_request("POST", "/v1/messages", json=body, headers=forward_headers(request.headers))
        try:
            resp = await client.send(req, stream=True)
        except Exception as exc:
            logger.warning("Backend streaming request failed: %r", exc)
            return backend_error_response(exc)

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
                            elif event_type == "content_block_start" or event_type != "content_block_stop":
                                yield pending_frame
                                pending_text_stop = None
                        if event_type == "content_block_start" and isinstance(index, int) and isinstance(payload.get("content_block"), dict) and payload["content_block"].get("type") == "text":
                            active_text_indexes.add(index)
                        if not injected and event_type == "content_block_stop" and isinstance(index, int) and index in active_text_indexes:
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
                yield pending_frame
        stream_headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
        if should_warn:
            stream_headers["X-Context-Proxy-Warning"] = "cd_point"
            stream_headers["X-Context-Proxy-Pure-Text-Tokens"] = str(pure_text_est)
        return StreamingResponse(stream_with_warning(), status_code=resp.status_code, headers=stream_headers, background=BackgroundTask(resp.aclose))
    try:
        resp = await client.post("/v1/messages", json=body, headers=forward_headers(request.headers))
    except Exception as exc:
        logger.warning("Backend non-streaming request failed: %r", exc)
        return backend_error_response(exc)
    if should_warn and resp.status_code == 200:
        try:
            return json_response(inject_warning_non_streaming(resp.json(), pure_text_est), resp.status_code)
        except Exception:
            pass
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def catch_all(request: Request, path: str):
    url = f"/{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    request_kwargs = {"method": request.method, "url": url, "headers": forward_headers(request.headers)}
    if request.method.upper() not in ("GET", "DELETE", "OPTIONS"):
        request_kwargs["content"] = await request.body()
    try:
        resp = await client.request(**request_kwargs)
    except Exception as exc:
        logger.warning("Backend catch-all request failed: %r", exc)
        return backend_error_response(exc)
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/json"))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
