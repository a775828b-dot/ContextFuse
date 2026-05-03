import asyncio
import base64
import io
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor

try:
    import httpx  # noqa: F401
    import fastapi  # noqa: F401
except ImportError:
    httpx_stub = types.ModuleType("httpx")

    class _DummyAsyncClient:
        pass

    class _DummyTimeout:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyHTTPError(Exception):
        pass

    httpx_stub.AsyncClient = _DummyAsyncClient
    httpx_stub.Timeout = _DummyTimeout
    httpx_stub.ConnectError = type("ConnectError", (_DummyHTTPError,), {})
    httpx_stub.ReadTimeout = type("ReadTimeout", (_DummyHTTPError,), {})
    httpx_stub.WriteTimeout = type("WriteTimeout", (_DummyHTTPError,), {})
    httpx_stub.HTTPError = _DummyHTTPError
    sys.modules.setdefault("httpx", httpx_stub)

    uvicorn_stub = types.ModuleType("uvicorn")
    uvicorn_stub.run = lambda *args, **kwargs: None
    sys.modules.setdefault("uvicorn", uvicorn_stub)

    fastapi_stub = types.ModuleType("fastapi")

    class _DummyFastAPI:
        def __init__(self, *args, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return lambda fn: fn

        def post(self, *args, **kwargs):
            return lambda fn: fn

        def api_route(self, *args, **kwargs):
            return lambda fn: fn

    fastapi_stub.FastAPI = _DummyFastAPI
    fastapi_stub.Request = type("Request", (), {})
    sys.modules.setdefault("fastapi", fastapi_stub)

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.Response = type("Response", (), {"__init__": lambda self, *args, **kwargs: None})
    responses_stub.StreamingResponse = type("StreamingResponse", (), {"__init__": lambda self, *args, **kwargs: None})
    sys.modules.setdefault("fastapi.responses", responses_stub)

    background_stub = types.ModuleType("starlette.background")
    background_stub.BackgroundTask = type("BackgroundTask", (), {"__init__": lambda self, *args, **kwargs: None})
    sys.modules.setdefault("starlette.background", background_stub)

import context_proxy as proxy


def anthropic_image(data="abcd", media_type="image/png"):
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def image_url(data="abcd", media_type="image/jpeg"):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{media_type};base64,{data}",
        },
    }


class MultimodalEstimationTests(unittest.TestCase):
    def test_top_level_images_count_as_fixed_tokens(self):
        body = {
            "messages": [
                {"role": "user", "content": [anthropic_image(), image_url(), {"type": "text", "text": "hello"}]},
            ]
        }

        expected = proxy.IMAGE_TOKEN_ESTIMATE * 2 + len("hello") // 4 + 4
        self.assertEqual(proxy.estimate_tokens(body), expected)

    def test_tool_result_nested_image_counts_as_fixed_tokens(self):
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "screenshot"},
                        anthropic_image(),
                    ],
                }
            ],
        }

        self.assertEqual(
            proxy.estimate_message_tokens(msg),
            proxy.IMAGE_TOKEN_ESTIMATE + len("screenshot") // 4 + 4,
        )

    def test_pure_text_estimate_ignores_images(self):
        body = {
            "messages": [
                {"role": "user", "content": [anthropic_image(), {"type": "text", "text": "visible"}]},
            ]
        }

        self.assertEqual(proxy.estimate_pure_text_tokens(body), len("visible") // 4 + 4)


class MultimodalCompactionTests(unittest.TestCase):
    def test_strip_non_text_replaces_old_image_with_placeholder(self):
        body = {
            "messages": [
                {"role": "user", "content": [anthropic_image(), {"type": "text", "text": "old"}]},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "recent 1"},
                {"role": "assistant", "content": "recent 2"},
                {"role": "user", "content": "recent 3"},
                {"role": "assistant", "content": "recent 4"},
                {"role": "user", "content": "recent 5"},
                {"role": "assistant", "content": "recent 6"},
            ]
        }

        compacted = proxy.strip_non_text(body)
        first_content = compacted["messages"][0]["content"]
        self.assertEqual(first_content[0]["type"], "text")
        self.assertIn("[image compacted:", first_content[0]["text"])
        self.assertEqual(first_content[1]["text"], "old")

    def test_strip_with_skip_replaces_images_when_protection_shrinks(self):
        body = {
            "messages": [
                {"role": "user", "content": [anthropic_image()]},
                {"role": "assistant", "content": "ok"},
            ]
        }

        compacted = proxy._strip_with_skip(body, 0)
        self.assertIn("[image compacted:", compacted["messages"][0]["content"][0]["text"])

    def test_truncate_large_tool_results_replaces_nested_image(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [anthropic_image(), {"type": "text", "text": "caption"}],
                        }
                    ],
                }
            ]
        }

        compacted = proxy.truncate_large_tool_results(body)
        nested = compacted["messages"][0]["content"][0]["content"]
        self.assertEqual(nested[0]["type"], "text")
        self.assertIn("[image compacted:", nested[0]["text"])
        self.assertEqual(nested[1]["text"], "caption")


@unittest.skipUnless(proxy.HAS_PILLOW, "Pillow is not installed")
class MultimodalResizeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_executor = proxy.image_executor
        self.old_min = proxy.IMAGE_RESIZE_MIN_BYTES
        proxy.image_executor = ThreadPoolExecutor(max_workers=1)
        proxy.IMAGE_RESIZE_MIN_BYTES = 0

    async def asyncTearDown(self):
        proxy.image_executor.shutdown(wait=False, cancel_futures=True)
        proxy.image_executor = self.old_executor
        proxy.IMAGE_RESIZE_MIN_BYTES = self.old_min

    async def test_large_image_resizes_to_max_dimension(self):
        from PIL import Image

        img = Image.new("RGB", (2048, 1536), "red")
        raw = io.BytesIO()
        img.save(raw, format="PNG")
        encoded = base64.b64encode(raw.getvalue()).decode("ascii")
        body = {"messages": [{"role": "user", "content": [anthropic_image(encoded)]}]}

        resized = await proxy.resize_all_images(body)
        source = resized["messages"][0]["content"][0]["source"]
        decoded = base64.b64decode(source["data"])
        with Image.open(io.BytesIO(decoded)) as out:
            self.assertLessEqual(max(out.size), proxy.IMAGE_MAX_DIM)
        self.assertEqual(source["media_type"], "image/jpeg")

    async def test_system_image_is_resized(self):
        from PIL import Image

        img = Image.new("RGB", (1536, 1536), "blue")
        raw = io.BytesIO()
        img.save(raw, format="PNG")
        encoded = base64.b64encode(raw.getvalue()).decode("ascii")
        body = {
            "system": [anthropic_image(encoded)],
            "messages": [{"role": "user", "content": "hello"}],
        }

        resized = await proxy.resize_all_images(body)
        source = resized["system"][0]["source"]
        decoded = base64.b64decode(source["data"])
        with Image.open(io.BytesIO(decoded)) as out:
            self.assertLessEqual(max(out.size), proxy.IMAGE_MAX_DIM)


if __name__ == "__main__":
    unittest.main()
