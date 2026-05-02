# Context Proxy v4 — 开发路线图

基于 GPT 评审反馈和 OpenClaw 兼容需求整理。

## 已完成的修复（v4.1/v4.2）

| 问题 | 修复 |
|------|------|
| read timeout 300s < 冷启动 340s | 拉到 600s |
| tools schema 未计入 token 估算 | 新增 `_tools_chars()` |
| RECENT_SKIP 固定不退让 | 压缩后仍超限时逐步缩小保护范围 |
| 流式警告在 message_stop 之后注入 | 改为在 message_stop 之前注入 |
| 配置需要手改 Python | 支持 `PROXY_*` 环境变量 |
| 缺少健康/探测端点 | 新增 `/health`、`/v1/models`、`/v1/messages/count_tokens` |
| catch-all 丢 query string | 已保留 query parameters |
| CD 点压缩路径口径不一致 | 压缩后改用纯文本 token 估算 |
| RECENT_SKIP 切断 tool_use/tool_result 对 | 压缩后删除 orphan `tool_result` |

## P0：近期必做

### 1. 用 llama-server count_tokens 替代 chars/4

llama-server 新版支持 `POST /v1/messages/count_tokens`，返回精确 token 数。
代理可以先调一次 count_tokens，得到精确数后再决定是否压缩。

```python
# 伪代码
resp = await client.post("/v1/messages/count_tokens", json=body)
actual_tokens = resp.json()["count"]
if actual_tokens > COMPACTION_TRIGGER_ACTUAL:
    body = compact_request(body)
```

回退策略：如果 count_tokens 不可用（旧版 llama-server），回退到 chars/4。

状态：已实现懒触发精确计数；先用 chars/4 估算，只有接近压缩阈值时才调用后端 count_tokens，后端不可用时回退估算。

### 2. 配置可通过环境变量覆盖

```python
BACKEND_URL = os.environ.get("PROXY_BACKEND_URL", "http://localhost:8000")
COMPACTION_TRIGGER = int(os.environ.get("PROXY_TRIGGER", "75000"))
CD_POINT = int(os.environ.get("PROXY_CD_POINT", "30000"))
RECENT_SKIP = int(os.environ.get("PROXY_RECENT_SKIP", "6"))
```

状态：v4.2 已实现。

### 3. 补齐兼容端点

```

状态：v4.2 已实现。
GET  /v1/models              → 返回模型列表（从 llama-server 代理或静态返回）
POST /v1/messages/count_tokens → 透传到 llama-server
GET  /health                 → 返回 {"status": "ok", "backend": "connected"}
```

### 4. catch-all 保留 query string

当前 catch-all 丢弃了 URL 的 query parameters。修复：
```python
url = f"/{path}?{request.url.query}" if request.url.query else f"/{path}"
```

状态：v4.2 已实现。

## P1：OpenClaw 兼容

### 5. 作为 OpenClaw Custom Provider 接入

**不要覆盖内置 anthropic provider**，用独立 custom provider：

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "local_gemma4": {
        "baseUrl": "http://127.0.0.1:8001",
        "api": "anthropic-messages",
        "apiKey": "local",
        "models": [{
          "id": "gemma4-26b-iq4xs-128k",
          "name": "Gemma 4 26B (Local, Context Proxy v4)",
          "contextWindow": 120000,
          "maxTokens": 8192
        }]
      }
    }
  }
}
```

注意 contextWindow 设 120000 而不是 131072 — 留余量给代理操作。

### 6. OpenClaw 特有的大 token 来源处理

OpenClaw 会在请求中携带：
- 大型 tools schema（几十个工具定义）
- workspace/bootstrap prompt
- 多工具并行调用的长 JSON 参数

代理需要：
- tools schema 计入估算 ✓（已做）
- 对超大单条 tool_result 先截断再决定是否整体压缩（已实现首尾保留截断）
- 对重复的文件读取结果做去重（同一路径多次出现只保留最新）

### 7. 提供 openclaw.json 配置样例文件

放在 deploy 文件夹中，用户 copy 即用。

状态：v4.2 已新增 `OPENCLAW.md` 和 `openclaw.models.json`。

## P2：后续扩展

### 8. OpenAI Chat Completions 适配

支持 `/v1/chat/completions` 路径，转换 OpenAI 格式到 Anthropic Messages 格式后转发。
让 v4 可以同时服务 Claude Code (Anthropic) 和 OpenClaw (OpenAI-compatible) 场景。

### 9. 分级压缩策略

对 tool_result 不是简单全删，而是分级：
- 最近的保留完整
- 中间的截断到 first/last 200 chars + `[compacted: N chars]`
- 最老的完全删除

### 10. 压缩事件指标

记录到 JSON 日志：
```json
{
  "timestamp": "...",
  "event": "compaction",
  "original_tokens": 95000,
  "after_tokens": 8000,
  "stripped": {"tool_use": 45, "tool_result": 45, "thinking": 45},
  "cd_point_reached": false
}
```

## 不做的事情

- ❌ 多 provider 路由（用 LiteLLM）
- ❌ Dashboard / Web UI
- ❌ 用户账号体系 / API Key 管理
- ❌ LLM 摘要管线（已验证 Gemma 4 摘要准确率 0%）
- ❌ 成本追踪 / billing
- ❌ 宣传"零智力损失"或"永不 400"

## 定位

**"Local llama-server 的 context fuse / circuit breaker"**

不是完整 memory system，不是通用 gateway。
是一个极简、可审计、无外部依赖的安全网代理。
