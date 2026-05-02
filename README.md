# ContextFuse

llama-server 的上下文熔断代理。防止 128K 上下文溢出，并尽量减少长会话中的智力损失。

```
客户端 ──▶ ContextFuse (:8001) ──▶ llama-server (:8000)
               │
               ├─ 小请求 → 直接转发
               └─ 大请求 → 五步级联压缩 → 安全转发
```

## 为什么需要它

llama-server 没有自动上下文管理。Claude Code / OpenClaw 对话到 128K tokens 时直接
400 错误，会话中断，工作丢失。ContextFuse 在请求到达后端之前自动压缩上下文，
对话尽量不再因为上下文溢出而中断。

## 快速部署

```bash
# 1. 创建虚拟环境
python -m venv proxy-venv
source proxy-venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动
python context_proxy.py

# 4. 将客户端指向代理
# Claude Code settings.json:
#   "ANTHROPIC_BASE_URL": "http://localhost:8001"
```

所有配置通过环境变量覆盖，无需改源码：

```bash
PROXY_BACKEND_URL=http://localhost:8000 \
PROXY_TRIGGER=75000 \
PROXY_LOG_FILE=./contextfuse.log \
python context_proxy.py
```

生产部署可以参考 [DEPLOYMENT.md](DEPLOYMENT.md) 和
[systemd/contextfuse.service.example](systemd/contextfuse.service.example)。

## 五步级联压缩

当预估 token 超过 `COMPACTION_TRIGGER`（默认 75K）时触发：

```
Step 1 — 剥离旧消息的 tool_use / tool_result / thinking 块
         保留所有 text 块完整，跳过最近 RECENT_SKIP 条消息
         ↓ 还超？
Step 2 — 截断保护区内的超大 tool_result（>20K chars → 前400+后400）
         ↓ 还超？
Step 3 — 从最早的消息开始丢弃完整的 user/assistant 对
         ↓ 还超？
Step 4 — 自适应缩小保护区（RECENT_SKIP 6→4→2→0），逐步剥离
         ↓ 还超？
Step 5 — 最后手段：再次丢弃最早的消息
```

每一步都有独立日志记录压缩前后的 token 数。绝大多数情况下 Step 1 就够了。

## CD 点提醒

当纯文本（剥离后剩余的对话内容）超过 `CD_POINT`（默认 30K tokens），
ContextFuse 会在模型响应末尾注入提醒：

> [Context Proxy 提醒] 当前会话已变长：纯文本约 30K tokens，
> 后端窗口约 131K tokens。建议完成当前小任务后新开会话...

流式响应通过 SSE 帧级解析，在最后一个 text block 关闭前注入 `text_delta`，
不会打进 thinking block。非流式响应追加到最后一个 text 块末尾。

## API 端点

| 端点 | 说明 |
|------|------|
| `GET /health` | 代理配置 + 后端连通状态 |
| `GET /v1/models` | 透传后端，失败时返回静态模型列表 |
| `POST /v1/messages/count_tokens` | 透传后端，失败时返回 chars/4 估算 |
| `POST /v1/messages` | 主路径：自动上下文保护 |
| `/{path}` | 其他路径透传到后端 |

## 默认配置说明

当前默认值基于以下硬件和模型调优：

- **GPU**: NVIDIA RTX 2000 Ada 16GB
- **模型**: Gemma 4 26B-A4B-it, IQ4_XS 量化 (~14GB)
- **上下文**: 128K tokens (`--ctx-size 131072`)
- **并发**: 单 slot（16GB VRAM 极限）
- **KV cache**: q8_0, flash-attn

如果你的硬件或模型不同，需要调整 `PROXY_TRIGGER`、`PROXY_CONTEXT_LIMIT` 等参数。
参考下方"如何调整 COMPACTION_TRIGGER"一节。

## 全量配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `PROXY_BACKEND_URL` | `http://localhost:8000` | 后端 llama-server 地址 |
| `PROXY_MODEL_ID` | `gemma4-26b-iq4xs-128k` | `/v1/models` 回退模型名 |
| `PROXY_CONTEXT_LIMIT` | `131072` | 用于 CD 提醒文案和 health 输出 |
| `PROXY_TRIGGER` | `75000` | 压缩触发阈值（chars/4 估算） |
| `PROXY_CD_POINT` | `30000` | 纯文本提醒阈值 |
| `PROXY_RESERVE_TOKENS` | `8192` | 输出预留 |
| `PROXY_RECENT_SKIP` | `6` | 默认保护最近消息数 |
| `PROXY_USE_COUNT_TOKENS` | `1` | 是否启用后端精确计数 |
| `PROXY_COUNT_TOKENS_PRECISE_THRESHOLD` | `0.7` | 估算超过 trigger×此值才调精确计数 |
| `PROXY_INJECT_STREAM_WARNINGS` | `1` | 是否在流式响应中注入 CD 提醒 |
| `PROXY_TOOL_RESULT_TRUNCATE_TRIGGER_CHARS` | `20000` | 大 tool_result 截断触发字符数 |
| `PROXY_TOOL_RESULT_TRUNCATE_KEEP_CHARS` | `400` | 截断时保留首尾各多少字符 |
| `PROXY_LOG_FILE` | _(空=仅stdout)_ | 日志文件路径 |
| `PROXY_LOG_MAX_BYTES` | `10485760` | 日志轮转大小（10MB） |
| `PROXY_LOG_BACKUP_COUNT` | `3` | 保留轮转备份数 |

### 如何调整 COMPACTION_TRIGGER

```
实际 tokens ≈ chars / 3 (Gemma 4 实测)
chars/4 估算 ≈ 实际 tokens × 0.75

要在实际 100K tokens 时触发:
  100K × 0.75 = 75K (chars/4 估算)  ← 默认值

要在实际 80K tokens 时触发:
  80K × 0.75 = 60K
```

## 错误处理

| 场景 | 响应 |
|------|------|
| 畸形 JSON body | 400 `{"error": "invalid JSON body"}` |
| JSON 非对象 | 400 `{"error": "JSON body must be an object"}` |
| 缺少 messages 字段 | 400 `{"error": "missing or invalid 'messages' field"}` |
| 后端连接失败 | 502 `{"error": "backend unreachable", "backend_url": "..."}` |
| 后端读超时（600s） | 504 `{"error": "backend timeout", ...}` |
| 后端写超时 | 504 `{"error": "backend write timeout", ...}` |

## 客户端接入

### Claude Code

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8001",
    "ANTHROPIC_AUTH_TOKEN": "local-dummy-key",
    "ANTHROPIC_MODEL": "gemma4-26b-iq4xs-128k"
  }
}
```

### OpenClaw (Custom Provider)

详见 [OPENCLAW.md](OPENCLAW.md) 和 [openclaw.models.json](openclaw.models.json)。

### 通过 SSH 隧道远程访问

```bash
ssh -L 18001:localhost:8001 user@your-server -N
# 然后客户端指向 http://localhost:18001
```

## 日志示例

```
Request: ~5 tokens (estimate_chars4), 1 messages, stream=True
```

```
Request: ~87581 tokens (backend_count_tokens), 241 messages, stream=True
Trigger hit (87581 > 75000) — compacting
Compaction baseline: trigger_count=87581 (backend_count_tokens), local_estimate_before=65000 (estimate_chars4)
Compaction: stripped 50 tool_use, 50 tool_result, 30 thinking blocks; saved ~336266 chars (~84066 tokens)
After stripping (estimate_chars4): 65000 -> 5202 tokens
Compaction: truncated 2 large retained tool_result blocks; saved ~45000 chars (~11250 tokens)
CD POINT reached: ~32000 tokens after compaction (threshold=30000)
```

## 已验证性能

| 指标 | 结果 |
|------|------|
| 事实回忆 | 内部长上下文测试中保持 7/7 命中 |
| 稳定性 | 内部长稳测试 80 轮无代理崩溃 |
| 最大压缩比 | 138K → 10K tokens |
| 延迟开销 | < 1ms (纯字符串操作，不调模型) |
| 并发安全 | 无状态，多客户端互不干扰 |

## 项目文件

| 文件 | 说明 |
|------|------|
| `.gitignore` | 防止提交日志、虚拟环境、缓存、本地配置 |
| `context_proxy.py` | 代理主程序 |
| `requirements.txt` | Python 依赖 |
| `DEPLOYMENT.md` | 本地、远程、systemd 部署说明 |
| `systemd/contextfuse.service.example` | systemd 服务示例 |
| `DEV_ROADMAP.md` | 开发路线图 |
| `OPENCLAW.md` | OpenClaw 接入指南 |
| `openclaw.models.json` | OpenClaw provider 示例 |

## 不适用场景

- 后端已有自动上下文管理（如 OpenAI API 的自动截断）
- 短对话不需要压缩

## 依赖

- Python 3.10+
- fastapi
- uvicorn
- httpx
