# OpenClaw + ContextFuse

Use a dedicated OpenClaw custom provider for the local llama-server proxy. Do
not override OpenClaw's built-in `anthropic` provider; a separate provider is
easier to debug and safer to roll back.

## Assumptions

- Context Proxy runs at `http://localhost:8001`.
- llama-server runs at `http://localhost:8000`.
- Model alias is `gemma4-26b-iq4xs-128k`.
- OpenClaw and the proxy run on the same server, or the proxy port is forwarded
  with SSH.

Health checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

## Provider Config

Merge `openclaw.models.json` into OpenClaw's model config:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "local_gemma4_proxy": {
        "baseUrl": "http://localhost:8001",
        "api": "anthropic-messages",
        "apiKey": "local-dummy-key",
        "request": {
          "allowPrivateNetwork": true
        },
        "models": [
          {
            "id": "gemma4-26b-iq4xs-128k",
            "name": "Gemma 4 26B (local via ContextFuse)",
            "contextWindow": 120000,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": "local_gemma4_proxy/gemma4-26b-iq4xs-128k"
    }
  }
}
```

If OpenClaw runs on another machine:

```bash
ssh -L 18001:localhost:8001 user@your-server -N
```

Then set:

```json
"baseUrl": "http://localhost:18001"
```

## Validation

```bash
openclaw --profile contextproxy config validate
openclaw --profile contextproxy models list --provider local_gemma4_proxy --plain
```

Direct provider smoke:

```bash
openclaw --profile contextproxy infer model run \
  --local \
  --model local_gemma4_proxy/gemma4-26b-iq4xs-128k \
  --prompt "Reply with exactly: openclaw-proxy-ok" \
  --json
```

Full agent smoke depends on your OpenClaw workspace. Use an isolated temporary
workspace when possible, so exact-output tests are not affected by existing
project memory or bootstrap files.

## Fixes From Server Testing

- First-run plugin dependency install may resolve a GitHub dependency through
  SSH (`git@github.com` / `ssh://git@github.com`). Servers without GitHub SSH
  auth fail there. The stability script exports a process-local git rewrite to
  HTTPS, so user git config is not changed.
- First agent runs can be slow while OpenClaw installs plugin runtime deps and
  builds the agent system prompt. Use a generous timeout for first-run tests.
- OpenClaw's default bootstrap workspace can pollute exact-output tests. The
  stability script initializes an isolated workspace and removes `BOOTSTRAP.md`
  after writing minimal `IDENTITY.md`, `USER.md`, and `SOUL.md`.

## Expected Result

When configured correctly:

- `infer model run --local` reaches Context Proxy.
- `agent --local` reaches Context Proxy.
- Winner provider is `local_gemma4_proxy`.
- Winner model is `gemma4-26b-iq4xs-128k`.
- Fallback is not used.
