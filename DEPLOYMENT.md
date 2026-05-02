# Deployment

ContextFuse is a small FastAPI proxy. It expects an Anthropic Messages compatible
llama-server backend, usually on `http://localhost:8000`, and exposes the proxy
on `http://localhost:8001`.

## Prerequisites

- Python 3.10+
- A running llama-server backend
- A model served with an Anthropic/llama.cpp chat template compatible with
  `/v1/messages`

Example backend health check:

```bash
curl http://localhost:8000/health
```

## Local Run

```bash
python -m venv proxy-venv
source proxy-venv/bin/activate
pip install -r requirements.txt

PROXY_BACKEND_URL=http://localhost:8000 \
PROXY_MODEL_ID=gemma4-26b-iq4xs-128k \
PROXY_LOG_FILE=./contextfuse.log \
python context_proxy.py
```

Check the proxy:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/v1/models
```

## systemd

Copy the repository to a stable location, for example:

```bash
sudo mkdir -p /opt/contextfuse
sudo cp -a . /opt/contextfuse/
cd /opt/contextfuse
python -m venv proxy-venv
./proxy-venv/bin/pip install -r requirements.txt
```

Install the service example:

```bash
sudo cp systemd/contextfuse.service.example /etc/systemd/system/contextfuse.service
sudo systemctl daemon-reload
sudo systemctl enable --now contextfuse
sudo systemctl status contextfuse
```

Logs:

```bash
journalctl -u contextfuse -f
tail -f /var/log/contextfuse.log
```

## Client Values

Claude Code:

```text
ANTHROPIC_BASE_URL=http://YOUR_SERVER_IP:8001
ANTHROPIC_AUTH_TOKEN=local-dummy-key
ANTHROPIC_MODEL=gemma4-26b-iq4xs-128k
```

OpenClaw:

```text
baseUrl: http://YOUR_SERVER_IP:8001
api: anthropic-messages
apiKey: local-dummy-key
model: local_gemma4_proxy/gemma4-26b-iq4xs-128k
```

If the proxy is not exposed on the network, use an SSH tunnel:

```bash
ssh -L 18001:localhost:8001 user@your-server -N
```

Then point clients to `http://localhost:18001`.

## Security Notes

ContextFuse does not implement authentication by itself. Run it on localhost,
behind an SSH tunnel, or behind a trusted reverse proxy if network access is
needed. Do not expose it directly to the public internet.
