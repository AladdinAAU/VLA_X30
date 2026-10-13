# Private Qwen3-VL access with Tailscale

The standalone model uses `docker/compose.models.lan.yml`, separate from the
main framework Compose definitions. Docker publishes its API only on localhost;
Tailscale Serve provides the HTTPS endpoint within your tailnet.

## Connect the devices

Install Tailscale on the server and client machine, then sign both in to the
same personal account. Keep only your own devices in the tailnet.

## Start the model on localhost

From the `x30_vln` directory in PowerShell on the server:

```powershell
$env:VLM_BIND_IP = '127.0.0.1'
$env:VLM_API_KEY = '<your-api-key>'
docker compose -f docker/compose.yml -f docker/compose.models.lan.yml --profile lan-vlm up -d --force-recreate vlm-lan-model
```

The default is `Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M`. First start downloads
the model into `D:/vlm-models` unless `VLM_MODELS_DIR` is set.

## Enable private HTTPS access

In Administrator PowerShell on the server, run:

```powershell
tailscale serve --bg 8000
```

If Tailscale asks to enable HTTPS for the tailnet, open the URL it prints and
approve that setting. Then run `tailscale serve status`; it reports the private
`https://<server>.<tailnet>.ts.net` URL.

On the client, configure your OpenAI-compatible SDK with that URL plus `/v1`,
model `qwen3-vl`, and the API key from `VLM_API_KEY`. For example:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<server>.<tailnet>.ts.net/v1",
    api_key="<your-api-key>",
)
response = client.chat.completions.create(
    model="qwen3-vl",
    messages=[{"role": "user", "content": "Describe this image."}],
)
print(response.choices[0].message.content)
```

Serve is tailnet-only. Do not enable Funnel for this service. To stop sharing,
run `tailscale serve off` in Administrator PowerShell. Stop the model with:

```powershell
docker compose -f docker/compose.yml -f docker/compose.models.lan.yml --profile lan-vlm stop vlm-lan-model
```
