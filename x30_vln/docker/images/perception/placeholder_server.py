"""Placeholder for the perception server so the container can be tested before the real code exists.

Answers GET /health with the GPU state; POST /segment returns 501. Replaced automatically as soon as
ros2_ws/src/x30_perception/perception_server/server.py exists (see entrypoint.sh).
"""
from fastapi import FastAPI, HTTPException

app = FastAPI(title="x30 perception placeholder")


@app.get("/health")
def health() -> dict:
    info = {"status": "placeholder", "cuda": False}
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # torch missing or driver problem: report, do not crash
        info["error"] = str(exc)
    return info


@app.post("/segment")
def segment() -> None:
    raise HTTPException(status_code=501, detail="placeholder server: real perception_server not mounted")
