# x30_common

Shared, ROS-free helpers used by every package that talks to a language model (ament_python, only numpy/PyYAML/Pillow needed).

| Module | What it does |
|---|---|
| `vlm_client.py` | `VLMClient`: OpenAI-compatible `/chat/completions` over plain `urllib` (sync, async, cancellable from another thread). Retry with backoff on 5xx/timeouts/connection errors, no retry on 4xx and context overflow (`VLMContextOverflow` carries `n_prompt`/`n_ctx`), usage and latency counters, `health()`, `server_props()`, tool calls normalised across llama.cpp quirks (arguments as string/dict, calls written into the text, reasoning in `reasoning_content`/`<think>`). |
| `toolschema.py` | `make_tool`, argument coercion to a JSON schema (`"5"` -> 5, `"a,b"` -> list, enum case, unknown keys dropped) with a list of problems; `is_fatal` separates fixable from fatal problems. |
| `jsonrepair.py` | Lenient JSON for malformed model output (single quotes, trailing commas, truncated objects, fenced blocks), text-form tool calls. |
| `images.py` | JPEG/base64 encoding with downscale, `user_message(text, images)`, `array_from_image_msg` (sensor_msgs/Image without cv_bridge), image token estimate. |
| `config.py` | `load_models_config()` reads `configs/models.yaml` (roles `perception`, `mapper_vlm`, `reasoner_vlm`), env overrides `X30_<ROLE>_<FIELD>`, `find_config_dir()` (`X30_CONFIG_DIR`, `X30_CONFIGS`, or a parent folder named `configs`), stable config hash. |
| `testing/fake_openai.py` | `FakeOpenAIServer`: scripted or callable stand-in for the model server (used by all tests). |

Use: `from x30_common import VLMClient, load_models_config; c = VLMClient.from_config("reasoner", load_models_config())`.
On the Windows host use `http://127.0.0.1:8000/v1`, not `localhost` (about 2 s extra per request, measured).

Tests: `cd ros2_ws/src/x30_common && python -m pytest -q` (59 tests, a few seconds, no GPU, no ROS).
