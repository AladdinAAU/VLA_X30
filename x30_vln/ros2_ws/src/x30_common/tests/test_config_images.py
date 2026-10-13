import io

import numpy as np
import pytest
from PIL import Image

from x30_common.config import load_models_config, stable_hash
from x30_common.images import array_from_image_msg, count_images, encode_image, estimate_image_tokens, image_part, strip_images, user_message


def write_yaml(tmp_path, text):
    p = tmp_path / "models.yaml"
    p.write_text(text)
    return p


def test_defaults_without_file(tmp_path):
    cfg = load_models_config(tmp_path / "missing.yaml", env={})
    assert cfg.reasoner_vlm.base_url.endswith("/v1") and cfg.perception.url.startswith("http")


def test_yaml_and_env_override(tmp_path):
    p = write_yaml(tmp_path, "reasoner_vlm: {base_url: 'http://a:1/v1', model: m1, max_tokens: 300}\nperception: {url: 'http://p:2'}\n")
    cfg = load_models_config(p, env={"X30_REASONER_URL": "http://b:3/v1", "X30_REASONER_MAX_TOKENS": "256", "X30_PERCEPTION_URL": "http://q:4"})
    assert cfg.reasoner_vlm.base_url == "http://b:3/v1"
    assert cfg.reasoner_vlm.model == "m1" and cfg.reasoner_vlm.max_tokens == 256
    assert cfg.perception.url == "http://q:4"
    assert cfg.mapper_vlm.model == "qwen3-vl"


def test_api_key_env_and_redaction(tmp_path):
    p = write_yaml(tmp_path, "mapper_vlm: {api_key_env: MY_KEY}\n")
    cfg = load_models_config(p, env={"MY_KEY": "sk-secret"})
    assert cfg.mapper_vlm.api_key == "sk-secret"
    assert "sk-secret" not in str(cfg.as_dict())


def test_extra_body_json_env(tmp_path):
    cfg = load_models_config(tmp_path / "x.yaml", env={"X30_REASONER_EXTRA_BODY": '{"cache_prompt": true}'})
    assert cfg.reasoner_vlm.extra_body == {"cache_prompt": True}


def test_hash_stable():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


class Msg:
    def __init__(self, arr, enc):
        self.height, self.width = arr.shape[:2]
        self.encoding = enc
        self.step = arr.shape[1] * (arr.shape[2] if arr.ndim == 3 else 1)
        self.data = arr.tobytes()


def test_array_from_image_msg_bgr_and_mono():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 0] = 200
    bgr = rgb[:, :, ::-1].copy()
    out = array_from_image_msg(Msg(bgr, "bgr8"))
    assert (out == rgb).all()
    assert array_from_image_msg(Msg(rgb, "rgb8")).shape == (4, 6, 3)
    assert array_from_image_msg(Msg(np.ones((3, 3), np.uint8), "mono8")).shape == (3, 3)
    with pytest.raises(ValueError):
        array_from_image_msg(Msg(rgb, "yuv422"))


def test_encode_and_message():
    img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    data = encode_image(img, max_side=320)
    assert data[:2] == b"\xff\xd8"
    assert max(Image.open(io.BytesIO(data)).size) == 320
    part = image_part(img, 128)
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    msg = user_message("hello", [img, img], max_side=64)
    assert count_images([msg]) == 2 and msg["content"][-1]["text"] == "hello"
    assert "chars" in strip_images([msg])[0]["content"][0]["image_url"]["url"]


def test_token_estimate():
    assert estimate_image_tokens(640, 480) == 20 * 15 + 4
    assert estimate_image_tokens(64, 64) == 4 + 4
