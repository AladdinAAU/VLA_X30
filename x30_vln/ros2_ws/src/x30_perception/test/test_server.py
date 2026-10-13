"""Server request handling with the colour-blob fake backend (no GPU, no weights)."""

import base64

import numpy as np
import pytest

import cv2

from perception_server import rle
from perception_server.backends import FakeBackend, Instance
from perception_server.pipeline import Segmenter, merge_instances

try:
    from fastapi.testclient import TestClient

    from perception_server.app import create_app

    HAVE_WEB = True
except ImportError:  # the ROS framework image has no fastapi; these tests run in the perception image
    HAVE_WEB = False

web = pytest.mark.skipif(not HAVE_WEB, reason='fastapi/httpx not installed')


def _jpeg_b64(img):
    return base64.b64encode(cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()).decode()


def _image():
    img = np.full((480, 640, 3), 128, np.uint8)
    cv2.rectangle(img, (100, 100), (220, 260), (0, 0, 255), -1)
    cv2.rectangle(img, (400, 300), (480, 400), (255, 0, 0), -1)
    return img


@pytest.fixture
def client():
    return TestClient(create_app(Segmenter(FakeBackend(), default_vocab=['thing'])))


@web
def test_health(client):
    j = client.get('/health').json()
    assert j['ok'] and j['backend'] == 'fake'


@web
def test_segment_returns_rle_masks_and_timing(client):
    r = client.post('/segment', json={'image': _jpeg_b64(_image()), 'vocab': ['chair', 'table'], 'conf': 0.2})
    assert r.status_code == 200
    j = r.json()
    assert len(j['instances']) == 2
    assert j['image'] == {'width': 640, 'height': 480}
    assert {'decode_ms', 'wait_ms', 'vocab_ms', 'infer_ms', 'rle_ms'} <= set(j['timing'])
    inst = max(j['instances'], key=lambda i: rle.area(i['mask_rle']))
    assert inst['label'] == 'chair' and inst['mask_source'] == 'model'
    mask = rle.decode(inst['mask_rle'])
    assert mask.shape == (480, 640)
    ys, xs = np.nonzero(mask)
    assert abs(xs.min() - 100) <= 3 and abs(ys.max() - 260) <= 3
    assert np.allclose(inst['box'], [100, 100, 221, 261], atol=3)


@web
def test_default_vocab_and_errors(client):
    ok = client.post('/segment', json={'image': _jpeg_b64(_image())}).json()
    assert ok['instances'][0]['label'] == 'thing'
    assert client.post('/segment', json={'image': 'not-base64-jpeg', 'vocab': ['a']}).status_code == 400
    empty = TestClient(create_app(Segmenter(FakeBackend())))
    assert empty.post('/segment', json={'image': _jpeg_b64(_image())}).status_code == 400


@web
def test_no_masks_option(client):
    j = client.post('/segment', json={'image': _jpeg_b64(_image()), 'vocab': ['x'], 'masks': False}).json()
    assert 'mask_rle' not in j['instances'][0]


@web
def test_regions_pass_maps_back_to_full_frame():
    class Tiny(FakeBackend):
        """Finds nothing in the full frame but the object when the frame is a crop (a 'small far object')."""

        def predict(self, bgr, phrases, conf, imgsz, iou, max_det):
            return super().predict(bgr, phrases, conf, imgsz, iou, max_det) if bgr.shape[1] < 640 else []

    c = TestClient(create_app(Segmenter(Tiny())))
    j = c.post('/segment', json={'image': _jpeg_b64(_image()), 'vocab': ['x'], 'regions': [[0.1, 0.1, 0.5, 0.6]]}).json()
    assert len(j['instances']) == 1
    m = rle.decode(j['instances'][0]['mask_rle'])
    assert m.shape == (480, 640) and m[180, 160] and not m[350, 440]


@web
def test_embed_not_available_is_501(client):
    assert client.post('/embed', json={'texts': ['a']}).status_code == 501
    assert client.post('/embed', json={'image': 'x'}).status_code == 501


def test_merge_keeps_higher_score_duplicate():
    a = Instance('chair', 0.5, [0, 0, 10, 10])
    b = Instance('chair', 0.8, [1, 1, 10, 10])
    c = Instance('table', 0.4, [1, 1, 10, 10])
    out = merge_instances([a], [b, c], 0.5)
    assert [i.score for i in out] == [0.8, 0.4]
