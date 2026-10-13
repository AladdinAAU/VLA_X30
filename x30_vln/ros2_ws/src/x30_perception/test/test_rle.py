import numpy as np
import pytest

from perception_server import rle


@pytest.mark.parametrize('shape', [(1, 1), (7, 5), (480, 640)])
def test_roundtrip_random(shape):
    rng = np.random.default_rng(1)
    for density in (0.0, 0.02, 0.5, 1.0):
        m = rng.random(shape) < density
        r = rle.encode(m)
        assert r['size'] == list(shape)
        assert (rle.decode(r) == m).all()
        assert rle.area(r) == int(m.sum())


def test_blob_is_compact():
    m = np.zeros((480, 640), bool)
    m[100:200, 300:420] = True
    r = rle.encode(m)
    assert len(r['counts']) < 600
    assert (rle.decode(r) == m).all()


def test_matches_pycocotools_if_available():
    mask_util = pytest.importorskip('pycocotools.mask')
    rng = np.random.default_rng(2)
    for _ in range(5):
        m = rng.random((60, 80)) < 0.3
        ref = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        mine = rle.encode(m)
        assert mine['counts'] == ref['counts'].decode()
        assert (mask_util.decode({'size': mine['size'], 'counts': mine['counts'].encode()}).astype(bool) == m).all()
