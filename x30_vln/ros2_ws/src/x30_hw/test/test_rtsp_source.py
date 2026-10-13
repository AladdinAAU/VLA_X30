"""FrameReader: newest-frame-only delivery, reconnect with backoff, synthetic source, intrinsics scaling."""

import time

import numpy as np

from x30_hw.rtsp_source import FrameReader, SyntheticSource, ffmpeg_capture_options, gstreamer_pipeline


class FlakyCap:
    """Delivers `n` frames then fails; the opener fails twice before the first success."""

    def __init__(self, n):
        self.n, self.i = n, 0

    def isOpened(self):
        return True

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        time.sleep(0.002)
        return True, np.full((4, 6, 3), self.i, np.uint8)

    def release(self):
        pass


def test_reconnect_and_newest_frame_only():
    calls = {"n": 0}

    def opener(url, backend, transport):
        calls["n"] += 1
        if calls["n"] <= 2:
            return None
        return FlakyCap(30)

    r = FrameReader("rtsp://x", opener=opener, reconnect_initial_s=0.01, reconnect_max_s=0.05, log=lambda m: None).start()
    time.sleep(0.5)  # consumer is slow: only the newest frame may be returned
    f = r.latest()
    assert f is not None and f.seq > 5  # the newest frame read so far, not the first one
    time.sleep(0.5)
    r.stop()
    assert r.reconnects >= 3 and r.frames_dropped_late > 0 and calls["n"] >= 4


def test_synthetic_source_runs():
    s = SyntheticSource("synthetic://64x48@100")
    ok, img = s.read()
    assert ok and img.shape == (48, 64, 3)


def test_option_strings():
    assert "rtsp_transport;tcp" in ffmpeg_capture_options("tcp")
    assert "protocols=tcp" in gstreamer_pipeline("rtsp://a/b")
    assert "appsink drop=true max-buffers=1" in gstreamer_pipeline("rtsp://a/b")
