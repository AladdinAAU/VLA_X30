"""Frame sources for camera_ingest / thermal_ingest (no ROS): RTSP via OpenCV (FFmpeg or GStreamer), files, V4L2 devices
and a synthetic test pattern. A reader thread keeps only the NEWEST frame (late frames are dropped, never queued) and
reconnects with exponential backoff."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR uint8 (HxWx3) as delivered by OpenCV
    arrival_wall: float  # time.time() when the frame left the decoder
    seq: int


def gstreamer_pipeline(url: str, transport: str = "tcp") -> str:
    proto = "tcp" if transport == "tcp" else "udp"
    return (f"rtspsrc location={url} protocols={proto} latency=0 drop-on-latency=true ! decodebin ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false")


def ffmpeg_capture_options(transport: str, stimeout_us: int = 5_000_000) -> str:
    """OPENCV_FFMPEG_CAPTURE_OPTIONS value: tcp interleaved, low delay, socket timeout so a dead stream cannot block read()."""
    return f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|stimeout;{stimeout_us}|max_delay;500000"


class SyntheticSource:
    """`synthetic://WxH@FPS`: moving gradient with a frame counter; lets the pipeline run without any camera."""

    def __init__(self, spec: str):
        body = spec.split("://", 1)[1] or "640x480@10"
        size, _, fps = body.partition("@")
        w, _, h = size.partition("x")
        self.w, self.h, self.fps = int(w or 640), int(h or 480), float(fps or 10)
        self.n = 0
        self._t_next = time.time()

    def isOpened(self) -> bool:
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self._t_next += 1.0 / self.fps
        time.sleep(max(0.0, self._t_next - time.time()))
        x = (np.arange(self.w, dtype=np.uint16)[None, :] + self.n * 4) % 256
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[..., 0] = x.astype(np.uint8)
        img[..., 1] = (np.arange(self.h, dtype=np.uint16)[:, None] % 256).astype(np.uint8)
        img[..., 2] = (self.n * 7) % 256
        self.n += 1
        return True, img

    def release(self) -> None:
        pass


def open_capture(url: str, backend: str = "ffmpeg", transport: str = "tcp"):
    """Return an object with isOpened/read/release. Raises RuntimeError when OpenCV is missing."""
    if url.startswith("synthetic://"):
        return SyntheticSource(url)
    import cv2  # imported lazily so the module can be unit-tested without OpenCV

    if url.isdigit():
        return cv2.VideoCapture(int(url))
    if backend == "gstreamer":
        return cv2.VideoCapture(gstreamer_pipeline(url, transport), cv2.CAP_GSTREAMER)
    if url.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = ffmpeg_capture_options(transport)
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG)


class FrameReader:
    """Background reader. `latest()` returns the newest frame not yet returned, or None."""

    def __init__(self, url: str, backend: str = "ffmpeg", transport: str = "tcp", reconnect_initial_s: float = 1.0,
                 reconnect_max_s: float = 10.0, opener: Callable = open_capture, log: Callable[[str], None] = print):
        self.url, self.backend, self.transport = url, backend, transport
        self.r0, self.rmax = reconnect_initial_s, reconnect_max_s
        self.opener, self.log = opener, log
        self._lock = threading.Lock()
        self._frame: Optional[Frame] = None
        self._returned_seq = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_read = 0
        self.frames_dropped_late = 0
        self.reconnects = 0
        self.connected = False

    def start(self) -> "FrameReader":
        self._thread = threading.Thread(target=self._run, name="frame_reader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def latest(self) -> Optional[Frame]:
        with self._lock:
            f = self._frame
            if f is None or f.seq == self._returned_seq:
                return None
            self.frames_dropped_late += max(0, f.seq - self._returned_seq - 1)
            self._returned_seq = f.seq
            return f

    def _run(self) -> None:
        backoff = self.r0
        while not self._stop.is_set():
            try:
                cap = self.opener(self.url, self.backend, self.transport)
            except Exception as e:  # noqa: BLE001 - any opener failure is a reconnect case
                self.log(f"open failed: {e}")
                cap = None
            if cap is None or not cap.isOpened():
                self.connected = False
                self.log(f"cannot open {self.url}; retry in {backoff:.1f} s")
                self._stop.wait(backoff)
                backoff = min(backoff * 2.0, self.rmax)
                self.reconnects += 1
                continue
            self.connected = True
            self.log(f"connected to {self.url}")
            fails = 0
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok or img is None:
                    fails += 1
                    if fails >= 5:
                        break
                    time.sleep(0.05)
                    continue
                fails = 0
                backoff = self.r0
                self.frames_read += 1
                with self._lock:
                    self._frame = Frame(img, time.time(), self.frames_read)
            cap.release()
            self.connected = False
            self.reconnects += 1
            self.log(f"stream lost; reconnecting in {backoff:.1f} s")
            self._stop.wait(backoff)
            backoff = min(backoff * 2.0, self.rmax)
