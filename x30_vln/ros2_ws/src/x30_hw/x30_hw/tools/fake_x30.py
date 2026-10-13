"""fake_x30: UDP fake of the X30 motion host (pure python, no ROS).

    python -m x30_hw.tools.fake_x30                       # listens on 127.0.0.1:43893, telemetry to 127.0.0.1:43897
    python -m x30_hw.tools.fake_x30 --listen-ip 0.0.0.0 --allow-non-loopback --register-sender   # container tests
    python -m x30_hw.tools.fake_x30 --replay capture.x30t  # replay recorded telemetry instead of simulating

Safety: it refuses to bind or send to anything outside loopback unless --allow-non-loopback is given, and it refuses
192.168.0.0/16 unconditionally (that range is the real robot's network; tests must never touch it).

A JSON-lines control channel (UDP, default 127.0.0.1:43999) offers fault injection and introspection, see
FakeRobot.control(): {"cmd": "status"}, {"cmd": "mute_telemetry", "seconds": 3}, {"cmd": "hardware_estop"},
{"cmd": "battery", "percent": 8}, {"cmd": "handheld", "seconds": 5}, {"cmd": "set", "key": "lag_tau_s", "value": 0.5}.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import select
import socket
import struct
import sys
import threading
import time
from typing import List, Optional, Tuple

from x30_hw import protocol as P
from x30_hw.tools.fake_core import Addr, FakeConfig, FakeRobot

CAPTURE_MAGIC = b"X30T1\n"
CAPTURE_REC = struct.Struct("<dI")  # seconds since start, datagram length


def check_address(ip: str, allow_non_loopback: bool) -> None:
    """Hard guard (AGENT-BRIEF rule 5): never the robot's 192.168.0.0/16, loopback only by default."""
    addr = ipaddress.ip_address(ip)
    if addr in ipaddress.ip_network("192.168.0.0/16"):
        raise SystemExit(f"fake_x30: refusing {ip}: 192.168.0.0/16 is the real robot network")
    if not addr.is_loopback and not addr.is_unspecified and not allow_non_loopback:
        raise SystemExit(f"fake_x30: refusing non-loopback address {ip} without --allow-non-loopback")


def read_capture(path: str) -> List[Tuple[float, bytes]]:
    with open(path, "rb") as f:
        if f.read(len(CAPTURE_MAGIC)) != CAPTURE_MAGIC:
            raise ValueError(f"{path} is not an x30 telemetry capture")
        recs = []
        while True:
            hdr = f.read(CAPTURE_REC.size)
            if len(hdr) < CAPTURE_REC.size:
                break
            t, n = CAPTURE_REC.unpack(hdr)
            recs.append((t, f.read(n)))
    return recs


class FakeX30Server:
    def __init__(self, cfg: FakeConfig, listen_ip: str = "127.0.0.1", listen_port: int = P.MOTION_HOST_PORT,
                 control_port: Optional[int] = 43999, allow_non_loopback: bool = False,
                 replay_file: Optional[str] = None, replay_loop: bool = True):
        check_address(listen_ip, allow_non_loopback)
        for ip, _ in cfg.registered:
            check_address(ip, allow_non_loopback)
        self.robot = FakeRobot(cfg)
        self.allow_non_loopback = allow_non_loopback
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((listen_ip, listen_port))
        self.listen_port = self.sock.getsockname()[1]
        self.ctl: Optional[socket.socket] = None
        if control_port is not None:
            self.ctl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.ctl.bind(("127.0.0.1", control_port))
            self.control_port = self.ctl.getsockname()[1]
        self.replay = read_capture(replay_file) if replay_file else None
        self.replay_loop = replay_loop
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    # -- loop ---------------------------------------------------------------------------------------------------
    def _send(self, dest: Addr, data: bytes) -> None:
        if not self.allow_non_loopback or ipaddress.ip_address(dest[0]) in ipaddress.ip_network("192.168.0.0/16"):
            check_address(dest[0], self.allow_non_loopback)
        try:
            self.sock.sendto(data, dest)
        except OSError:
            pass

    def serve(self) -> None:
        t0 = time.monotonic()
        replay_idx = 0
        replay_origin = 0.0
        socks = [self.sock] + ([self.ctl] if self.ctl else [])
        while not self._stop.is_set():
            now = time.monotonic() - t0
            with self.lock:
                if self.replay is None:
                    out = self.robot.advance(now)
                    nxt = self.robot.next_tick
                    timeout = 0.05 if nxt is None else max(0.0, min(0.05, nxt - now))
                else:
                    out = []
                    timeout = 0.002
                    while replay_idx < len(self.replay) and self.replay[replay_idx][0] + replay_origin <= now:
                        for dest in self.robot.registered:
                            out.append((dest, self.replay[replay_idx][1]))
                        replay_idx += 1
                    if replay_idx >= len(self.replay) and self.replay_loop and self.replay:
                        replay_idx = 0
                        replay_origin = now + 0.01 - self.replay[0][0]
                for dest, data in out:
                    self._send(dest, data)
            ready, _, _ = select.select(socks, [], [], timeout)
            for s in ready:
                try:
                    data, addr = s.recvfrom(2048)
                except OSError:
                    continue
                with self.lock:
                    if s is self.sock:
                        self.robot.handle(data, addr, time.monotonic() - t0)
                    else:
                        try:
                            reply = self.robot.control(json.loads(data.decode()))
                        except (ValueError, KeyError) as e:
                            reply = {"ok": False, "error": str(e)}
                        s.sendto(json.dumps(reply).encode(), addr)

    def start(self) -> "FakeX30Server":
        self._thread = threading.Thread(target=self.serve, name="fake_x30", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.sock.close()
        if self.ctl:
            self.ctl.close()

    def status(self) -> dict:
        with self.lock:
            return self.robot.status()


def control_request(req: dict, port: int = 43999, timeout: float = 1.0) -> dict:
    """Client helper for the control channel (used by tests and scripts)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(json.dumps(req).encode(), ("127.0.0.1", port))
        data, _ = s.recvfrom(65535)
        return json.loads(data.decode())
    finally:
        s.close()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen-ip", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=P.MOTION_HOST_PORT)
    ap.add_argument("--register", action="append", default=None, metavar="IP:PORT",
                    help="telemetry destination (network.toml ips/ports pair); default 127.0.0.1:43897")
    ap.add_argument("--register-sender", action="store_true", help="also register the heartbeat sender at --telemetry-port")
    ap.add_argument("--no-register", action="store_true", help="register nobody: telemetry never flows")
    ap.add_argument("--telemetry-port", type=int, default=P.TELEMETRY_PORT)
    ap.add_argument("--control-port", type=int, default=43999, help="JSON control channel on 127.0.0.1 (0 = off)")
    ap.add_argument("--allow-non-loopback", action="store_true")
    ap.add_argument("--time-scale", type=float, default=1.0, help="speed up state transitions (e.g. 5)")
    ap.add_argument("--truth-table", choices=["api", "dx", "nb"], default="api",
                    help="speed table the simulated robot really uses (api = vendor Appendix B)")
    ap.add_argument("--lag", type=float, default=0.25, help="first-order velocity lag tau [s]")
    ap.add_argument("--latency", type=float, default=0.10, help="pure command-to-motion delay [s]")
    ap.add_argument("--accel", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.0, help="velocity noise stddev [m/s]")
    ap.add_argument("--odom-scale", type=float, default=1.0)
    ap.add_argument("--yaw-drift", type=float, default=0.0, help="reported yaw drift [rad/s]")
    ap.add_argument("--telemetry-hz", type=float, default=200.0)
    ap.add_argument("--battery", type=int, default=80)
    ap.add_argument("--replay", default=None, help="replay a capture written by record_telemetry.py")
    ap.add_argument("--seed", type=int, default=1)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    regs: List[Addr] = []
    if not args.no_register:
        for r in args.register or [f"127.0.0.1:{args.telemetry_port}"]:
            ip, port = r.rsplit(":", 1)
            regs.append((ip, int(port)))
    cfg = FakeConfig(
        registered=regs, register_sender=args.register_sender, telemetry_port=args.telemetry_port,
        telemetry_hz=args.telemetry_hz, time_scale=args.time_scale, truth_table=args.truth_table,
        lag_tau_s=args.lag, velocity_latency_s=args.latency, max_accel=args.accel, velocity_noise=args.noise,
        odom_scale_error=args.odom_scale, yaw_drift_rad_s=args.yaw_drift, battery_level=args.battery, seed=args.seed)
    srv = FakeX30Server(cfg, args.listen_ip, args.listen_port, args.control_port or None, args.allow_non_loopback,
                        args.replay)
    print(f"fake_x30: listening on {args.listen_ip}:{srv.listen_port}, telemetry to {regs or 'nobody'}, "
          f"control {'127.0.0.1:%d' % srv.control_port if srv.ctl else 'off'}", flush=True)
    try:
        srv.serve()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
