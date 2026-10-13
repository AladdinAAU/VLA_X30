#!/usr/bin/env python3
"""DDS interop probe: publish and/or subscribe realistic robot traffic and report rates, latency, drops and CPU.

Traffic (sizes taken from the real robot bag and the interface contract):
  <prefix>/lidar_points  PointCloud2, 80000 pts x 26 B = 2.08 MB, 10 Hz
  <prefix>/image         Image rgb8 640x480 = 0.92 MB, 15 Hz
  <prefix>/tf            TFMessage with 3 transforms, 50 Hz
  <prefix>/cmd_vel       Twist, 20 Hz
Latency = receive wall time - header.stamp (publisher wall clock; both ends share the host kernel clock, so the figure is a
valid one-way estimate that includes serialisation, UDP fragmentation and Python callback queueing). The sequence number
travels in header.frame_id so drops are counted exactly.

Usage (run the same file in both containers, pubs and subs mirrored):
  dds_probe.py --pub h2j --sub j2h --duration 20 --warmup 4 --qos best_effort --out /tmp/humble.json
"""
import argparse
import json
import statistics
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, PointField, PointCloud2
from tf2_msgs.msg import TFMessage

CLOUD_PTS = 80000
CLOUD_STEP = 26
RATES = {"lidar_points": 10.0, "image": 15.0, "tf": 50.0, "cmd_vel": 20.0}


def now_stamp(msg_stamp) -> float:
    t = time.time()
    msg_stamp.sec = int(t)
    msg_stamp.nanosec = int((t - int(t)) * 1e9)
    return t


SEC_MARK = 0x11223344
SEQ_WIDTH = 10


def build_templates():
    """Serialise one message per topic with marker values and locate stamp / sequence fields inside the CDR buffer.
    Publishing pre-serialised bytes and receiving raw bytes keeps Python out of the data path, so the probe measures DDS and
    not rclpy's message conversion (which is slow for multi-MB uint8 arrays on Humble)."""
    import struct
    from rclpy.serialization import serialize_message
    zero = "0" * SEQ_WIDTH
    cloud = PointCloud2()
    cloud.header.stamp.sec = SEC_MARK; cloud.header.frame_id = zero
    cloud.height = 1; cloud.width = CLOUD_PTS; cloud.point_step = CLOUD_STEP; cloud.row_step = CLOUD_STEP * CLOUD_PTS
    cloud.fields = [PointField(name=n, offset=o, datatype=7, count=1) for n, o in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))]
    cloud.is_dense = True; cloud.data = bytes(CLOUD_PTS * CLOUD_STEP)
    img = Image()
    img.header.stamp.sec = SEC_MARK; img.header.frame_id = zero
    img.height = 480; img.width = 640; img.encoding = "rgb8"; img.step = 640 * 3; img.data = bytes(640 * 480 * 3)
    tf = TFMessage()
    for i in range(3):
        t = TransformStamped(); t.header.stamp.sec = SEC_MARK; t.header.frame_id = zero; t.child_frame_id = f"c{i}"
        t.transform.rotation.w = 1.0
        tf.transforms.append(t)
    tw = Twist(); tw.linear.x = 1234.5; tw.angular.z = 6789.5
    out = {}
    for name, msg in (("lidar_points", cloud), ("image", img), ("tf", tf), ("cmd_vel", tw)):
        buf = bytearray(serialize_message(msg))
        if name == "cmd_vel":
            out[name] = (buf, buf.find(struct.pack("<d", 1234.5)), buf.find(struct.pack("<d", 6789.5)))
        else:
            out[name] = (buf, buf.find(struct.pack("<i", SEC_MARK)), buf.find(zero.encode()))
        assert out[name][1] > 0 and out[name][2] > 0, name
    return out


class Probe(Node):
    def __init__(self, a):
        super().__init__("dds_probe_" + (a.pub or "none") + "_" + str(int(time.time()) % 1000))
        self.a = a
        reliable = a.qos == "reliable"
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5, durability=DurabilityPolicy.VOLATILE)
        ctl_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        self.stats = {k: {"n": 0, "lat": [], "seqs": [], "t0": None, "bytes": 0} for k in RATES}
        self.measuring = False
        self.seq = {k: 0 for k in RATES}
        self.tpl = build_templates()
        types = {"lidar_points": (PointCloud2, sensor_qos), "image": (Image, sensor_qos), "tf": (TFMessage, ctl_qos), "cmd_vel": (Twist, ctl_qos)}

        if a.sub:
            for name, (typ, qos) in types.items():
                self.create_subscription(typ, f"/{a.sub}/{name}", lambda raw, n=name: self.rx(n, raw), qos, raw=True)
        if a.pub:
            self.pubs = {n: self.create_publisher(t, f"/{a.pub}/{n}", q) for n, (t, q) in types.items()}
            for name, hz in RATES.items():
                self.create_timer(1.0 / hz, lambda n=name: self.publish(n))

    def publish(self, name):
        import struct
        s = self.seq[name]
        self.seq[name] += 1
        buf, o_stamp, o_seq = self.tpl[name]
        t = time.time()
        if name == "cmd_vel":
            struct.pack_into("<d", buf, o_stamp, float(s))   # linear.x = sequence
            struct.pack_into("<d", buf, o_seq, t)            # angular.z = send time
        else:
            struct.pack_into("<iI", buf, o_stamp, int(t), int((t - int(t)) * 1e9))
            buf[o_seq:o_seq + SEQ_WIDTH] = b"%0*d" % (SEQ_WIDTH, s)
        self.pubs[name].publish(bytes(buf))

    def rx(self, name, raw):
        import struct
        t = time.time()
        if not self.measuring:
            return
        _, o_stamp, o_seq = self.tpl[name]
        if name == "cmd_vel":
            seq = int(struct.unpack_from("<d", raw, o_stamp)[0]); sent = struct.unpack_from("<d", raw, o_seq)[0]
        else:
            sec, nsec = struct.unpack_from("<iI", raw, o_stamp)
            sent = sec + nsec * 1e-9; seq = int(raw[o_seq:o_seq + SEQ_WIDTH])
        st = self.stats[name]
        if st["t0"] is None:
            st["t0"] = t
        st["n"] += 1; st["bytes"] += len(raw); st["lat"].append(t - sent); st["seqs"].append(seq); st["tl"] = t


def summarise(stats, duration):
    out = {}
    for name, st in stats.items():
        if st["n"] == 0:
            out[name] = {"received": 0}
            continue
        seqs = st["seqs"]
        expected = seqs[-1] - seqs[0] + 1
        lat = sorted(st["lat"])
        span = max(st["tl"] - st["t0"], 1e-6)
        out[name] = {
            "received": st["n"], "expected_in_span": expected, "dropped": expected - st["n"],
            "rate_hz": round((st["n"] - 1) / span, 2) if st["n"] > 1 else 0.0, "nominal_hz": RATES[name],
            "lat_ms_mean": round(statistics.mean(lat) * 1e3, 1), "lat_ms_p50": round(lat[len(lat) // 2] * 1e3, 1),
            "lat_ms_p95": round(lat[int(len(lat) * 0.95) - 1] * 1e3, 1), "lat_ms_max": round(lat[-1] * 1e3, 1),
            "MB_per_s": round(st["bytes"] / span / 1e6, 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pub", default="", help="topic prefix to publish on")
    ap.add_argument("--sub", default="", help="topic prefix to subscribe to")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=4.0, help="seconds before statistics start (discovery)")
    ap.add_argument("--qos", choices=["best_effort", "reliable"], default="best_effort")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rclpy.init()
    node = Probe(a)
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()
    time.sleep(a.warmup)
    cpu0, wall0 = time.process_time(), time.time()
    node.measuring = True
    time.sleep(a.duration)
    node.measuring = False
    cpu = (time.process_time() - cpu0) / (time.time() - wall0) * 100.0
    res = {"pub": a.pub, "sub": a.sub, "qos": a.qos, "duration_s": a.duration, "process_cpu_percent_of_one_core": round(cpu, 1),
           "topics": summarise(node.stats, a.duration)}
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    rclpy.shutdown()
    th.join(timeout=3)
    node.destroy_node()


if __name__ == "__main__":
    main()
