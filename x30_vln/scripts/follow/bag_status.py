"""Print /follow/status from a bag in a time window: python3 bag_status.py <bag_dir> <t0> <t1>  (seconds since first status). Run in e2e-fw."""
import sys, json
import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

r = rosbag2_py.SequentialCompressionReader()
r.open(rosbag2_py.StorageOptions(uri=sys.argv[1], storage_id=''), rosbag2_py.ConverterOptions('', ''))
t0 = None
lo, hi = float(sys.argv[2]), float(sys.argv[3])
last = None
while r.has_next():
    topic, data, ts = r.read_next()
    if topic != '/follow/status':
        continue
    if t0 is None:
        t0 = ts
    t = (ts - t0) / 1e9
    if lo <= t <= hi:
        d = json.loads(deserialize_message(data, String).data)
        key = (d.get('state'), d.get('target'))
        keep = {k: d[k] for k in d if k in ('state', 'target', 'range', 'bearing', 'cost', 'second', 'why', 'lost_for', 'mode', 'goal', 'line_clear', 'cmd')}
        print(round(t, 1), json.dumps(keep)[:400])
