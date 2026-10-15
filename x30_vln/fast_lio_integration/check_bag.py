"""Fail early when a recording lacks FAST-LIO's actual inputs."""
import pathlib
import sqlite3
import sys

bag = pathlib.Path(sys.argv[1]).resolve()
dbs = list(bag.glob("*.db3"))
if not dbs:
    sys.exit(f"No rosbag2 SQLite data in {bag}; metadata alone cannot be played")
topics = {}
for db in dbs:
    with sqlite3.connect(db) as conn:
        topics.update(conn.execute("SELECT name, type FROM topics"))
required = {
    "/livox/lidar": "livox_ros_driver2/msg/CustomMsg",
    "/imu/data": "sensor_msgs/msg/Imu",
}
missing = [f"{name} ({kind})" for name, kind in required.items() if topics.get(name) != kind]
if missing:
    sys.exit("FAST-LIO bag inputs missing: " + ", ".join(missing) + ". Recorded topics: " + repr(topics))
print("FAST-LIO bag topics present:", topics)
