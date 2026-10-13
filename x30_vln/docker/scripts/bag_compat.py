#!/usr/bin/env python3
"""Make a rosbag2 directory written with metadata version >= 7 (Iron/Jazzy, or the `rosbags` converter) readable by Humble.

  bag_compat.py <bagdir>      prints the directory to play: <bagdir> itself if its metadata is already Humble-compatible,
                              otherwise /tmp/x30_bag_compat/<name> with symlinked storage files and a rewritten metadata.yaml
Humble's rosbag2 aborts with "yaml-cpp: bad conversion" on the newer fields (type_description_hash, custom_data, lists of qos).
"""
import os
import sys

import yaml

src = os.path.abspath(sys.argv[1])
meta_path = os.path.join(src, "metadata.yaml")
meta = yaml.safe_load(open(meta_path))["rosbag2_bagfile_information"]
if meta.get("version", 0) <= 6 and not any(t["topic_metadata"].get("type_description_hash") for t in meta["topics_with_message_count"]):
    print(src)
    sys.exit(0)

dst = os.path.join("/tmp/x30_bag_compat", os.path.basename(src))
os.makedirs(dst, exist_ok=True)
for rel in meta["relative_file_paths"]:
    link = os.path.join(dst, rel)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.join(src, rel), link)
meta["version"] = 5
meta.pop("custom_data", None)
for t in meta["topics_with_message_count"]:
    tm = t["topic_metadata"]
    tm.pop("type_description_hash", None)
    q = tm.get("offered_qos_profiles")
    tm["offered_qos_profiles"] = q if isinstance(q, str) else ""
for f in meta.get("files", []):
    f.pop("custom_data", None)
with open(os.path.join(dst, "metadata.yaml"), "w") as fh:
    yaml.safe_dump({"rosbag2_bagfile_information": meta}, fh)
print(dst)
