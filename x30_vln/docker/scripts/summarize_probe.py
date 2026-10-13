#!/usr/bin/env python3
"""Print dds_probe JSON files as one table: summarize_probe.py file.json ..."""
import json
import sys

for f in sys.argv[1:]:
    d = json.load(open(f))
    print(f"{f}: pub={d['pub']} sub={d['sub']} qos={d['qos']} process CPU={d['process_cpu_percent_of_one_core']}% of one core")
    for n, t in d["topics"].items():
        if not t.get("received"):
            print(f"   {n:13s} NOTHING RECEIVED"); continue
        print(f"   {n:13s} {t['rate_hz']:6.2f}/{t['nominal_hz']:.0f} Hz  dropped {t['dropped']:3d}/{t['expected_in_span']:3d}  "
              f"lat ms mean {t['lat_ms_mean']:6.1f} p95 {t['lat_ms_p95']:6.1f} max {t['lat_ms_max']:6.1f}  {t['MB_per_s']:5.2f} MB/s")
