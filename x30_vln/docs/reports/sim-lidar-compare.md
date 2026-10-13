# Lidar: simulation vs recorded bag

bag `datasets/real/lab_2026-10-06` (60 scans), sim `results/sim_lidar_compare/sim_scans.npz` (30 scans). Geometry differs (real lab vs sim room, the real robot was lying down); compare the shape of the distributions.

| statistic | real bag | simulation |
|---|---|---|
| points per scan (mean +- std) | 80000 +- 783 | 79872 +- 0 |
| valid fraction (not at sensor origin) | 0.472 | 0.596 |
| block 0 (rl) valid / median range m | 0.52 / 1.79 | 0.72 / 1.80 |
| block 1 (rr) valid / median range m | 0.39 / 1.19 | 0.43 / 0.51 |
| block 2 (fr) valid / median range m | 0.41 / 1.70 | 0.54 / 1.02 |
| block 3 (fl) valid / median range m | 0.57 / 1.71 | 0.70 / 2.27 |
| azimuth coverage (36 bins, > 5 pts) | 1.00 | 1.00 |
| intensity p5/p25/p50/p75/p95/max | 0/1/4/18/81/255 | 0/1/3/15/106/255 |
| line values | 0..15 (16) | 0..15 (16) |
| tag (valid points) | {0: 0.7753, 1: 0.0059, 2: 0.0062, 4: 0.0059, 5: 0.0, 6: 0.0, 8: 0.0008, 9: 0.0, 10: 0.0, 16: 0.1907, 17: 0.0, 18: 0.0, 32: 0.0137, 33: 0.0, 34: 0.0, 64: 0.0012, 65: 0.0, 66: 0.0, 68: 0.0, 72: 0.0, 74: 0.0} | {0: 0.7763, 1: 0.0057, 2: 0.0061, 4: 0.0059, 8: 0.0006, 16: 0.1922, 32: 0.0119, 64: 0.0014} |
| tag (invalid points) | {0: 0.6535, 1: 0.0001, 2: 0.0, 4: 0.0001, 5: 0.0, 6: 0.0, 8: 0.0, 9: 0.0, 16: 0.0135, 17: 0.0, 18: 0.0, 32: 0.002, 33: 0.0, 34: 0.0, 64: 0.0013, 65: 0.0, 72: 0.0, 128: 0.0283, 192: 0.3013} | {0: 0.6344, 16: 0.0118, 32: 0.0021, 64: 0.0012, 128: 0.0297, 192: 0.3208} |
| timestamp span within a scan (ms) | 100.3 | 100.0 |
| header stamp - first point timestamp (ms) | 0.0 | 0.0 |

Jensen-Shannon distance (0 = identical): range 0.055, elevation 0.016, height 0.027

## Range (fraction of valid points)

| bin | real | sim |
|---|---|---|
| 0-0.2 m | 0.155 | 0.127 |
| 0.2-0.5 m | 0.134 | 0.221 |
| 0.5-1 m | 0.091 | 0.087 |
| 1-2 m | 0.196 | 0.116 |
| 2-3 m | 0.245 | 0.169 |
| 3-5 m | 0.115 | 0.280 |
| 5-8 m | 0.025 | 0.000 |
| 8-12 m | 0.032 | 0.000 |
| 12-20 m | 0.006 | 0.000 |
| 20-40 m | 0.000 | 0.000 |
| >40 m | 0.000 | 0.000 |

## Elevation about each sensor origin (fraction of valid points)

| bin | real | sim |
|---|---|---|
| -90--75 deg | 0.007 | 0.001 |
| -75--60 deg | 0.017 | 0.039 |
| -60--45 deg | 0.035 | 0.081 |
| -45--30 deg | 0.066 | 0.113 |
| -30--15 deg | 0.106 | 0.131 |
| -15-0 deg | 0.156 | 0.143 |
| 0-15 deg | 0.171 | 0.126 |
| 15-30 deg | 0.139 | 0.109 |
| 30-45 deg | 0.115 | 0.090 |
| 45-60 deg | 0.082 | 0.078 |
| 60-75 deg | 0.082 | 0.067 |
| 75-90 deg | 0.024 | 0.024 |

## Height z in the body frame (fraction of valid points)

| bin | real | sim |
|---|---|---|
| -1--0.75 m | 0.000 | 0.000 |
| -0.75--0.5 m | 0.000 | 0.000 |
| -0.5--0.25 m | 0.000 | 0.000 |
| -0.25-0 m | 0.401 | 0.508 |
| 0-0.25 m | 0.112 | 0.060 |
| 0.25-0.5 m | 0.072 | 0.048 |
| 0.5-0.75 m | 0.067 | 0.034 |
| 0.75-1 m | 0.048 | 0.020 |
| 1-1.25 m | 0.025 | 0.020 |
| 1.25-1.5 m | 0.016 | 0.017 |
| 1.5-1.75 m | 0.009 | 0.018 |
| 1.75-2 m | 0.020 | 0.021 |
| 2-2.25 m | 0.038 | 0.021 |
| 2.25-2.5 m | 0.169 | 0.233 |
| 2.5-2.75 m | 0.018 | 0.000 |
| 2.75-3 m | 0.001 | 0.000 |
| 3-3.25 m | 0.002 | 0.000 |
| 3.25-3.5 m | 0.003 | 0.000 |
