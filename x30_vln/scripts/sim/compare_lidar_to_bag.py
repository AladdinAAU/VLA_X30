#!/usr/bin/env python3
"""Compare the simulated merged lidar cloud with the recorded real bag (distribution shape, not geometry).

  python scripts/sim/compare_lidar_to_bag.py --sim results/sim_lidar_compare/sim_scans.npz [--bag datasets/real/lab_2026-10-06] [--out DIR]

The real side is read with the `rosbags` library (no ROS needed). The simulated side is an .npz written by capture_sim_scans.py
(or a ROS 2 bag directory). Both are split into the four lidar blocks (line // 4; order rear-up, rear-down, front-down, front-up in the
real cloud, `merged_order` in configs/robot_sim.yaml) and compared on: points per scan, invalid (sensor-origin) fraction, range
histogram, elevation histogram (relative to each sensor origin), azimuth coverage, height histogram, intensity percentiles, tag and line
values, timestamp span. Prints a table, writes report.md, report.json and histograms.png into --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

RANGE_BINS = [0, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20, 40, 1e9]
ELEV_BINS = np.arange(-90, 91, 15)
Z_BINS = np.arange(-1.0, 3.51, 0.25)
DTYPE = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'tag', 'line', 'timestamp'], 'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
                  'offsets': [0, 4, 8, 12, 16, 17, 18], 'itemsize': 26})


def load_bag(path: Path, max_scans: int):
    from rosbags.highlevel import AnyReader
    scans, stamps = [], []
    with AnyReader([path]) as reader:
        conns = [c for c in reader.connections if c.topic == '/lidar_points']
        for conn, _t, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            dt = np.dtype({'names': [f.name for f in msg.fields], 'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
                           'offsets': [f.offset for f in msg.fields], 'itemsize': msg.point_step})
            scans.append(np.frombuffer(bytes(msg.data), dtype=dt).copy())
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            if len(scans) >= max_scans:
                break
    return scans, np.array(stamps)


def load_npz(path: Path):
    d = np.load(path, allow_pickle=False)
    return [s for s in d['scans']], d['header_stamps']


def block_ids(scan):
    return (scan['line'] // 4).astype(int)


def analyse(scans, stamps, origins_by_block):
    """origins_by_block: (4,3) sensor positions in the order the blocks appear in the cloud."""
    out = {'scans': len(scans)}
    npts = np.array([len(s) for s in scans])
    out['points_per_scan'] = {'mean': float(npts.mean()), 'std': float(npts.std()), 'min': int(npts.min()), 'max': int(npts.max())}
    rng_h = np.zeros(len(RANGE_BINS) - 1)
    elev_h = np.zeros(len(ELEV_BINS) - 1)
    z_h = np.zeros(len(Z_BINS) - 1)
    az_cov, valid_frac, inten, tags_v, tags_i, per_block = [], [], [], {}, {}, [[] for _ in range(4)]
    nvalid_total = 0
    span = []
    for s in scans:
        p = np.c_[s['x'], s['y'], s['z']].astype(float)
        b = np.clip(block_ids(s), 0, 3)
        rel = p - origins_by_block[b]
        r = np.linalg.norm(rel, axis=1)
        valid = r > 0.01
        valid_frac.append(valid.mean())
        for k in range(4):
            sel = b == k
            per_block[k].append((float(valid[sel].mean()) if sel.any() else 0.0, float(np.median(r[sel & valid])) if (sel & valid).any() else 0.0))
        rv, relv = r[valid], rel[valid]
        rng_h += np.histogram(rv, bins=RANGE_BINS)[0]
        elev_h += np.histogram(np.degrees(np.arcsin(np.clip(relv[:, 2] / rv, -1, 1))), bins=ELEV_BINS)[0]
        z_h += np.histogram(p[valid, 2], bins=Z_BINS)[0]
        az = np.degrees(np.arctan2(p[valid, 1], p[valid, 0]))
        az_cov.append((np.histogram(az, bins=36, range=(-180, 180))[0] > 5).mean())
        inten.append(s['intensity'][valid])
        for flag, tags in ((valid, tags_v), (~valid, tags_i)):
            u, c = np.unique(s['tag'][flag], return_counts=True)
            for uu, cc in zip(u, c):
                tags[int(uu)] = tags.get(int(uu), 0) + int(cc)
        nvalid_total += int(valid.sum())
        ts = s['timestamp']
        span.append(float(ts.max() - ts.min()))
    inten = np.concatenate(inten)
    out['valid_fraction'] = float(np.mean(valid_frac))
    out['per_block'] = {f'block{k}': {'valid_fraction': float(np.mean([v[0] for v in per_block[k]])),
                                      'median_range': float(np.mean([v[1] for v in per_block[k]]))} for k in range(4)}
    out['range_hist'] = (rng_h / rng_h.sum()).tolist()
    out['elevation_hist'] = (elev_h / elev_h.sum()).tolist()
    out['height_hist'] = (z_h / z_h.sum()).tolist()
    out['azimuth_coverage_36bins'] = float(np.mean(az_cov))
    out['intensity_pct'] = {str(q): float(np.percentile(inten, q)) for q in (5, 25, 50, 75, 95, 100)}
    out['tags_valid'] = {k: round(v / max(sum(tags_v.values()), 1), 4) for k, v in sorted(tags_v.items())}
    out['tags_invalid'] = {k: round(v / max(sum(tags_i.values()), 1), 4) for k, v in sorted(tags_i.items())}
    out['lines'] = sorted(int(v) for v in np.unique(np.concatenate([s['line'] for s in scans])))
    out['timestamp_span_ms'] = float(np.mean(span))
    t0 = np.array([s['timestamp'].min() for s in scans])
    out['header_minus_first_timestamp_ms'] = float(np.mean(stamps * 1000.0 - t0))
    out['range_percentiles'] = None
    return out


def js(p, q):
    p, q = np.asarray(p) + 1e-9, np.asarray(q) + 1e-9
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def labels(bins, unit=''):
    return [f'{bins[i]:g}-{bins[i + 1]:g}{unit}' if bins[i + 1] < 1e8 else f'>{bins[i]:g}{unit}' for i in range(len(bins) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', default='datasets/real/lab_2026-10-06')
    ap.add_argument('--sim', required=True, help='.npz from capture_sim_scans.py or a ROS 2 bag directory')
    ap.add_argument('--config', default='configs/robot_sim.yaml')
    ap.add_argument('--out', default='results/sim_lidar_compare')
    ap.add_argument('--max-scans', type=int, default=60)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding='utf-8'))
    order = cfg['lidar']['merged_order']
    origins = np.array([cfg['lidar']['mounts'][k]['position_mm'] for k in order]) / 1000.0
    real_scans, real_t = load_bag(Path(a.bag), a.max_scans)
    sp = Path(a.sim)
    sim_scans, sim_t = load_npz(sp) if sp.suffix == '.npz' else load_bag(sp, a.max_scans)
    real = analyse(real_scans, real_t, origins)
    sim = analyse(sim_scans, sim_t, origins)
    dist = {k: js(real[k], sim[k]) for k in ('range_hist', 'elevation_hist', 'height_hist')}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps({'real': real, 'sim': sim, 'jensen_shannon': dist}, indent=1))

    L = []
    L.append(f'# Lidar: simulation vs recorded bag\n\nbag `{a.bag}` ({real["scans"]} scans), sim `{a.sim}` ({sim["scans"]} scans). Geometry differs (real lab vs sim room, '
             'the real robot was lying down); compare the shape of the distributions.\n')
    L.append('| statistic | real bag | simulation |\n|---|---|---|')
    L.append(f'| points per scan (mean +- std) | {real["points_per_scan"]["mean"]:.0f} +- {real["points_per_scan"]["std"]:.0f} | {sim["points_per_scan"]["mean"]:.0f} +- {sim["points_per_scan"]["std"]:.0f} |')
    L.append(f'| valid fraction (not at sensor origin) | {real["valid_fraction"]:.3f} | {sim["valid_fraction"]:.3f} |')
    for k in range(4):
        r, s = real['per_block'][f'block{k}'], sim['per_block'][f'block{k}']
        L.append(f'| block {k} ({order[k]}) valid / median range m | {r["valid_fraction"]:.2f} / {r["median_range"]:.2f} | {s["valid_fraction"]:.2f} / {s["median_range"]:.2f} |')
    L.append(f'| azimuth coverage (36 bins, > 5 pts) | {real["azimuth_coverage_36bins"]:.2f} | {sim["azimuth_coverage_36bins"]:.2f} |')
    L.append(f'| intensity p5/p25/p50/p75/p95/max | {"/".join(f"{v:.0f}" for v in real["intensity_pct"].values())} | {"/".join(f"{v:.0f}" for v in sim["intensity_pct"].values())} |')
    L.append(f'| line values | {real["lines"][0]}..{real["lines"][-1]} ({len(real["lines"])}) | {sim["lines"][0]}..{sim["lines"][-1]} ({len(sim["lines"])}) |')
    L.append(f'| tag (valid points) | {real["tags_valid"]} | {sim["tags_valid"]} |')
    L.append(f'| tag (invalid points) | {real["tags_invalid"]} | {sim["tags_invalid"]} |')
    L.append(f'| timestamp span within a scan (ms) | {real["timestamp_span_ms"]:.1f} | {sim["timestamp_span_ms"]:.1f} |')
    L.append(f'| header stamp - first point timestamp (ms) | {real["header_minus_first_timestamp_ms"]:.1f} | {sim["header_minus_first_timestamp_ms"]:.1f} |')
    L.append(f'\nJensen-Shannon distance (0 = identical): range {dist["range_hist"]:.3f}, elevation {dist["elevation_hist"]:.3f}, height {dist["height_hist"]:.3f}\n')
    for title, key, bins, unit in (('Range', 'range_hist', RANGE_BINS, ' m'), ('Elevation about each sensor origin', 'elevation_hist', list(ELEV_BINS), ' deg'),
                                   ('Height z in the body frame', 'height_hist', list(Z_BINS), ' m')):
        L.append(f'## {title} (fraction of valid points)\n\n| bin | real | sim |\n|---|---|---|')
        for lab, r, s in zip(labels(bins, unit), real[key], sim[key]):
            L.append(f'| {lab} | {r:.3f} | {s:.3f} |')
        L.append('')
    (out / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L[:20]))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        for ax, (title, key, bins, unit) in zip(axs, (('range', 'range_hist', RANGE_BINS, ' m'), ('elevation', 'elevation_hist', list(ELEV_BINS), ' deg'),
                                                      ('height z', 'height_hist', list(Z_BINS), ' m'))):
            x = np.arange(len(real[key]))
            ax.bar(x - 0.2, real[key], 0.4, label='real bag')
            ax.bar(x + 0.2, sim[key], 0.4, label='simulation')
            ax.set_xticks(x[::max(1, len(x) // 8)])
            ax.set_xticklabels(labels(bins, unit)[::max(1, len(x) // 8)], rotation=45, ha='right', fontsize=7)
            ax.set_title(title)
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'histograms.png', dpi=110)
    except ImportError:
        print('matplotlib missing: no histograms.png')


if __name__ == '__main__':
    main()
