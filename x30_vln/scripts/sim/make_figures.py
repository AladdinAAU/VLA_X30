#!/usr/bin/env python3
"""Build the progress figures of the simulator report in docs/figures/sim/ (host python: numpy, matplotlib, pillow, rosbags, pymupdf).

Inputs (written by the sim tools): docs/figures/sim/raw/*.png (grab_frame.py), raw/frames*.pdf (tf2_tools view_frames),
results/sim_lidar_compare/sim_scans.npz, results/sim_soak/kinematic_soak.csv, the real bag.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'figures' / 'sim'
RAW = OUT / 'raw'


def save(fig, name):
    fig.savefig(OUT / name, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(name, (OUT / name).stat().st_size // 1024, 'kB')


def rgb_thermal():
    rgb = Image.open(RAW / 'camera_color_image_raw.png').convert('RGB')
    th = np.array(Image.open(RAW / 'thermal_image_raw.png'))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].imshow(rgb)
    ax[0].set_title('RGB 640x480 (/camera/color/image_raw)')
    ax[1].imshow(th, cmap='inferno')
    ax[1].set_title('thermal 384x288, mono8 display (/thermal/image_raw)')
    for a in ax:
        a.axis('off')
    save(fig, 'rgb_thermal.png')


def cloud_xy(scan, origins):
    p = np.c_[scan['x'], scan['y'], scan['z']].astype(float)
    block = np.clip((scan['line'] // 4).astype(int), 0, 3)
    valid = np.linalg.norm(p - origins[block], axis=1) > 0.01
    return p[valid]


def lidar_compare():
    import yaml
    from rosbags.highlevel import AnyReader
    cfg = yaml.safe_load(open(ROOT / 'configs' / 'robot_sim.yaml', encoding='utf-8'))
    origins = np.array([cfg['lidar']['mounts'][k]['position_mm'] for k in cfg['lidar']['merged_order']]) / 1000.0
    with AnyReader([ROOT / 'datasets' / 'real' / 'lab_2026-10-06']) as reader:
        for conn, _t, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            dt = np.dtype({'names': [f.name for f in msg.fields], 'formats': ['<f4', '<f4', '<f4', '<f4', 'u1', 'u1', '<f8'],
                           'offsets': [f.offset for f in msg.fields], 'itemsize': msg.point_step})
            real = np.frombuffer(bytes(msg.data), dtype=dt)
            break
    sim = np.load(ROOT / 'results' / 'sim_lidar_compare' / 'sim_scans.npz')['scans'][-1]
    arena = np.load(ROOT / 'results' / 'sim_lidar_compare' / 'arena_scans.npz')['scans'][-1]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, scan, title, lim in ((axs[0], real, 'real bag (lab, robot lying down), top view', 8), (axs[1], sim, 'sim: lab_room, same colour scale', 8),
                                 (axs[2], arena, 'sim: default_arena (12 x 10 m)', 8)):
        p = cloud_xy(scan, origins)
        sc = ax.scatter(p[:, 1], p[:, 0], c=np.clip(p[:, 2], -0.6, 2.8), s=0.3, cmap='viridis', vmin=-0.6, vmax=2.8)
        ax.plot(0, 0, 'r^', ms=8)
        ax.set_xlim(lim, -lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\n{len(p)} valid points of {len(scan)}', fontsize=9)
        ax.set_xlabel('y (m, left = +)')
        ax.set_ylabel('x (m, forward)')
    fig.colorbar(sc, ax=axs, shrink=0.7, label='z in body frame (m)')
    save(fig, 'lidar_real_vs_sim.png')


def tf_tree():
    import fitz
    pdfs = sorted(RAW.glob('frames*.pdf'))
    if not pdfs:
        return
    page = fitz.open(pdfs[-1])[0]
    pix = page.get_pixmap(dpi=90)
    pix.save(OUT / 'tf_tree.png')
    print('tf_tree.png', (OUT / 'tf_tree.png').stat().st_size // 1024, 'kB')


def soak():
    rows = list(csv.DictReader(open(ROOT / 'results' / 'sim_soak' / 'kinematic_soak.csv')))
    t = [float(r['t']) / 60 for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(t, [float(r['rtf']) for r in rows])
    ax[0].set_ylim(0, 1.2)
    ax[0].set_title('real-time factor (kinematic, all sensors)')
    ax[1].plot(t, [float(r['gz_rss_mb']) for r in rows], label='gz sim')
    ax[1].plot(t, [float(r['ros_rss_mb']) for r in rows], label='ROS nodes')
    ax[1].set_ylim(0, 800)
    ax[1].legend()
    ax[1].set_title('resident memory (MB), 12 min soak')
    ax[2].plot(t, [float(r['hz_lidar_points']) for r in rows], label='lidar')
    ax[2].plot(t, [float(r['hz_camera_color_image_raw']) for r in rows], label='rgb')
    ax[2].plot(t, [float(r['hz_thermal_image_raw']) for r in rows], label='thermal')
    ax[2].legend()
    ax[2].set_title('wall-clock topic rates (Hz; x RTF = sim-time rates)')
    for a in ax:
        a.set_xlabel('minutes')
        a.grid(alpha=0.3)
    save(fig, 'soak_rtf_memory.png')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    rgb_thermal()
    lidar_compare()
    tf_tree()
    soak()
