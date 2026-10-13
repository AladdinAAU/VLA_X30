"""Vertex-clustering decimation of the robot's visual meshes (OBJ / binary STL -> OBJ), numpy only.

The upstream X30 meshes hold about 1.3 million vertices (shank.obj alone 210 k, four times). Every Gazebo sensor frame re-renders them,
which costs about 100 ms per frame under software rendering and made the simulator run at 0.2x real time even with tiny sensors.
The robot's own visuals matter little to the sensors (lidar self-hits are masked, the cameras look forward), so a coarse copy is used.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                verts.append(line.split()[1:4])
            elif line.startswith('f '):
                idx = [int(tok.split('/')[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):          # triangulate fans
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def read_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = Path(path).read_bytes()
    if data[:5] == b'solid' and b'facet' in data[:300]:
        raise ValueError('ascii STL not supported')
    n = struct.unpack('<I', data[80:84])[0]
    rec = np.frombuffer(data, dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n, offset=84)
    tri = rec['v'].reshape(-1, 3).astype(np.float64)
    verts, inv = np.unique(tri.round(6), axis=0, return_inverse=True)
    return verts, inv.reshape(-1, 3)


def cluster(verts: np.ndarray, faces: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge all vertices inside a voxel into their mean; drop degenerate faces."""
    keys = np.floor(verts / voxel).astype(np.int64)
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    new_v = np.zeros((len(first), 3))
    np.add.at(new_v, inv, verts)
    new_v /= np.bincount(inv, minlength=len(first))[:, None]
    f = inv[faces]
    ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[ok]
    f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f
    return new_v, f


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, 'w', encoding='utf-8') as out:
        out.write('# decimated by x30_sim.mesh_simplify\n')
        for v in verts:
            out.write(f'v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n')
        for t in faces:
            out.write(f'f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n')


def simplified_copy(src: Path, out_dir: Path, voxel: float) -> Path:
    """Return the path of the decimated OBJ for src (created on first use, reused afterwards)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f'{src.stem}_v{int(round(voxel * 1000))}mm.obj'
    if dst.is_file():
        return dst
    verts, faces = read_stl(src) if src.suffix.lower() == '.stl' else read_obj(src)
    nv, nf = cluster(verts, faces, voxel)
    write_obj(dst, nv, nf)
    return dst
