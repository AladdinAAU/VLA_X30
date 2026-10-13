"""Annotated top-down semantic map (MapNav-style ASM) as a raster image and as text (no ROS imports).

The image is built from the 2D occupancy grid (free = white, occupied = dark, unknown = grey) with room regions tinted
and named, the robot trajectory, the robot as an arrow showing its heading, and every object drawn as its footprint with
an "id label" text. The north-up layout has +x to the right and +y up (the map frame); `egocentric=True` rotates it so the
robot heading points up. The text variant is a character grid with a legend.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .config import RenderCfg
from .gridtools import GridView
from .rooms import region_label
from .semantic_map import MapObject, SemanticMap

UNKNOWN_RGB = (205, 205, 210)
FREE_RGB = (255, 255, 255)
OCC_RGB = (45, 45, 55)
ROBOT_RGB = (20, 90, 220)
TRAJ_RGB = (90, 160, 255)
ROOM_TINTS = [(255, 214, 170), (190, 230, 190), (200, 205, 250), (250, 200, 230), (255, 240, 170), (190, 235, 240)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class _View:
    """World (x, y) -> canvas pixel (u, v): rotation by `rot` (rad), uniform scale, y flipped."""
    scale: float
    rot: float
    origin: tuple[float, float]       # world point mapped to the canvas offset (after rotation)
    offset: tuple[float, float]
    size: tuple[int, int]             # (width, height) of the canvas

    def matrix(self) -> np.ndarray:
        c, s = math.cos(self.rot), math.sin(self.rot)
        # rotate the world, then flip y into the image, then scale and translate
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        flip = np.array([[self.scale, 0.0, 0.0], [0.0, -self.scale, 0.0], [0.0, 0.0, 1.0]])
        shift = np.array([[1.0, 0.0, self.offset[0]], [0.0, 1.0, self.offset[1]], [0.0, 0.0, 1.0]])
        return shift @ flip @ rot

    def to_px(self, x: float, y: float) -> tuple[int, int]:
        p = self.matrix() @ np.array([x, y, 1.0])
        return int(round(p[0])), int(round(p[1]))


def _make_view(points: np.ndarray, rot: float, max_px: int, margin_m: float) -> _View:
    c, s = math.cos(rot), math.sin(rot)
    rp = points @ np.array([[c, s], [-s, c]])        # (R p)^T = p^T R^T
    lo, hi = rp.min(axis=0) - margin_m, rp.max(axis=0) + margin_m
    span = np.maximum(hi - lo, 1e-3)
    scale = max_px / float(span.max())
    w, h = int(math.ceil(span[0] * scale)), int(math.ceil(span[1] * scale))
    return _View(scale, rot, (float(lo[0]), float(hi[1])), (-lo[0] * scale, hi[1] * scale), (w, h))


def _halo_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color=(20, 20, 20), thick: int = 1
               ) -> tuple[int, int, int, int]:
    """Draw text with a white halo; returns its bounding box (x0, y0, x1, y1)."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    for dx in (-1, 0, 1):          # same thickness at 8 offsets: glyph spacing depends on the thickness
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(img, text, (org[0] + dx, org[1] + dy), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)
    return org[0] - 2, org[1] - th - 2, org[0] + tw + 2, org[1] + base + 2


def _overlaps(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


class MapRenderer:
    def __init__(self, cfg: RenderCfg):
        self.cfg = cfg

    # ------------------------------------------------------------------------------------ image
    def render(self, smap: SemanticMap, grid: GridView | None, pose: tuple[float, float, float] | None,
               trajectory: list[tuple[float, float]] | None = None, egocentric: bool | None = None,
               show_captions: bool | None = None, max_size_px: int | None = None, include_tentative: bool = False,
               title: str = '') -> np.ndarray:
        cfg = self.cfg
        egocentric = cfg.egocentric if egocentric is None else egocentric
        show_captions = cfg.show_captions if show_captions is None else show_captions
        max_px = max_size_px or cfg.max_size_px
        with smap.lock:
            objs = smap.visible_objects(include_tentative)
            regions = dict(smap.regions)
            room_map = smap.room_map
            names = {o.id: smap.label(o) for o in objs}
            captions = {o.id: o.caption for o in objs}
            colors = {o.id: smap.vocab.color(smap.label(o)) for o in objs}
            status = {o.id: smap.status(o) for o in objs}
        pts = [np.array(o.position[:2]) for o in objs]
        if grid is not None and grid.known_bounds() is not None:
            x0, y0, x1, y1 = grid.known_bounds()
            pts += [np.array([x0, y0]), np.array([x1, y1]), np.array([x0, y1]), np.array([x1, y0])]
        if pose is not None:
            pts.append(np.array(pose[:2]))
        if trajectory:
            pts += [np.array(p) for p in trajectory[::max(1, len(trajectory) // 50)]]
        if not pts:
            pts = [np.zeros(2), np.ones(2)]
        rot = (math.pi / 2.0 - pose[2]) if (egocentric and pose is not None) else 0.0
        view = _make_view(np.array(pts), rot, max_px, cfg.margin_m)
        w, h = view.size
        img = np.full((h, w, 3), UNKNOWN_RGB, np.uint8)
        m3 = view.matrix()
        if grid is not None:
            img = self._draw_grid(img, grid, m3, view)
            if room_map is not None and regions:
                img = self._draw_rooms(img, room_map, regions, m3, view)
        if trajectory and len(trajectory) > 1:
            poly = np.array([view.to_px(x, y) for x, y in trajectory], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [poly], False, TRAJ_RGB, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(poly[0, 0]), 4, TRAJ_RGB, -1, cv2.LINE_AA)
        placed: list[tuple[int, int, int, int]] = []
        for rid, r in regions.items():
            px = view.to_px(r.centroid[0], r.centroid[1])
            txt = region_label(r).upper()
            (tw, th), _ = cv2.getTextSize(txt, FONT, cfg.font_scale * 1.3, 2)
            org = (int(np.clip(px[0] - tw // 2, 2, w - tw - 2)), int(np.clip(px[1] + th // 2, th + 2, h - 2)))
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
                cv2.putText(img, txt, (org[0] + dx, org[1] + dy), FONT, cfg.font_scale * 1.3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, txt, org, FONT, cfg.font_scale * 1.3, (90, 90, 110), 2, cv2.LINE_AA)
            placed.append((org[0], org[1] - th, org[0] + tw, org[1]))
        self._draw_objects(img, objs, names, captions, colors, status, view, placed, show_captions)
        if pose is not None:
            self._draw_robot(img, view, pose)
        self._draw_legend(img, view, grid, egocentric, title)
        return img

    def _draw_grid(self, img: np.ndarray, grid: GridView, m3: np.ndarray, view: _View) -> np.ndarray:
        data = grid.data
        base = np.empty(data.shape + (3,), np.uint8)
        base[:] = UNKNOWN_RGB
        base[(data >= 0) & (data < 50)] = FREE_RGB
        base[data >= 50] = OCC_RGB
        # cell (col, row) corner -> world (ox + col*res, oy + row*res); the image row axis is the grid row axis (y up)
        cell2world = np.array([[grid.resolution, 0.0, grid.origin_x], [0.0, grid.resolution, grid.origin_y], [0.0, 0.0, 1.0]])
        a = (m3 @ cell2world)[:2]
        return cv2.warpAffine(base, a, view.size, img, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_TRANSPARENT)

    def _draw_rooms(self, img: np.ndarray, room_map, regions, m3: np.ndarray, view: _View) -> np.ndarray:
        g = room_map.grid
        tint = np.zeros(room_map.labels.shape + (3,), np.uint8)
        alpha = np.zeros(room_map.labels.shape, np.uint8)
        for i, rid in enumerate(sorted(regions)):
            m = room_map.labels == rid
            tint[m] = ROOM_TINTS[i % len(ROOM_TINTS)]
            alpha[m] = 1
        cell2world = np.array([[g.resolution, 0.0, g.origin_x], [0.0, g.resolution, g.origin_y], [0.0, 0.0, 1.0]])
        a = (m3 @ cell2world)[:2]
        t = cv2.warpAffine(tint, a, view.size, flags=cv2.INTER_NEAREST)
        al = cv2.warpAffine(alpha, a, view.size, flags=cv2.INTER_NEAREST).astype(bool)
        out = img.copy()
        out[al] = (0.55 * img[al] + 0.45 * t[al]).astype(np.uint8)
        return out

    def _draw_objects(self, img, objs: list[MapObject], names, captions, colors, status, view: _View, placed,
                      show_captions: bool) -> None:
        cfg = self.cfg
        h, w = img.shape[:2]
        for o in sorted(objs, key=lambda o: -float(np.prod(np.maximum(o.size[:2], 0.1)))):   # big ones first
            col = colors[o.id]
            cx, cy = view.to_px(o.position[0], o.position[1])
            half = np.maximum(o.size[:2], 0.25) / 2.0
            corners = np.array([view.to_px(o.position[0] + sx * half[0], o.position[1] + sy * half[1])
                                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))], np.int32)
            if status[o.id] == 'tentative':
                cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, col, 1, cv2.LINE_AA)
            elif o.bearing_only:
                cv2.circle(img, (cx, cy), max(6, int(0.4 * view.scale)), col, 1, cv2.LINE_AA)
            else:
                overlay = img.copy()
                cv2.fillPoly(overlay, [corners.reshape(-1, 1, 2)], col)
                cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
                cv2.polylines(img, [corners.reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1, cv2.LINE_AA)
            if o.moving and np.linalg.norm(o.velocity[:2]) > 0.05:
                v = o.velocity[:2] * 1.5
                cv2.arrowedLine(img, (cx, cy), view.to_px(o.position[0] + v[0], o.position[1] + v[1]), (200, 40, 40), 2,
                                cv2.LINE_AA, tipLength=0.3)
            text = f'{o.id} {names[o.id]}' + ('?' if o.bearing_only else '')
            if show_captions and captions[o.id]:
                text += f': {captions[o.id][:28]}'
            (tw, th), base = cv2.getTextSize(text, FONT, cfg.font_scale, 1)
            r_px = int(max(corners[:, 0].max() - cx, corners[:, 1].max() - cy, 6))
            for dx, dy in ((0, -r_px - 4), (0, r_px + th + 6), (r_px + 4, th // 2), (-r_px - tw - 4, th // 2),
                           (r_px, -r_px - 4), (-tw, -r_px - 4), (0, -2 * r_px - 8), (0, 2 * r_px + th + 10)):
                org = (int(np.clip(cx + dx - (tw // 2 if dx == 0 else 0), 2, w - tw - 2)),
                       int(np.clip(cy + dy, th + 2, h - 2)))
                box = (org[0] - 2, org[1] - th - 2, org[0] + tw + 2, org[1] + base + 2)
                if not any(_overlaps(box, p) for p in placed):
                    placed.append(_halo_text(img, text, org, cfg.font_scale))
                    break
            else:
                placed.append(_halo_text(img, text, (int(np.clip(cx, 2, w - tw - 2)), int(np.clip(cy, th + 2, h - 2))),
                                         cfg.font_scale))

    def _draw_robot(self, img: np.ndarray, view: _View, pose: tuple[float, float, float]) -> None:
        x, y, yaw = pose
        L = max(10, int(0.35 * view.scale))
        tip = view.to_px(x + 0.35 * math.cos(yaw), y + 0.35 * math.sin(yaw))
        left = view.to_px(x + 0.2 * math.cos(yaw + 2.5), y + 0.2 * math.sin(yaw + 2.5))
        right = view.to_px(x + 0.2 * math.cos(yaw - 2.5), y + 0.2 * math.sin(yaw - 2.5))
        if max(abs(tip[0] - left[0]), abs(tip[1] - left[1])) < L // 2:      # keep it visible on a large map
            c = view.to_px(x, y)
            ang = math.atan2(-(tip[1] - c[1]), tip[0] - c[0]) if tip != c else 0.0
            tip = (int(c[0] + L * math.cos(ang)), int(c[1] - L * math.sin(ang)))
            left = (int(c[0] + 0.6 * L * math.cos(ang + 2.5)), int(c[1] - 0.6 * L * math.sin(ang + 2.5)))
            right = (int(c[0] + 0.6 * L * math.cos(ang - 2.5)), int(c[1] - 0.6 * L * math.sin(ang - 2.5)))
        tri = np.array([tip, left, right], np.int32)
        cv2.fillPoly(img, [tri], ROBOT_RGB)
        cv2.polylines(img, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)
        c = view.to_px(x, y)
        ahead = (int(c[0] + (tip[0] - c[0]) * 2.2), int(c[1] + (tip[1] - c[1]) * 2.2))
        cv2.arrowedLine(img, c, ahead, ROBOT_RGB, 2, cv2.LINE_AA, tipLength=0.25)
        _halo_text(img, 'robot', (c[0] + 8, c[1] + 18), self.cfg.font_scale, ROBOT_RGB)

    def _draw_legend(self, img: np.ndarray, view: _View, grid: GridView | None, egocentric: bool, title: str) -> None:
        h, w = img.shape[:2]
        # scale bar (1 m)
        bar = max(1.0, round(80.0 / view.scale))
        px = int(bar * view.scale)
        cv2.line(img, (10, h - 12), (10 + px, h - 12), (0, 0, 0), 3)
        _halo_text(img, f'{bar:g} m', (12, h - 18), self.cfg.font_scale)
        # north arrow
        c, s = math.cos(view.rot), math.sin(view.rot)
        nx, ny = -s, -c             # image direction of world +y after the view rotation (y is flipped on screen)
        o = (w - 30, 32)
        cv2.arrowedLine(img, o, (int(o[0] + 18 * nx), int(o[1] + 18 * ny)), (60, 60, 60), 2, cv2.LINE_AA, tipLength=0.35)
        _halo_text(img, 'N(+y)', (w - 62, 14), self.cfg.font_scale * 0.9)
        label = title or ('robot-centred, heading up' if egocentric else 'map frame, +x right, +y up')
        _halo_text(img, label, (8, 16), self.cfg.font_scale * 0.9, (60, 60, 60))

    # ------------------------------------------------------------------------------------ text variant
    def render_text(self, smap: SemanticMap, grid: GridView | None, pose: tuple[float, float, float] | None,
                    trajectory: list[tuple[float, float]] | None = None, cols: int | None = None,
                    include_tentative: bool = False) -> str:
        """Character map: '#' obstacle, '.' free, ' ' unknown, '*' trajectory, arrow = robot, letters = objects
        (see the legend below the grid). North (+y) is up."""
        cols = cols or self.cfg.ascii_cols
        with smap.lock:
            objs = smap.visible_objects(include_tentative)
            regions = dict(smap.regions)
            infos = [(o.id, smap.label(o), o.position.copy(), smap.room_name_of(o), o.bearing_only) for o in objs]
        pts = [i[2][:2] for i in infos] + ([np.array(pose[:2])] if pose else [])
        if grid is not None and grid.known_bounds():
            x0, y0, x1, y1 = grid.known_bounds()
            pts += [np.array([x0, y0]), np.array([x1, y1])]
        if not pts:
            return '(empty map)'
        arr = np.array(pts)
        lo, hi = arr.min(axis=0) - 0.5, arr.max(axis=0) + 0.5
        cell = max((hi[0] - lo[0]) / cols, 0.1, grid.resolution if grid else 0.0)
        # character cells are about twice as tall as wide: each row covers 2 cells in y
        ncol = int(math.ceil((hi[0] - lo[0]) / cell))
        nrow = int(math.ceil((hi[1] - lo[1]) / (2 * cell)))
        canvas = [[' '] * ncol for _ in range(nrow)]

        def put(x: float, y: float, ch: str, force: bool = True) -> tuple[int, int] | None:
            c, r = int((x - lo[0]) / cell), nrow - 1 - int((y - lo[1]) / (2 * cell))
            if 0 <= c < ncol and 0 <= r < nrow and (force or canvas[r][c] in ' .'):
                canvas[r][c] = ch
                return c, r
            return None
        if grid is not None:
            for r in range(nrow):
                for c in range(ncol):
                    wx0, wx1 = lo[0] + c * cell, lo[0] + (c + 1) * cell
                    wy0 = lo[1] + (nrow - 1 - r) * 2 * cell
                    wy1 = wy0 + 2 * cell
                    c0, r0 = grid.world_to_cell(wx0, wy0)
                    c1, r1 = grid.world_to_cell(wx1, wy1)
                    blk = grid.data[max(r0, 0):max(r1, 0) + 1, max(c0, 0):max(c1, 0) + 1]
                    if blk.size == 0:
                        continue
                    if (blk >= 65).any():
                        canvas[r][c] = '#'
                    elif (blk >= 0).any():
                        canvas[r][c] = '.'
        for x, y in (trajectory or [])[::2]:
            put(x, y, '*', force=False)
        legend = []
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        for k, (oid, lab, pos, room, brg) in enumerate(sorted(infos, key=lambda i: i[0])[:len(letters)]):
            ch = letters[k]
            put(pos[0], pos[1], ch)
            legend.append(f'{ch} = #{oid} {lab}{"?" if brg else ""} at ({pos[0]:.1f}, {pos[1]:.1f})' +
                          (f' in {room}' if room else ''))
        if pose is not None:
            arrows = '>^<v'
            put(pose[0], pose[1], arrows[int(round(((pose[2] % (2 * math.pi)) / (math.pi / 2)))) % 4])
        lines = [''.join(r).rstrip() for r in canvas]
        head = (f'Top-down map, north (+y) up, x from {lo[0]:.1f} to {hi[0]:.1f} m, y from {lo[1]:.1f} to {hi[1]:.1f} m; '
                f'one character = {cell:.2f} m wide x {2 * cell:.2f} m tall. # obstacle  . free  (blank) unknown  * path'
                + ('  ' + '>^<v'[int(round(((pose[2] % (2 * math.pi)) / (math.pi / 2)))) % 4] + ' = robot heading'
                   if pose else ''))
        room_txt = [f'room {region_label(r)} centre ({r.centroid[0]:.1f}, {r.centroid[1]:.1f})' for r in regions.values()]
        return '\n'.join([head, *lines, 'Legend:', *legend, *room_txt])


def to_png(img_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError('PNG encoding failed')
    return buf.tobytes()


def describe_options() -> dict[str, Any]:
    return {'mode': 'image|text', 'egocentric': 'bool', 'show_captions': 'bool', 'max_size_px': 'int',
            'format': 'png_base64 (image mode)'}
