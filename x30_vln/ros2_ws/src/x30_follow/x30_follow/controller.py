"""Velocity controller that keeps a set distance behind a person (no ROS imports).

Input: the person's position and velocity in the robot's base frame (x forward, y left). Output: (vx, vy, wz) for /cmd_vel_nav, which goes
through the mux and the safety guard like every other command. The X30 manual (p.36) says people must stay at least 2 m from the robot during
autonomous operation, so `d_min` (default 2.0 m) is a hard floor: below it the robot never moves forward and backs off slowly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FollowParams:
    d_target: float = 2.5          # distance to keep [m]
    d_min: float = 2.0             # hard floor [m]: never approach below this
    kv: float = 0.6                # forward gain [1/s] on the distance error
    kw: float = 1.2                # turn gain [1/s] on the bearing
    vmax: float = 0.5              # forward speed limit [m/s] (kinematic sim 0.6, real robot trial limit 0.4: set per backend)
    vback: float = 0.15            # back-off speed limit [m/s]
    wzmax: float = 0.6             # turn rate limit [rad/s]
    deadband_m: float = 0.15       # no forward/back command inside d_target +- deadband
    ff_gain: float = 0.8           # feed-forward of the person's speed along the line of sight
    turn_first_rad: float = 0.6    # beyond this bearing the forward speed is scaled down (turn first)
    ax: float = 0.8                # acceleration limits for the slew limiter
    aw: float = 1.5


def command(px: float, py: float, pvx: float, pvy: float, p: FollowParams, prev: tuple[float, float, float] = (0.0, 0.0, 0.0),
            dt: float = 0.05) -> tuple[float, float, float]:
    d = math.hypot(px, py)
    bearing = math.atan2(py, px)
    err = d - p.d_target
    if abs(err) <= p.deadband_m:
        vx = 0.0
    else:
        vx = p.kv * (err - math.copysign(p.deadband_m, err))
    if d > 1e-6:                                   # person speed along the line of sight (positive = walking away)
        v_away = (pvx * px + pvy * py) / d
        if err > -p.deadband_m:
            vx += p.ff_gain * max(0.0, v_away)
    vx = max(-p.vback, min(p.vmax, vx))
    if d < p.d_min:                                # hard floor: no forward motion, back off in proportion to the intrusion
        vx = -min(p.vback, 0.05 + 0.5 * (p.d_min - d))
    elif d < p.d_min + p.deadband_m:
        vx = min(vx, 0.0)
    if vx > 0.0 and abs(bearing) > p.turn_first_rad:
        vx *= max(0.0, math.cos(bearing))
    wz = max(-p.wzmax, min(p.wzmax, p.kw * bearing))
    # slew limit
    vx = prev[0] + max(-p.ax * dt, min(p.ax * dt, vx - prev[0]))
    wz = prev[2] + max(-p.aw * dt, min(p.aw * dt, wz - prev[2]))
    return vx, 0.0, wz
