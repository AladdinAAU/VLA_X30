"""Validate registered worlds inside a Gazebo Harmonic container (x30-sim image).

    docker run --rm --name x30-worlds-probe -v <project>:/x30 x30-sim:latest \
        python3 /x30/datasets/scripts/gz_validate.py apt_01 aws_hospital --out /x30/docs/reports/worlds-gazebo

For each world: `gz sdf -k`, then a headless software-rendered server with a probe model (RGB camera at
spawn, optional overhead camera, one gpu_lidar) is started, one frame of each camera is saved as PNG and the
real-time factor reported by the world stats topic is averaged. Uses only the standard library, PIL and numpy.
"""
import argparse
import base64
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(os.environ.get("X30_ROOT", "/x30"))
PKG = ROOT / "ros2_ws" / "src" / "x30_worlds"


def sensor_models(spawn, bounds, overview, lidar):
    x, y, yaw = spawn["x"], spawn["y"], spawn["yaw"]
    cam = lambda name, topic, w, h, fov: (  # noqa: E731
        f'<sensor name="{name}" type="camera"><always_on>1</always_on><update_rate>10</update_rate><topic>{topic}</topic>'
        f'<camera><horizontal_fov>{fov}</horizontal_fov><image><width>{w}</width><height>{h}</height></image>'
        f'<clip><near>0.05</near><far>200</far></clip></camera></sensor>')
    out = (f'<model name="probe"><static>true</static><pose>{x} {y} 0.55 0 0.05 {yaw}</pose><link name="l">'
           + cam("cam", "probe_cam", 640, 480, 1.4))
    if lidar:
        out += ('<sensor name="lidar" type="gpu_lidar"><always_on>1</always_on><update_rate>10</update_rate>'
                '<topic>probe_lidar</topic><lidar><scan><horizontal><samples>900</samples><min_angle>-3.14159</min_angle>'
                '<max_angle>3.14159</max_angle></horizontal><vertical><samples>16</samples><min_angle>-0.3</min_angle>'
                '<max_angle>0.3</max_angle></vertical></scan><range><min>0.1</min><max>60</max></range></lidar></sensor>')
    out += "</link></model>"
    if overview:
        cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
        ext = max(bounds[2] - bounds[0], (bounds[3] - bounds[1]) * 4 / 3)
        h = ext / (2 * np.tan(1.0 / 2)) * 1.05
        out += (f'<model name="probe_top"><static>true</static><pose>{cx} {cy} {h:.1f} 0 1.5708 0</pose><link name="l">'
                + cam("top", "probe_top", 800, 600, 1.0) + "</link></model>")
    return out


def run(cmd, timeout=60):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def grab(topic, path, tries=40):
    for _ in range(tries):
        try:
            rc, txt = run(["gz", "topic", "-e", "-t", topic, "-n", "1", "--json-output"], timeout=30)
        except subprocess.TimeoutExpired:
            continue
        if '"data"' in txt:
            j = json.loads(txt[txt.index("{"):])
            w, h = j["width"], j["height"]
            img = np.frombuffer(base64.b64decode(j["data"]), np.uint8).reshape(h, w, 3)
            Image.fromarray(img).save(path)
            return True
        time.sleep(1)
    return False


def rtf_samples(world_name, n=6):
    rtf = []
    for _ in range(n):
        try:
            rc, txt = run(["gz", "topic", "-e", "-t", f"/world/{world_name}/stats", "-n", "1", "--json-output"], 20)
            m = re.search(r'"realTimeFactor":\s*([0-9.eE+-]+)', txt) or re.search(r'"real_time_factor":\s*([0-9.eE+-]+)', txt)
            if m:
                rtf.append(float(m.group(1)))
        except subprocess.TimeoutExpired:
            pass
    return rtf


def validate(wid, out_dir, overview=True, lidar=True, warmup=15, sensors=True):
    reg = yaml.safe_load((ROOT / "datasets" / "registry.yaml").read_text())["worlds"][wid]
    sdf = PKG / reg["sdf"]
    res = {"world": wid}
    rc, txt = run(["gz", "sdf", "-k", str(sdf)])
    res["gz_sdf_k"] = "Valid" if rc == 0 and "Valid" in txt else txt[:300]
    text = sdf.read_text()
    name = re.search(r'<world name="([^"]+)"', text).group(1)
    probe = sdf.with_name("_probe.sdf")
    probe.write_text(text.replace("</world>", (sensor_models(reg["spawn"], reg["bounds"], overview, lidar) if sensors else "") + "</world>"))
    env = {**os.environ, "GZ_SIM_RESOURCE_PATH": f"{PKG / 'models'}:{os.environ.get('GZ_SIM_RESOURCE_PATH', '')}",
           "LIBGL_ALWAYS_SOFTWARE": "1"}
    log = open(out_dir / f"{wid}.gz.log", "w")
    t0 = time.time()
    proc = subprocess.Popen(["gz", "sim", "-s", "-r", "--headless-rendering", "-v", "3", str(probe)], env=env,
                            stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    try:
        up = False
        while time.time() - t0 < 240:
            if proc.poll() is not None:
                break
            if "/probe_cam" in run(["gz", "topic", "-l"], 20)[1] or (not sensors and f"/world/{name}/stats" in run(["gz", "topic", "-l"], 20)[1]):
                up = True
                break
            time.sleep(2)
        res["load_s"] = round(time.time() - t0, 1) if up else None
        res["loaded"] = up
        if up:
            if sensors:
                res["view_png"] = bool(grab("/probe_cam", out_dir / f"{wid}_view.png"))
            if sensors and overview:
                res["top_png"] = bool(grab("/probe_top", out_dir / f"{wid}_top.png"))
            time.sleep(warmup)
            s = rtf_samples(name)
            res["rtf"] = round(float(np.mean(s)), 3) if s else None
            res["rtf_samples"] = [round(v, 3) for v in s]
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        probe.unlink(missing_ok=True)
        log.close()
    errs = [l for l in (out_dir / f"{wid}.gz.log").read_text(errors="ignore").splitlines() if "Error" in l or "rror]" in l]
    res["log_errors"] = errs[:6]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worlds", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "docs" / "reports" / "worlds-gazebo"))
    ap.add_argument("--no-overview", action="store_true")
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-sensors", action="store_true", help="physics/scene only (measures the world itself)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for w in a.worlds:
        r = validate(w, out, not a.no_overview, not a.no_lidar, sensors=not a.no_sensors)
        print(json.dumps(r), flush=True)
        with open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
