#!/usr/bin/env python3
"""Build docs/PROGRESS.html: a single self-contained page with every progress image.

Collects images from docs/figures/**, docs/reports/** and results/** (not models or reference),
groups them by area, embeds them as downscaled JPEG so the file can be opened or sent on its own.
Captions come from a CAPTIONS.md next to the images ("file.png: caption" per line) or the file name.
Rerun any time:  python scripts/make_gallery.py
"""
import base64
import html
import io
import re
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "docs" / "figures", ROOT / "docs" / "reports", ROOT / "results"]
MAX_W = 720
SKIP = re.compile(r"(^|/)(models|meshes|external|reference|test/assets|traces|episodes|sweeps|runs|fake_world|tmp)(/|$)|(^|/)(fake|ep|frame)_?\d")
MAX_PER_GROUP = 14  # evenly sampled when a folder holds more
AREA_TITLES = {
    "sim": "Simulation: robot and sensors",
    "sim_smoke": "Simulation: smoke test (kinematic and legged)",
    "sim_lidar_compare": "Simulation: lidar versus the real bag",
    "worlds-gazebo": "Worlds loaded in Gazebo",
    "navigation": "Navigation: odometry, maps, Nav2",
    "perception": "Perception",
    "semantic_map": "Semantic map",
}


def captions_for(folder: Path) -> dict:
    out = {}
    f = folder / "CAPTIONS.md"
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"\W*([\w.\-/]+\.(?:png|jpg|jpeg))\W*[:\-]\s*(.+)", line)
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def embed(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > MAX_W:
        im = im.resize((MAX_W, round(im.height * MAX_W / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    groups: dict[str, list[tuple[Path, str]]] = {}
    for src in SOURCES:
        if not src.exists():
            continue
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            rel = p.relative_to(ROOT).as_posix()
            if SKIP.search(rel) or p.stat().st_size < 3000:
                continue
            area = p.parent.name if p.parent not in SOURCES else p.stem.split("_")[0]
            if p.parent.parent.name == "results" and p.parent.parent.parent == ROOT:
                area = p.parent.name
            if area in ("frames", "kinematic", "legged", "raw", "modes", "figures"):
                area = f"{p.parent.parent.name} {area}"
            cap = captions_for(p.parent).get(p.name, p.stem.replace("_", " "))
            groups.setdefault(area, []).append((p, cap))

    parts = []
    total = 0
    for area, items in sorted(groups.items()):
        if len(items) > MAX_PER_GROUP:
            step = len(items) / MAX_PER_GROUP
            items = [items[int(i * step)] for i in range(MAX_PER_GROUP)]
        cards = []
        for p, cap in items:
            total += 1
            cards.append(
                f'<figure><img loading="lazy" src="{embed(p)}" alt="{html.escape(cap)}">'
                f"<figcaption>{html.escape(cap)}<br><span>{html.escape(p.relative_to(ROOT).as_posix())}</span></figcaption></figure>"
            )
        title = AREA_TITLES.get(area, area.replace("_", " ").replace("-", " ").title())
        parts.append(f"<h2>{html.escape(title)} <small>({len(items)})</small></h2><div class='grid'>{''.join(cards)}</div>")

    stamp = time.strftime("%Y-%m-%d %H:%M")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>X30 VLN progress</title>
<style>
:root{{--bg:#fff;--fg:#1d2430;--mut:#6b7280;--card:#f3f4f6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#12151b;--fg:#e6e9ef;--mut:#9aa3b2;--card:#1c212b}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;max-width:1200px;margin-inline:auto}}
h1{{margin:.2em 0}} h2{{margin:1.6em 0 .5em;font-size:1.15em}} small{{color:var(--mut);font-weight:400}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border-radius:8px;overflow:hidden}}
img{{width:100%;display:block}} figcaption{{padding:8px 10px;font-size:13px}} figcaption span{{color:var(--mut);font-size:11px;word-break:break-all}}
p.n{{color:var(--mut)}}
</style></head><body>
<h1>X30 VLN framework: progress images</h1>
<p class="n">Generated {stamp}. {total} images. These are outputs of the agents' own tests (synthetic or simulated, not the real robot unless the caption says the bag). Rerun <code>python scripts/make_gallery.py</code> to refresh.</p>
{''.join(parts)}
</body></html>"""
    out = ROOT / "docs" / "PROGRESS.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {total} images)")


if __name__ == "__main__":
    main()
