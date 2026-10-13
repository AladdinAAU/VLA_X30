"""Download models from Gazebo Fuel (OpenRobotics collection by default) and record their licences.

    python datasets/scripts/fetch_fuel.py --dest datasets/external/fuel/models AdjTable Chair ...

Each model lands in <dest>/<Name>/ (unzipped, gz-sim readable) next to <dest>/<Name>/LICENSE_FUEL.json
with the licence name/url reported by the Fuel API. Nothing is accepted on the user's behalf: licences
are reported, the user decides whether to keep the models.
"""
import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

API = "https://fuel.gazebosim.org/1.0"


def fetch(owner: str, name: str, dest: Path) -> dict:
    meta = json.load(urllib.request.urlopen(f"{API}/{owner}/models/{name}", timeout=60))
    data = urllib.request.urlopen(f"{API}/{owner}/models/{name}.zip", timeout=300).read()
    out = dest / name
    out.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)
    info = {k: meta.get(k) for k in ("name", "owner", "license_name", "license_url", "version", "modify_date")}
    info["source"] = f"{API}/{owner}/models/{name}"
    (out / "LICENSE_FUEL.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--owner", default="OpenRobotics")
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    dest = Path(a.dest)
    bad = 0
    for n in a.names:
        if (dest / n / "LICENSE_FUEL.json").exists():
            continue
        try:
            i = fetch(a.owner, n, dest)
            print(f"{n}: {i['license_name']}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"{n}: FAILED {e}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
