"""Command line: ``python -m x30_datasets list|validate|gen-world|import-scene|import-episodes|gen-episodes|convert-aws``."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from . import registry
from .episodes import io as eio
from .paths import datasets_dir, episodes_dir, external_dir, worlds_pkg_dir


def cmd_list(args: argparse.Namespace) -> int:
    reg = registry.load()
    print(f"{'world':<22}{'dataset':<18}{'objects':>8}{'area m2':>9}  licence")
    for wid, e in reg["worlds"].items():
        st = e.get("stats", {})
        print(f"{wid:<22}{e.get('source_dataset', ''):<18}{st.get('objects', '-'):>8}"
              f"{st.get('reachable_area_m2', st.get('navigable_area_m2', '-')):>9}  {e.get('license', '')[:60]}")
    print(f"\n{len(reg['episodes'])} episode sets:")
    for sid, p in reg["episodes"].items():
        path = datasets_dir() / p
        n = len(eio.read_episodes(path)) if path.exists() else "MISSING"
        print(f"  {sid:<34}{n:>5}  {p}")
    return 0


def _check_world(wid: str, e: dict, errors: list[str], gz: bool) -> None:
    root = worlds_pkg_dir()
    for key in registry.REQUIRED_WORLD_KEYS:
        if key not in e:
            errors.append(f"world {wid}: missing key '{key}'")
    sdf = root / e.get("sdf", "")
    if not sdf.exists():
        errors.append(f"world {wid}: sdf {sdf} not found")
        return
    try:
        ET.parse(sdf)
    except ET.ParseError as ex:
        errors.append(f"world {wid}: sdf is not well-formed XML ({ex})")
    for key in ("objects_file", "rooms_file", "map_file"):
        if e.get(key) and not (root / e[key]).exists():
            errors.append(f"world {wid}: {key} {e[key]} not found")
    b = e.get("bounds", [])
    if len(b) != 4 or b[0] >= b[2] or b[1] >= b[3]:
        errors.append(f"world {wid}: invalid bounds {b}")
    sp = e.get("spawn", {})
    if b and not (b[0] <= sp.get("x", 1e9) <= b[2] and b[1] <= sp.get("y", 1e9) <= b[3]):
        errors.append(f"world {wid}: spawn outside bounds")
    if gz and shutil.which("gz"):
        r = subprocess.run(["gz", "sdf", "-k", str(sdf)], capture_output=True, text=True,
                           env={"GZ_SIM_RESOURCE_PATH": str(root / "models"), **__import__("os").environ})
        if r.returncode != 0 or "Error" in r.stdout + r.stderr:
            errors.append(f"world {wid}: gz sdf -k failed: {(r.stdout + r.stderr)[:300]}")


def cmd_validate(args: argparse.Namespace) -> int:
    reg = registry.load()
    errors: list[str] = []
    for wid, e in reg["worlds"].items():
        _check_world(wid, e, errors, args.gz)
    objs_cache: dict = {}
    n_eps = 0
    scenes = set(reg["worlds"])
    for sid, rel in reg["episodes"].items():
        path = datasets_dir() / rel
        if not path.exists():
            errors.append(f"episode set {sid}: {path} missing")
            continue
        try:
            eps = eio.read_episodes(path)
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{sid}: unreadable ({ex})")
            continue
        ids = set()
        for ep in eps:
            n_eps += 1
            wid = ep.get("scene")
            if wid not in objs_cache:
                f = reg["worlds"].get(wid, {}).get("objects_file")
                objs_cache[wid] = ({o["instance"]: o for o in json.loads((worlds_pkg_dir() / f).read_text())}
                                   if f and (worlds_pkg_dir() / f).exists() else None)
            msgs = eio.schema_errors(ep) + eio.semantic_errors(ep, objs_cache[wid], scenes)
            if ep.get("id") in ids:
                msgs.append("duplicate id")
            ids.add(ep.get("id"))
            errors += [f"{sid}/{ep.get('id')}: {m}" for m in msgs]
    for e in errors[:60]:
        print("ERROR", e)
    print(f"checked {len(reg['worlds'])} worlds, {len(reg['episodes'])} episode sets, {n_eps} episodes: "
          f"{len(errors)} problem(s)")
    return 1 if errors else 0


def cmd_gen_world(args: argparse.Namespace) -> int:
    from .procgen.builder import PRESETS, generate_world
    from .viz import preview
    ids = list(PRESETS) if args.all else [args.id]
    for wid in ids:
        res = generate_world(wid, seed=args.seed)
        registry.register_world(wid, res.entry)
        preview(worlds_pkg_dir() / "worlds" / wid)
        s = res.entry["stats"]
        print(f"{wid}: {s['rooms']} rooms, {s['objects']} objects, reachable {s['reachable_area_m2']} m2")
    return 0


def cmd_import_scene(args: argparse.Namespace) -> int:
    from .adapters import habitat
    from .adapters.scene2sdf import scene2sdf
    from .viz import preview
    src = Path(args.path)
    kw = dict(voxel=args.voxel, scale=args.scale, floor="auto" if args.floor == "auto" else float(args.floor))
    if src.is_file():
        res = scene2sdf(src, args.id, up=args.up, license=args.license or "see source", source=src.name,
                        source_dataset=args.dataset if args.dataset != "auto" else "mesh-scene", **kw)
        results = [res]
    else:
        scenes = habitat.find_scenes(src, args.dataset)
        if not scenes:
            print(f"no scenes recognised under {src} (dataset={args.dataset}); see docs/WORLDS.md for layouts")
            return 2
        if args.scene:
            scenes = [s for s in scenes if s["name"] == args.scene or args.scene in s["name"]]
        elif not args.all:
            print("scenes found:", ", ".join(s["name"] for s in scenes), "\nchoose one with --scene NAME or use --all")
            return 2
        results = []
        for sc in scenes:
            wid = f"{args.id}_{sc['name']}" if (args.all or len(scenes) > 1) else args.id
            results.append(habitat.import_scene(sc, wid, license=args.license, **kw))
    for res in results:
        registry.register_world(res.world_id, res.entry)
        preview(worlds_pkg_dir() / "worlds" / res.world_id)
        print(f"{res.world_id}: {res.stats}")
    return 0


def cmd_import_episodes(args: argparse.Namespace) -> int:
    from .adapters import episodes_habitat as eh
    fn = eh.import_objectnav if args.kind == "objectnav" else eh.import_r2r
    kw = {"max_episodes": args.max} if args.max else {}
    by_world, stats = fn(Path(args.path), **({"worlds_root": worlds_pkg_dir()} if args.kind == "objectnav" else {}),
                         split_name=args.split, **kw)
    for wid, eps in by_world.items():
        name = f"{wid}_hab_{args.kind}"
        eio.write_episodes(episodes_dir() / f"{name}.yaml", eps, f"# imported from {Path(args.path).name} ({args.kind}, split {args.split})")
        registry.register_episodes(name, f"episodes/{name}.yaml")
        print(f"wrote {name}: {len(eps)} episodes")
    print("stats:", stats)
    return 0


def cmd_gen_episodes(args: argparse.Namespace) -> int:
    from .episodes.generate import GENERATORS, generate_for_world
    from .viz import preview  # noqa: F401
    reg = registry.load()
    ids = [w for w, e in reg["worlds"].items() if e.get("objects_file")] if args.all else [args.world]
    counts = {"object_nav": args.n_object, "goal_pose": args.n_pose, "velocity_follow": args.n_follow}
    counts = {k: v for k, v in counts.items() if v is not None}
    modes = args.modes or list(GENERATORS)
    for wid in ids:
        out = generate_for_world(wid, modes, seed=args.seed, counts=counts, dev_fraction=args.dev_fraction)
        sizes = {m: len(eio.read_episodes(p)) for m, p in out.items()}
        print(f"{wid}: {sizes}")
    return 0


def cmd_convert_aws(args: argparse.Namespace) -> int:
    from .adapters import aws
    for name in args.names or list(aws.WORLDS):
        res = aws.convert(name, Path(args.src) if args.src else external_dir() / "aws")
        registry.register_world(res.world_id, res.entry)
        print(f"{res.world_id}: {res.stats}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m x30_datasets", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show registered worlds and episode sets").set_defaults(fn=cmd_list)

    v = sub.add_parser("validate", help="validate the registry, world files and every episode file")
    v.add_argument("--gz", action="store_true", help="also run `gz sdf -k` on each world (needs gz on PATH)")
    v.set_defaults(fn=cmd_validate)

    g = sub.add_parser("gen-world", help="(re)generate procedural worlds and register them")
    g.add_argument("id", nargs="?", help="preset id, e.g. apt_01")
    g.add_argument("--all", action="store_true")
    g.add_argument("--seed", type=int, default=None, help="override the preset seed")
    g.set_defaults(fn=cmd_gen_world)

    i = sub.add_parser("import-scene", help="import a mesh file or a Habitat-style dataset folder as a world")
    i.add_argument("path", help="mesh file (glb/gltf/obj/ply) or dataset folder (datasets/external/<name>)")
    i.add_argument("--id", required=True, help="world id (prefix when several scenes are imported)")
    i.add_argument("--dataset", default="auto", choices=["auto", "hm3d", "mp3d", "replica", "scannet"])
    i.add_argument("--scene", help="scene name inside a dataset folder")
    i.add_argument("--all", action="store_true", help="import every scene found in the folder")
    i.add_argument("--up", default="auto", choices=["auto", "y", "z", "-y", "-z"], help="up axis (single mesh files)")
    i.add_argument("--scale", type=float, default=1.0, help="multiply source units to get metres")
    i.add_argument("--floor", default="auto", help="floor height in metres (Z-up frame) or 'auto'")
    i.add_argument("--voxel", type=float, default=0.1, help="collision voxel size in metres")
    i.add_argument("--license", default="")
    i.set_defaults(fn=cmd_import_scene)

    e = sub.add_parser("import-episodes", help="import Habitat ObjectNav / R2R(-CE) *.json.gz episodes")
    e.add_argument("path")
    e.add_argument("--kind", required=True, choices=["objectnav", "r2r"])
    e.add_argument("--split", default="val_unseen", help="official split name of the file (maps to dev/heldout)")
    e.add_argument("--max", type=int, default=0)
    e.set_defaults(fn=cmd_import_episodes)

    ge = sub.add_parser("gen-episodes", help="generate object_nav / goal_pose / velocity_follow episodes")
    ge.add_argument("world", nargs="?")
    ge.add_argument("--all", action="store_true")
    ge.add_argument("--modes", nargs="+", choices=["object_nav", "goal_pose", "velocity_follow"])
    ge.add_argument("--seed", type=int, default=0)
    ge.add_argument("--dev-fraction", type=float, default=0.6)
    ge.add_argument("--n-object", type=int)
    ge.add_argument("--n-pose", type=int)
    ge.add_argument("--n-follow", type=int)
    ge.set_defaults(fn=cmd_gen_episodes)

    a = sub.add_parser("convert-aws", help="convert cloned AWS RoboMaker worlds to Gazebo Harmonic SDF")
    a.add_argument("names", nargs="*", help="small_house hospital bookstore small_warehouse (default: all)")
    a.add_argument("--src", help="directory containing the cloned aws-robomaker-*-world repos")
    a.set_defaults(fn=cmd_convert_aws)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "gen-world" and not (args.all or args.id):
        print("gen-world needs an id or --all")
        return 2
    if args.cmd == "gen-episodes" and not (args.all or args.world):
        print("gen-episodes needs a world id or --all")
        return 2
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
