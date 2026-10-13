"""World lookup by id (datasets/registry.yaml) and preparation of the world SDF for the simulator.

Registry format (owned by the worlds agent):  worlds: {<id>: {sdf: <path relative to the x30_worlds share>, spawn: {x,y,z,yaw}}}
Worlds shipped with this package (share worlds/<id>.sdf: `default_arena`, `lab_room`) win over the registry.

prepare_world() writes a copy of the SDF with
  * all relative <uri> / texture paths made absolute (the copy lives elsewhere),
  * the Gazebo systems the simulator needs added when missing (physics, user commands, scene broadcaster, sensors (ogre2), contact;
    the IMU system comes with the robot model, a second instance would duplicate every IMU message),
  * a <physics> step size when the world defines none.
The original file is never modified. Custom <x30:...> elements are kept (namespace declared on <sdf> when the author forgot it).
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BUILTIN_WORLD = 'default_arena'

SYSTEMS = [
    ('gz-sim-physics-system', 'gz::sim::systems::Physics'),
    ('gz-sim-user-commands-system', 'gz::sim::systems::UserCommands'),
    ('gz-sim-scene-broadcaster-system', 'gz::sim::systems::SceneBroadcaster'),
    ('gz-sim-sensors-system', 'gz::sim::systems::Sensors'),
    ('gz-sim-contact-system', 'gz::sim::systems::Contact'),
]
_URI_TAGS = {'uri', 'albedo_map', 'normal_map', 'roughness_map', 'metalness_map', 'emissive_map', 'environment_map', 'light_map'}
X30_NS = 'http://x30.local/sdf'
ET.register_namespace('x30', X30_NS)


def parse_sdf_text(text: str) -> ET.Element:
    """Parse SDF text; a use of the x30: prefix without a namespace declaration is repaired by declaring it on <sdf>."""
    if 'x30:' in text and 'xmlns:x30' not in text:
        text = re.sub(r'<sdf', f'<sdf xmlns:x30="{X30_NS}"', text, count=1)
    return ET.fromstring(text)


def x30_child(el: ET.Element, name: str) -> ET.Element | None:
    """Child element <x30:name> (custom namespace) of el."""
    return el.find(f'{{{X30_NS}}}{name}')


@dataclass
class WorldSpec:
    world_id: str
    sdf: Path
    spawn: dict = field(default_factory=lambda: {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0})
    model_dirs: list = field(default_factory=list)


def project_root() -> Path:
    env = os.environ.get('X30_ROOT')
    if env and Path(env).is_dir():
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / 'datasets').is_dir() and (parent / 'ros2_ws').is_dir():
            return parent
    return Path('/x30')


def worlds_package_dir() -> Path | None:
    """Directory of the x30_worlds package (installed share or the source tree), None when absent."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory('x30_worlds'))
        if share.is_dir():
            return share
    except Exception:
        pass
    src = project_root() / 'ros2_ws' / 'src' / 'x30_worlds'
    return src if src.is_dir() else None


def builtin_world_path(world_id: str = BUILTIN_WORLD) -> Path:
    """SDF shipped with this package (worlds/<id>.sdf)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('x30_sim')) / 'worlds' / f'{world_id}.sdf'
    except Exception:
        return Path(__file__).resolve().parents[1] / 'worlds' / f'{world_id}.sdf'


def resolve_world(world_id: str, registry: str | os.PathLike | None = None) -> WorldSpec:
    if builtin_world_path(world_id).is_file():
        return WorldSpec(world_id, builtin_world_path(world_id), model_dirs=_model_dirs())
    reg_path = Path(registry) if registry else project_root() / 'datasets' / 'registry.yaml'
    if not reg_path.is_file():
        raise FileNotFoundError(f'world "{world_id}": registry {reg_path} not found (built-in world: {BUILTIN_WORLD})')
    reg = yaml.safe_load(reg_path.read_text(encoding='utf-8')) or {}
    worlds = reg.get('worlds', {})
    if world_id not in worlds:
        raise KeyError(f'world "{world_id}" not in {reg_path}; known: {sorted(worlds)}')
    entry = worlds[world_id]
    pkg = worlds_package_dir()
    sdf = Path(entry['sdf'])
    if not sdf.is_absolute():
        sdf = (pkg or project_root() / 'ros2_ws' / 'src' / 'x30_worlds') / sdf
    spec = WorldSpec(world_id, sdf, model_dirs=_model_dirs())
    spec.spawn.update({k: float(v) for k, v in (entry.get('spawn') or {}).items() if k in ('x', 'y', 'z', 'yaw')})
    return spec


def _model_dirs() -> list[Path]:
    dirs: list[Path] = []
    pkg = worlds_package_dir()
    if pkg is not None and (pkg / 'models').is_dir():
        dirs.append(pkg / 'models')
    ext = project_root() / 'datasets' / 'external'
    if ext.is_dir():
        dirs.extend(p for p in ext.glob('*/models') if p.is_dir())
    for p in os.environ.get('GZ_SIM_RESOURCE_PATH', '').split(os.pathsep):
        if p and Path(p).is_dir() and Path(p) not in dirs:
            dirs.append(Path(p))
    return dirs


def _absolutize(root: ET.Element, base: Path) -> None:
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        if tag in _URI_TAGS and el.text:
            txt = el.text.strip()
            if txt and '://' not in txt and not os.path.isabs(txt):
                el.text = str((base / txt).resolve())


def _has_plugin(world: ET.Element, filename: str) -> bool:
    return any(p.get('filename') == filename for p in world.findall('plugin'))


def prepare_world(spec: WorldSpec, out_dir: str | os.PathLike, physics_dt: float, rtf: float = 1.0, ambient_k: float | None = None) -> tuple[Path, str]:
    """Write the prepared copy and return (path, world name). The physics step and target real-time factor are set to the given values."""
    root = parse_sdf_text(spec.sdf.read_text(encoding='utf-8'))
    world = root.find('world')
    if world is None:
        raise ValueError(f'{spec.sdf} has no <world>')
    name = world.get('name', 'default')
    _absolutize(root, spec.sdf.parent)
    phys = world.find('physics')
    if phys is None:
        phys = ET.Element('physics', name='x30_physics', type='ignored')
        world.insert(0, phys)
    for tag, value in (('max_step_size', physics_dt), ('real_time_factor', rtf)):
        el = phys.find(tag)
        if el is None:
            el = ET.SubElement(phys, tag)
        el.text = str(value)
    if world.find('atmosphere') is None and ambient_k is not None:     # ambient of the thermal camera: uniform, no height gradient
        atm = ET.SubElement(world, 'atmosphere', type='adiabatic')
        ET.SubElement(atm, 'temperature').text = str(ambient_k)
        ET.SubElement(atm, 'temperature_gradient').text = '0.0'
    for filename, plugin_name in SYSTEMS:
        if not _has_plugin(world, filename):
            plugin = ET.SubElement(world, 'plugin', filename=filename, name=plugin_name)
            if 'sensors' in filename:
                ET.SubElement(plugin, 'render_engine').text = 'ogre2'
    if world.find('scene') is None:
        scene = ET.SubElement(world, 'scene')
        ET.SubElement(scene, 'ambient').text = '0.5 0.5 0.5 1'
        ET.SubElement(scene, 'background').text = '0.6 0.75 0.9 1'
        ET.SubElement(scene, 'grid').text = 'false'
    out = Path(out_dir) / f'x30_{spec.world_id}.sdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)
    return out, name


def resource_path_for(spec: WorldSpec) -> str:
    """GZ_SIM_RESOURCE_PATH value: the world folder, the model folders and what is already set."""
    parts = [str(spec.sdf.parent)] + [str(d) for d in spec.model_dirs] + [p for p in os.environ.get('GZ_SIM_RESOURCE_PATH', '').split(os.pathsep) if p]
    seen: list[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)
    return os.pathsep.join(seen)
