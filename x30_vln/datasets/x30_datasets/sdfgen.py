"""Minimal SDF (Gazebo Harmonic, sdf 1.9) writers: primitives, models, worlds.

Conventions used by every generated world (also documented in docs/WORLDS.md):

* every semantic object is a top-level ``<model name="<label>_<NN>">`` that is ``<static>`` and
  carries ``gz::sim::systems::Label`` with the integer id of ``config/labels.yaml``;
* heat sources additionally carry ``gz::sim::systems::Thermal`` with ``<temperature>`` in kelvin;
* walls/floors are labelled the same way (``wall`` = 1, ``floor`` = 2, ``door`` = 3, ``window`` = 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from xml.sax.saxutils import escape

COLORS: dict[str, tuple[float, float, float]] = {
    "white": (0.92, 0.92, 0.92),
    "black": (0.08, 0.08, 0.08),
    "gray": (0.5, 0.5, 0.52),
    "red": (0.78, 0.12, 0.12),
    "blue": (0.15, 0.30, 0.75),
    "green": (0.15, 0.55, 0.22),
    "yellow": (0.90, 0.78, 0.15),
    "orange": (0.93, 0.50, 0.12),
    "brown": (0.42, 0.27, 0.15),
    "beige": (0.80, 0.72, 0.55),
    "wood": (0.58, 0.40, 0.22),
    "metal": (0.65, 0.67, 0.70),
    "cream": (0.95, 0.92, 0.80),
    "pink": (0.90, 0.55, 0.65),
    "purple": (0.45, 0.25, 0.60),
}


def rgba(color) -> tuple[float, float, float, float]:
    if isinstance(color, str):
        r, g, b = COLORS[color]
        return r, g, b, 1.0
    c = tuple(color)
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)


def _f(v: float) -> str:
    s = f"{float(v):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def fmt_pose(pos=(0, 0, 0), rpy=(0, 0, 0)) -> str:
    return " ".join(_f(v) for v in (*pos, *rpy))


@dataclass
class Prim:
    """One geometric primitive of a model (in the model frame)."""

    shape: str  # box | cylinder | sphere
    dims: tuple  # box (sx,sy,sz); cylinder (radius,length); sphere (radius,)
    pos: tuple = (0.0, 0.0, 0.0)
    rpy: tuple = (0.0, 0.0, 0.0)
    color: object = "white"
    collide: bool = True
    alpha: float = 1.0
    emissive: float = 0.0
    visible: bool = True  # False: collision-only primitive

    def geometry(self) -> str:
        d = self.dims
        if self.shape == "box":
            return f"<box><size>{_f(d[0])} {_f(d[1])} {_f(d[2])}</size></box>"
        if self.shape == "cylinder":
            return f"<cylinder><radius>{_f(d[0])}</radius><length>{_f(d[1])}</length></cylinder>"
        if self.shape == "sphere":
            return f"<sphere><radius>{_f(d[0])}</radius></sphere>"
        raise ValueError(self.shape)


def material_xml(color, alpha: float = 1.0, emissive: float = 0.0) -> str:
    r, g, b, _ = rgba(color)
    em = f"<emissive>{_f(r * emissive)} {_f(g * emissive)} {_f(b * emissive)} 1</emissive>" if emissive else ""
    return (
        f"<material><ambient>{_f(r)} {_f(g)} {_f(b)} {_f(alpha)}</ambient>"
        f"<diffuse>{_f(r)} {_f(g)} {_f(b)} {_f(alpha)}</diffuse>"
        f"<specular>0.1 0.1 0.1 1</specular>{em}</material>"
    )


def model_xml(
    name: str,
    prims: list[Prim],
    pose=(0, 0, 0, 0, 0, 0),
    label_id: int | None = None,
    temperature_k: float | None = None,
    mass: float | None = None,
    comment: str | None = None,
    indent: str = "    ",
) -> str:
    """A static single-link model made of primitives."""
    i = indent
    out = []
    if comment:
        out.append(f"{i}<!-- {escape(comment)} -->")
    out.append(f'{i}<model name="{escape(name)}">')
    out.append(f"{i}  <static>true</static>")
    out.append(f"{i}  <pose>{fmt_pose(pose[:3], pose[3:])}</pose>")
    out.append(f'{i}  <link name="link">')
    for k, p in enumerate(prims):
        pp = fmt_pose(p.pos, p.rpy)
        if p.collide:
            out.append(
                f'{i}    <collision name="c{k}"><pose>{pp}</pose><geometry>{p.geometry()}</geometry></collision>'
            )
        if p.visible:
            out.append(
                f'{i}    <visual name="v{k}"><pose>{pp}</pose><geometry>{p.geometry()}</geometry>'
                f"{material_xml(p.color, p.alpha, p.emissive)}</visual>"
            )
    out.append(f"{i}  </link>")
    if label_id is not None:
        out.append(
            f'{i}  <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
            f"<label>{int(label_id)}</label></plugin>"
        )
    if temperature_k is not None:
        out.append(
            f'{i}  <plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal">'
            f"<temperature>{_f(temperature_k)}</temperature></plugin>"
        )
    out.append(f"{i}</model>")
    return "\n".join(out)


def mesh_model_xml(name: str, mesh_uri: str, pose=(0, 0, 0, 0, 0, 0), label_id: int | None = None,
                   scale: float = 1.0, indent: str = "    ") -> str:
    """Static visual-only model with a single mesh (collision is provided by a separate model)."""
    i = indent
    sc = f"<scale>{_f(scale)} {_f(scale)} {_f(scale)}</scale>"
    plugin = ""
    if label_id is not None:
        plugin = (f'{i}  <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label">'
                  f"<label>{int(label_id)}</label></plugin>\n")
    return (
        f'{i}<model name="{escape(name)}">\n{i}  <static>true</static>\n'
        f"{i}  <pose>{fmt_pose(pose[:3], pose[3:])}</pose>\n"
        f'{i}  <link name="link"><visual name="visual"><geometry><mesh><uri>{escape(mesh_uri)}</uri>{sc}</mesh>'
        f"</geometry></visual></link>\n{plugin}{i}</model>"
    )


def object_model_xml(o: dict, indent: str = "    ") -> str:
    """Ground-truth carrier of one object record (objects.json): labelled box, no contact response.

    The simulator reads class (``x30:label``), instance name and bounding box of top-level models; the physical
    obstacle is the scene collision, so this box must not collide (``collide_bitmask`` 0).
    """
    i = indent
    x0, y0, z0, x1, y1, z1 = o["bbox"]
    sx, sy, sz = max(x1 - x0, 0.02), max(y1 - y0, 0.02), max(z1 - z0, 0.02)
    name = escape(o["instance"])
    lines = [
        f'{i}<model name="{name}">',
        f"{i}  <static>true</static>",
        f"{i}  <pose>{_f((x0 + x1) / 2)} {_f((y0 + y1) / 2)} {_f((z0 + z1) / 2)} 0 0 0</pose>",
        f"{i}  <x30:label>{escape(o['label'])}</x30:label>",
        f"{i}  <x30:instance>{name}</x30:instance>",
        f'{i}  <link name="link"><collision name="box"><geometry><box><size>{_f(sx)} {_f(sy)} {_f(sz)}</size></box></geometry>'
        "<surface><contact><collide_bitmask>0x00</collide_bitmask></contact></surface></collision></link>",
        f"{i}</model>",
    ]
    return "\n".join(lines)


WORLD_PLUGINS = """\
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>"""


@dataclass
class WorldSpec:
    name: str
    bounds: tuple  # xmin, ymin, xmax, ymax of the area covered by the ground plane
    models: list = field(default_factory=list)  # model xml strings
    includes: list = field(default_factory=list)  # (name, uri, pose6) for third-party models
    ambient_k: float = 293.15
    ambient_light: float = 0.65
    sun: bool = True
    ground_color: object = (0.28, 0.28, 0.30)
    ground_friction: float = 1.0
    comment: str = ""
    extra_lights: list = field(default_factory=list)  # (name, x, y, z, intensity, range)
    ground_visible: bool = True  # False: the ground plane keeps its collision but is not drawn (scanned floors)
    sun_diffuse: float = 0.55


def world_xml(w: WorldSpec) -> str:
    x0, y0, x1, y1 = w.bounds
    sx, sy = (x1 - x0), (y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    parts = [
        '<?xml version="1.0" ?>',
        f"<!-- {escape(w.comment)} -->" if w.comment else "",
        '<sdf version="1.9" xmlns:x30="http://x30.local/sdf">',
        f'  <world name="{escape(w.name)}">',
        '    <physics name="1ms" type="ignored">',
        "      <max_step_size>0.001</max_step_size>",
        "      <real_time_factor>1.0</real_time_factor>",
        "    </physics>",
        WORLD_PLUGINS,
        f'    <atmosphere type="adiabatic"><temperature>{_f(w.ambient_k)}</temperature>'
        "<pressure>101325</pressure></atmosphere>",
        "    <scene>",
        f"      <ambient>{w.ambient_light} {w.ambient_light} {w.ambient_light} 1</ambient>",
        "      <background>0.55 0.65 0.8 1</background>",
        "      <shadows>false</shadows>",
        "      <grid>false</grid>",
        "    </scene>",
    ]
    if w.sun:
        parts += [
            '    <light type="directional" name="sun">',
            "      <cast_shadows>false</cast_shadows>",
            "      <pose>0 0 10 0 0 0</pose>",
            f"      <diffuse>{w.sun_diffuse} {w.sun_diffuse} {w.sun_diffuse} 1</diffuse>",
            "      <specular>0.1 0.1 0.1 1</specular>",
            "      <direction>-0.3 0.2 -0.9</direction>",
            "    </light>",
        ]
    for (ln, lx, ly, lz, inten, rng) in w.extra_lights:
        parts += [
            f'    <light type="point" name="{ln}">',
            "      <cast_shadows>false</cast_shadows>",
            f"      <pose>{_f(lx)} {_f(ly)} {_f(lz)} 0 0 0</pose>",
            "      <diffuse>1 0.97 0.9 1</diffuse>",
            "      <specular>0.1 0.1 0.1 1</specular>",
            f"      <attenuation><range>{_f(rng)}</range><constant>0.6</constant><linear>0.1</linear>"
            f"<quadratic>0.02</quadratic></attenuation>",
            f"      <intensity>{_f(inten)}</intensity>",
            "    </light>",
        ]
    r, g, b, _ = rgba(w.ground_color)
    parts += [
        '    <model name="ground_plane">',
        "      <static>true</static>",
        f"      <pose>{_f(cx)} {_f(cy)} 0 0 0 0</pose>",
        '      <link name="link">',
        "        <collision name=\"collision\"><geometry><plane><normal>0 0 1</normal>"
        f"<size>{_f(sx)} {_f(sy)}</size></plane></geometry>",
        f"          <surface><friction><ode><mu>{w.ground_friction}</mu><mu2>{w.ground_friction}</mu2>"
        "</ode></friction></surface></collision>",
        ('        <visual name="visual"><geometry><plane><normal>0 0 1</normal>'
         f"<size>{_f(sx)} {_f(sy)}</size></plane></geometry>"
         f"          <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material></visual>")
        if w.ground_visible else "",
        "      </link>",
        '      <plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>2</label></plugin>',
        "    </model>",
    ]
    parts += w.models
    for inc in w.includes:  # (name, uri, pose6[, extra xml such as label/thermal plugins])
        name, uri, pose = inc[:3]
        parts += [
            "    <include>",
            f"      <name>{escape(name)}</name>",
            f"      <uri>{escape(uri)}</uri>",
            f"      <pose>{fmt_pose(pose[:3], pose[3:])}</pose>",
        ]
        if len(inc) > 3 and inc[3]:
            parts.append("      " + inc[3])
        parts.append("    </include>")
    parts += ["  </world>", "</sdf>", ""]
    return "\n".join(p for p in parts if p != "")
