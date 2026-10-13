"""Configuration of the semantic map (dataclasses with defaults, overridden by configs/semantic_map.yaml).

The YAML file is strict: an unknown key is an error, so a typo cannot silently fall back to a default.
The vocabulary (categories, synonyms, room rules) lives only in the YAML file.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = 'semantic_map.yaml'
ENV_CONFIG_DIR = 'X30_CONFIG_DIR'
ENV_CONFIG_FILE = 'X30_SEMANTIC_MAP_CONFIG'


@dataclass
class AssociationCfg:
    gate_min_m: float = 0.35            # smallest association radius (small objects)
    gate_max_m: float = 1.5             # largest association radius (large objects)
    gate_size_factor: float = 0.6       # gate = factor * horizontal extent (clipped to the limits above)
    bearing_gate_m: float = 1.5         # gate for a depth detection against a triangulated (bearing) object
    iou_assoc: float = 0.30             # boxes overlapping at least this much match regardless of distance
    cross_label_iou: float = 0.50       # different-label detections this much on top of an object vote for it
    cross_label_dist_factor: float = 0.5   # ... or closer than this fraction of the gate (same category only)
    recency_tau_s: float = 5.0          # small cost bonus for recently seen candidates
    history: int = 40                   # 3D observations kept per object for the robust position/size estimate
    obs_min_dt_s: float = 0.25          # observations closer in time than this count once (frames are correlated)
    min_points: int = 5                 # lidar points under the mask needed for a 3D observation
    max_extent_m: float = 6.0           # larger than this is a segmentation failure (floor, wall) -> ignored
    position_trim: float = 2.5          # observations farther than this many MADs from the median are dropped
    size_percentile: float = 80.0       # partial views under-estimate the size, so use a high percentile


@dataclass
class ConfirmCfg:
    min_obs: int = 4                    # counted observations needed before an object is reported
    min_mean_score: float = 0.30        # mean detector score needed


@dataclass
class ForgetCfg:
    tentative_ttl_s: float = 20.0       # an unconfirmed object not seen for this long is deleted
    miss_limit: float = 8.0             # in-view-but-not-detected evidence (net of hit credit) that removes a confirmed object
    hit_credit: float = 1.0             # a detection cancels this much miss evidence
    view_margin: float = 0.8            # only objects inside this fraction of the field of view collect misses
    view_min_range_m: float = 0.6
    view_max_range_m: float = 5.0       # beyond this a miss is not evidence (detector recall falls off)
    rejected_ttl_s: float = 300.0       # how long a deleted/rejected place is remembered
    rejected_radius_m: float = 0.6
    rejected_extra_obs: int = 3         # extra observations needed to confirm an object re-appearing in such a place
    max_objects: int = 3000


@dataclass
class DynamicCfg:
    speed_max_mps: float = 1.6          # association gate grows with speed_max * dt
    gate_max_m: float = 2.5
    moving_speed_mps: float = 0.30      # faster than this over the window = "moving"
    window_s: float = 2.0
    ttl_moving_s: float = 10.0          # a moving object is dropped this long after it was last seen
    ttl_static_s: float = 120.0         # a dynamic-class object that stands still is kept longer
    merge_dist_m: float = 0.5


@dataclass
class BearingCfg:
    enabled: bool = True
    ttl_s: float = 60.0                 # rays older than this are dropped
    object_ttl_s: float = 180.0         # a triangulated object nobody re-observed is dropped after this
    max_rays: int = 500
    min_baseline_m: float = 0.4         # the two camera positions must differ by this much
    min_angle_deg: float = 6.0          # and the rays must cross at this angle
    min_range_m: float = 0.4
    max_range_m: float = 15.0
    max_skew_m: float = 0.35            # closest distance between the rays (they must agree)
    min_rays: int = 2
    sigma_rad: float = 0.02             # assumed angular noise of a bearing (perpendicular error = range * sigma)
    max_pos_std_m: float = 0.5          # a triangulation with a larger position uncertainty is rejected
    chi_max: float = 4.0                # rays must agree: chi2 per degree of freedom limit
    explain_mahalanobis: float = 3.0    # a 3D object this close (in sigmas) to a triangulated one absorbs it
    default_size_m: float = 0.3         # size assumed for a triangulated object
    assoc_ray_dist_m: float = 0.5       # a ray passing this close to a compatible object counts as a sighting of it
    support_objects: bool = True        # use rays as extra sightings of existing objects


@dataclass
class MergeCfg:
    period_s: float = 2.0
    dist_factor: float = 0.7            # merge when centres are closer than factor * larger horizontal extent ...
    dist_min_m: float = 0.30            # ... but at least this (small objects) ...
    dist_max_m: float = 0.9             # ... and at most this
    iou: float = 0.30
    doubtful_factor: float = 2.0        # pairs within this multiple of the merge distance are "doubtful" (for the VLM)


@dataclass
class ThermalCfg:
    # The stock X30 has no thermal camera: thermal attributes are off by default. When enabled, NaN values from
    # perception are still skipped, so a missing camera never produces an attribute.
    enabled: bool = False
    warm_delta_c: float = 4.0           # warm: mean above the median of all objects by this much ...
    warm_abs_c: float = 29.0            # ... or above this absolute mean
    hot_c: float = 50.0                 # hot: max temperature above this
    person_min_mean_c: float = 28.0


@dataclass
class SizeCfg:
    # size classes by the largest dimension (m): tiny < t0 <= small < t1 <= medium < t2 <= large < t3 <= huge
    thresholds_m: list[float] = field(default_factory=lambda: [0.15, 0.45, 1.0, 2.0])
    names: list[str] = field(default_factory=lambda: ['tiny', 'small', 'medium', 'large', 'huge'])


@dataclass
class ViewsCfg:
    per_object: int = 3                 # distinct views stored per object
    min_separation_deg: float = 30.0    # views must differ in viewing direction by this much
    min_crop_px: int = 24               # smaller crops are useless for a VLM
    sharpness_weight: float = 0.3       # weight of the Laplacian variance term in the view quality


@dataclass
class RoomsCfg:
    enabled: bool = True
    period_s: float = 5.0
    free_threshold: int = 25            # occupancy value below this is free
    occupied_threshold: int = 65
    door_clearance_m: float = 0.55      # free cells closer than this to an obstacle are removed to split at doors
    min_room_area_m2: float = 2.0
    corridor_max_width_m: float = 2.0
    corridor_min_aspect: float = 2.5
    assign_max_dist_m: float = 1.5      # free cells farther than this from a room core stay unassigned
    object_room_search_m: float = 0.9   # objects lie on walls/furniture: look this far for the nearest room cell
    min_score: float = 2.0              # score a rule-based room name needs
    rules: dict[str, dict[str, float]] = field(default_factory=dict)   # room type -> {label: weight}


@dataclass
class FrontierCfg:
    min_length_m: float = 0.6
    min_unknown_cells: int = 40         # unknown pockets smaller than this are not unexplored space
    max_report: int = 4


@dataclass
class RelationsCfg:
    near_m: float = 1.2                 # near: closer than this plus half the sizes
    on_tol_m: float = 0.25              # vertical gap tolerance for on-top-of / under
    on_overlap: float = 0.3             # fraction of the smaller footprint that must overlap for on/under
    between_tol_m: float = 0.5          # lateral distance to the segment for "between"
    front_back_deg: float = 45.0        # egocentric sector half-angle for front/behind
    radius_m: float = 4.0


@dataclass
class SummaryCfg:
    token_budget: int = 700
    chars_per_token: float = 3.6
    max_objects: int = 30
    caption_chars: int = 70
    include_tentative: bool = False


@dataclass
class RenderCfg:
    max_size_px: int = 800
    margin_m: float = 1.0
    ascii_cols: int = 60
    trajectory_min_step_m: float = 0.15
    trajectory_max_points: int = 4000
    show_captions: bool = False
    egocentric: bool = False
    font_scale: float = 0.45


@dataclass
class CategoryCfg:
    labels: list[str] = field(default_factory=list)
    dynamic: bool = False
    min_obs: int | None = None
    min_mean_score: float | None = None
    gate_m: float | None = None         # fixed gate (overrides the size-based one)
    min_points: int | None = None
    max_extent_m: float | None = None
    min_height_m: float | None = None   # flatter detections are floor patches / shadows
    color: list[int] = field(default_factory=lambda: [60, 60, 200])   # RGB for markers and the annotated map


@dataclass
class VlmCfg:
    enabled: bool = True
    # Endpoint, model and key come from configs/models.yaml `mapper_vlm` (X30_MAPPER_* environment overrides).
    max_tokens: int = 300               # per call (captions are short); the endpoint default is not used
    temperature: float = 0.1
    timeout_s: float = 30.0
    retries: int = 1
    response_format: str = 'json_object'     # json_object | json_schema | none
    max_image_px: int = 336
    jpeg_quality: int = 85
    # budget and rate limits
    min_interval_s: float = 1.0              # minimum time between two calls
    max_calls_per_min: int = 20
    max_calls_total: int = 2000
    max_completion_tokens_total: int = 400000
    queue_max: int = 64
    # when the endpoint is down
    fail_threshold: int = 3                  # consecutive failures before the circuit opens
    backoff_s: float = 15.0
    backoff_max_s: float = 300.0
    # what to annotate
    caption_objects: bool = True
    caption_min_obs: int = 4
    recaption_quality_gain: float = 1.6      # a view this much better than the captioned one triggers a new caption
    max_captions_per_object: int = 3
    name_rooms: bool = True
    disambiguate: bool = True
    find_timeout_s: float = 8.0
    find_recheck_top_k: int = 2
    find_max_candidates: int = 12
    room_types: list[str] = field(default_factory=lambda: [
        'kitchen', 'bedroom', 'bathroom', 'living room', 'dining room', 'office', 'corridor', 'hall',
        'lobby', 'meeting room', 'lab', 'storage room', 'stairwell', 'other'])
    cache_size: int = 512
    log_path: str = ''                       # JSONL of every call; empty = in-memory only
    rejected_label_votes: int = 2            # VLM says "not a <label>" this many times -> object is demoted


@dataclass
class MapperCfg:
    map_frame: str = 'map'
    base_frame: str = 'base_link'
    camera_frame: str = 'camera_optical_frame'
    detections_topic: str = '/perception/detections'
    map_topic: str = '/map'
    camera_info_topic: str = '/camera/color/camera_info'
    publish_period_s: float = 1.0
    markers_period_s: float = 1.0
    annotated_period_s: float = 2.0
    publish_tentative: bool = False
    use_view_misses: bool = True
    save_dir: str = ''
    autosave_period_s: float = 0.0


@dataclass
class SemanticMapConfig:
    mapper: MapperCfg = field(default_factory=MapperCfg)
    association: AssociationCfg = field(default_factory=AssociationCfg)
    confirm: ConfirmCfg = field(default_factory=ConfirmCfg)
    forget: ForgetCfg = field(default_factory=ForgetCfg)
    dynamic: DynamicCfg = field(default_factory=DynamicCfg)
    bearing: BearingCfg = field(default_factory=BearingCfg)
    merge: MergeCfg = field(default_factory=MergeCfg)
    thermal: ThermalCfg = field(default_factory=ThermalCfg)
    size: SizeCfg = field(default_factory=SizeCfg)
    views: ViewsCfg = field(default_factory=ViewsCfg)
    rooms: RoomsCfg = field(default_factory=RoomsCfg)
    frontiers: FrontierCfg = field(default_factory=FrontierCfg)
    relations: RelationsCfg = field(default_factory=RelationsCfg)
    summary: SummaryCfg = field(default_factory=SummaryCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    vlm: VlmCfg = field(default_factory=VlmCfg)
    categories: dict[str, CategoryCfg] = field(default_factory=dict)
    synonyms: list[list[str]] = field(default_factory=list)   # label groups; the first entry is canonical
    colors: dict[str, list[int]] = field(default_factory=dict)  # colour name -> RGB reference for crop colour naming

    def validate(self) -> None:
        if self.size.thresholds_m != sorted(self.size.thresholds_m):
            raise ValueError('size.thresholds_m must be ascending')
        if len(self.size.names) != len(self.size.thresholds_m) + 1:
            raise ValueError('size.names needs one more entry than size.thresholds_m')
        seen: dict[str, str] = {}
        for cname, cat in self.categories.items():
            for lab in cat.labels:
                key = lab.lower()
                if key in seen and seen[key] != cname:
                    raise ValueError(f'label {lab!r} is in categories {seen[key]!r} and {cname!r}')
                seen[key] = cname
        if self.vlm.response_format not in ('json_object', 'json_schema', 'none'):
            raise ValueError('vlm.response_format must be json_object, json_schema or none')


def _build(cls: type, data: Any, path: str) -> Any:
    """Recursively create dataclass `cls` from a dict, rejecting unknown keys."""
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected a mapping, got {type(data).__name__}')
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f'{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted(fields)}')
    kwargs = {}
    hints = _type_hints(cls)
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, f'{path}.{name}')
        elif name == 'categories':
            kwargs[name] = {k: _build(CategoryCfg, v, f'{path}.categories.{k}') for k, v in (value or {}).items()}
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _type_hints(cls: type) -> dict[str, Any]:
    import typing
    return typing.get_type_hints(cls)


def config_from_dict(data: dict[str, Any] | None) -> SemanticMapConfig:
    """Accepts the whole YAML mapping (with or without a top-level `semantic_map:` key)."""
    data = dict(data or {})
    if set(data) == {'semantic_map'}:
        data = dict(data['semantic_map'] or {})
    cfg = _build(SemanticMapConfig, data, 'semantic_map')
    cfg.validate()
    return cfg


def find_config_file(explicit: str = '') -> Path | None:
    """Search order: explicit path, $X30_SEMANTIC_MAP_CONFIG, $X30_CONFIG_DIR/semantic_map.yaml, then a
    `configs/` directory in any parent of this file (source-tree layout)."""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    env_file = os.environ.get(ENV_CONFIG_FILE, '')
    if env_file and Path(env_file).is_file():
        return Path(env_file)
    for var in (ENV_CONFIG_DIR, 'X30_CONFIGS'):                 # X30_CONFIGS is set by docker/compose.yml
        env_dir = os.environ.get(var, '')
        if env_dir and (Path(env_dir) / CONFIG_NAME).is_file():
            return Path(env_dir) / CONFIG_NAME
    for parent in Path(__file__).resolve().parents:
        cand = parent / 'configs' / CONFIG_NAME
        if cand.is_file():
            return cand
    return None


def load_config(path: str = '') -> SemanticMapConfig:
    """Load the configuration. With no file found, defaults are returned (the vocabulary will be empty)."""
    file = find_config_file(path)
    if file is None:
        if path:
            raise FileNotFoundError(f'semantic map config not found: {path}')
        return config_from_dict({})
    with open(file, encoding='utf-8') as f:
        return config_from_dict(yaml.safe_load(f) or {})
