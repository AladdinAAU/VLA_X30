"""Model-free ranking of map objects for a text request such as "the red chair near the window" (no ROS imports).

The request is split into a head noun phrase (label + attributes) and relation phrases ("near the table", "in the
kitchen", "left of the sofa", "between the table and the couch"). Every candidate is scored from label match,
attribute match (colour, size, thermal, VLM attributes, caption words) and the relations to its anchor objects or
rooms. The annotator can re-rank the top candidates with the VLM; this module is also the fallback when it is down.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .attributes import DEFAULT_PALETTE
from .relations import between, pair_relations, viewer_relation
from .semantic_map import MapObject, SemanticMap
from .vocabulary import normalise

ARTICLES = {'the', 'a', 'an', 'that', 'this', 'my', 'some', 'any', 'one'}
SIZE_WORDS = {'tiny': 'tiny', 'small': 'small', 'little': 'small', 'medium': 'medium', 'large': 'large',
              'big': 'large', 'huge': 'huge', 'giant': 'huge'}
STATE_WORDS = {'open', 'closed', 'empty', 'full', 'broken', 'on', 'off', 'wooden', 'metal', 'plastic', 'glass'}
THERMAL_WORDS = {'warm', 'hot'}
MOTION_WORDS = {'moving', 'stationary'}
SUPERLATIVES = {'nearest': 'nearest', 'closest': 'nearest', 'farthest': 'farthest', 'furthest': 'farthest'}
REL_PATTERNS = [
    (r'\bin front of\b', 'in front of'), (r'\bbehind\b', 'behind'), (r'\bto the left of\b', 'left of'),
    (r'\bleft of\b', 'left of'), (r'\bto the right of\b', 'right of'), (r'\bright of\b', 'right of'),
    (r'\bon top of\b', 'on'), (r'\bunder(?:neath)?\b', 'under'), (r'\bbelow\b', 'under'),
    (r'\bbetween\b', 'between'), (r'\bnext to\b', 'near'), (r'\bbeside\b', 'near'), (r'\bclose to\b', 'near'),
    (r'\bnear(?: to)?\b', 'near'), (r'\bby\b', 'near'), (r'\bnearby\b', 'near'),
    (r'\binside\b', 'in'), (r'\bin\b', 'in'), (r'\bat\b', 'in'), (r'\bon\b', 'on'),
]
COLOURS = set(DEFAULT_PALETTE) | {'grey', 'silver', 'golden', 'gold', 'tan'}
COLOUR_ALIAS = {'grey': 'gray', 'silver': 'gray', 'golden': 'yellow', 'gold': 'yellow', 'tan': 'beige'}


@dataclass
class Phrase:
    label: str = ''
    colour: str = ''
    size: str = ''
    other_attrs: list[str] = field(default_factory=list)
    superlative: str = ''


@dataclass
class ParsedRequest:
    text: str
    head: Phrase
    relations: list[tuple[str, list[Phrase]]] = field(default_factory=list)   # (relation, anchors)

    def to_dict(self) -> dict[str, Any]:
        def ph(p: Phrase) -> dict[str, Any]:
            return {'label': p.label, 'colour': p.colour, 'size': p.size, 'attrs': p.other_attrs,
                    'superlative': p.superlative}
        return {'text': self.text, 'head': ph(self.head),
                'relations': [{'relation': r, 'anchors': [ph(a) for a in anchors]} for r, anchors in self.relations]}


def _phrase(words: list[str], known: set[str]) -> Phrase:
    p = Phrase()
    rest = []
    for w in words:
        if w in ARTICLES:
            continue
        if w in COLOURS:
            p.colour = COLOUR_ALIAS.get(w, w)
        elif w in SIZE_WORDS:
            p.size = SIZE_WORDS[w]
        elif w in THERMAL_WORDS or w in MOTION_WORDS or w in STATE_WORDS:
            p.other_attrs.append(w)
        elif w in SUPERLATIVES:
            p.superlative = SUPERLATIVES[w]
        else:
            rest.append(w)
    # longest known n-gram wins ("fire extinguisher", "potted plant"), else the whole remainder
    best = ''
    for n in range(len(rest), 0, -1):
        for i in range(0, len(rest) - n + 1):
            cand = ' '.join(rest[i:i + n])
            if cand in known or (cand.endswith('s') and cand[:-1] in known):
                best = cand[:-1] if (cand not in known and cand.endswith('s')) else cand
                break
        if best:
            break
    p.label = best or ' '.join(rest)
    return p


def parse_request(text: str, known_labels: set[str]) -> ParsedRequest:
    t = ' ' + normalise(re.sub(r'[,.;!?]', ' ', text)) + ' '
    t = t.replace(' where is ', ' ').replace(' find ', ' ').replace(' go to ', ' ')
    cuts = []
    for pat, rel in REL_PATTERNS:
        for m in re.finditer(pat, t):
            if not any(a <= m.start() < b for a, b, _ in cuts):
                cuts.append((m.start(), m.end(), rel))
    cuts.sort()
    # keep only the first match at each position range (longer patterns are listed first)
    cleaned, last_end = [], -1
    for a, b, rel in cuts:
        if a >= last_end:
            cleaned.append((a, b, rel))
            last_end = b
    head_words = t[:cleaned[0][0]].split() if cleaned else t.split()
    parsed = ParsedRequest(text=text, head=_phrase(head_words, known_labels))
    for i, (a, b, rel) in enumerate(cleaned):
        end = cleaned[i + 1][0] if i + 1 < len(cleaned) else len(t)
        seg = t[b:end].strip()
        if rel == 'between' or ' and ' in f' {seg} ':
            parts = [s for s in re.split(r'\band\b', seg) if s.strip()]
            anchors = [_phrase(s.split(), known_labels) for s in parts]
        else:
            anchors = [_phrase(seg.split(), known_labels)] if seg else []
        if anchors:
            parsed.relations.append((rel, anchors))
    return parsed


# ----------------------------------------------------------------------------------------------- scoring
def _label_score(smap: SemanticMap, o: MapObject, label: str) -> float:
    if not label:
        return 0.0
    canon = smap.vocab.canon(label)
    if smap.canon(o) == canon:
        return 3.0
    if smap.vocab.compat(smap.label(o), label) == 1:
        return 1.0
    if o.caption and re.search(r'\b' + re.escape(label) + r's?\b', o.caption.lower()):
        return 1.5
    return 0.0


def _attr_score(smap: SemanticMap, o: MapObject, p: Phrase) -> tuple[float, list[str]]:
    score, why = 0.0, []
    attrs = {a.lower() for a in smap.attribute_list(o)}
    cap = o.caption.lower()
    texts = attrs | {v.lower() for v in o.vlm_attrs.values()}
    if p.colour:
        have = (o.vlm_attrs.get('colour') or o.colour).lower()
        if have == p.colour or p.colour in cap.split():
            score += 1.5
            why.append(f'colour {p.colour}')
        elif have:
            score -= 0.7
            why.append(f'colour {have} != {p.colour}')
    if p.size:
        if p.size in attrs:
            score += 0.8
            why.append(f'size {p.size}')
    for a in p.other_attrs:
        if a in texts or a in cap.split():
            score += 1.0
            why.append(a)
    return score, why


def _anchors(smap: SemanticMap, p: Phrase, exclude: int) -> list[MapObject]:
    out = []
    for a in smap.visible_objects(True):
        if a.id == exclude:
            continue
        if _label_score(smap, a, p.label) >= 1.5 and (not p.colour or _attr_score(smap, a, p)[0] > 0):
            out.append(a)
    return out


def _room_names(smap: SemanticMap) -> set[str]:
    return {r.name.lower() for r in smap.regions.values() if r.name}


def score_candidate(smap: SemanticMap, o: MapObject, req: ParsedRequest,
                    pose: tuple[float, float, float] | None) -> tuple[float, list[str]]:
    cfg = smap.cfg.relations
    reasons: list[str] = []
    score = 0.0
    if req.head.label:
        ls = _label_score(smap, o, req.head.label)
        if ls == 0.0:
            return 0.0, []
        score += ls
        reasons.append(f'label {smap.label(o)}')
    a_score, a_why = _attr_score(smap, o, req.head)
    score += a_score
    reasons += a_why
    rooms = _room_names(smap)
    for rel, anchors in req.relations:
        if rel == 'in' and anchors and anchors[0].label in rooms:
            if smap.room_name_of(o).lower() == anchors[0].label:
                score += 2.0
                reasons.append(f'in the {anchors[0].label}')
            else:
                score -= 1.0
            continue
        if rel == 'between' and len(anchors) >= 2:
            best = 0.0
            for a in _anchors(smap, anchors[0], o.id):
                for b in _anchors(smap, anchors[1], o.id):
                    if a.id != b.id and between(o, a, b, cfg):
                        best = 2.5
                        reasons.append(f'between #{a.id} and #{b.id}')
            score += best
            continue
        best, why = 0.0, ''
        for a in _anchors(smap, anchors[0], o.id):
            dist = float(np.linalg.norm(a.position[:2] - o.position[:2]))
            rels = pair_relations(o, a, cfg)
            if rel == 'near':
                s = 2.0 if 'near' in rels else 2.0 * math.exp(-max(dist - cfg.near_m, 0.0) / 3.0) * 0.5
            elif rel in ('on', 'under'):
                s = 2.5 if rel in rels else (0.5 if 'near' in rels else 0.0)
            elif rel in ('in front of', 'behind', 'left of', 'right of'):
                s = 0.0
                if dist <= 2.5 * cfg.near_m:
                    if pose is None:
                        s = 0.5
                    elif viewer_relation(o.position, a.position, pose, cfg) == rel:
                        s = 2.0
            elif rel == 'in':
                s = 1.5 if dist <= 2.0 else 0.0
            else:
                s = 0.0
            if s > best:
                best, why = s, f'{rel} #{a.id} {smap.label(a)}'
        score += best
        if why:
            reasons.append(why)
    return score, reasons


def rank(smap: SemanticMap, text: str, pose: tuple[float, float, float] | None = None, limit: int = 10,
         include_tentative: bool = False) -> dict[str, Any]:
    """Rank objects for `text`. Result: parsed request, matches (best first) with score in 0..1 and reasons."""
    with smap.lock:
        req = parse_request(text, smap.vocab.known_labels())
        cands = []
        for o in smap.visible_objects(include_tentative):
            s, why = score_candidate(smap, o, req, pose)
            if s <= 0:
                continue
            s *= 0.6 + 0.4 * smap.confidence(o)
            cands.append((s, o, why))
        sup = req.head.superlative
        if pose is not None and sup and cands:
            dist = [np.hypot(o.position[0] - pose[0], o.position[1] - pose[1]) for _, o, _ in cands]
            order = np.argsort(dist if sup == 'nearest' else [-d for d in dist])
            cands = [(s + (1.5 if i == int(order[0]) else 0.0), o, why + ([sup] if i == int(order[0]) else []))
                     for i, (s, o, why) in enumerate(cands)]
        best = max((s for s, _, _ in cands), default=1.0)
        cands.sort(key=lambda c: (-c[0], c[1].id))
        matches = []
        for s, o, why in cands[:limit]:
            d = smap.describe(o, pose)
            d.update({'score': round(s / max(best, 1e-9), 3), 'raw_score': round(s, 3), 'reasons': why})
            matches.append(d)
        return {'parsed': req.to_dict(), 'matches': matches, 'method': 'heuristic'}
