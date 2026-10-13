"""Label vocabulary: synonyms (canonical names), categories and per-category parameters."""
from __future__ import annotations

import re

from .config import CategoryCfg, SemanticMapConfig

DEFAULT_CATEGORY = 'default'


def normalise(label: str) -> str:
    return re.sub(r'\s+', ' ', label.lower().replace('_', ' ').replace('-', ' ')).strip()


class Vocabulary:
    def __init__(self, cfg: SemanticMapConfig):
        self._canon: dict[str, str] = {}
        for group in cfg.synonyms:
            names = [normalise(g) for g in group]
            for n in names:
                self._canon.setdefault(n, names[0])
        self._cat_of: dict[str, str] = {}
        self._cats: dict[str, CategoryCfg] = dict(cfg.categories)
        for cname, cat in cfg.categories.items():
            for lab in cat.labels:
                n = normalise(lab)
                self._cat_of[n] = cname
                self._cat_of.setdefault(self._canon.get(n, n), cname)
        self._default = CategoryCfg()

    def canon(self, label: str) -> str:
        n = normalise(label)
        return self._canon.get(n, n)

    def category(self, label: str) -> str:
        n = normalise(label)
        return self._cat_of.get(n) or self._cat_of.get(self._canon.get(n, n)) or DEFAULT_CATEGORY

    def cat_cfg(self, label: str) -> CategoryCfg:
        return self._cats.get(self.category(label), self._default)

    def compat(self, a: str, b: str) -> int:
        """2 = same canonical label, 1 = same (non-default) category, 0 = unrelated."""
        ca, cb = self.canon(a), self.canon(b)
        if ca == cb:
            return 2
        cata, catb = self.category(a), self.category(b)
        return 1 if (cata == catb and cata != DEFAULT_CATEGORY) else 0

    def is_dynamic(self, label: str) -> bool:
        return self.cat_cfg(label).dynamic

    def color(self, label: str) -> tuple[int, int, int]:
        c = self.cat_cfg(label).color
        return int(c[0]), int(c[1]), int(c[2])

    def synonyms_of(self, label: str) -> set[str]:
        """All surface forms that share the canonical name of `label` (including itself)."""
        canon = self.canon(label)
        out = {canon, normalise(label)}
        out.update(k for k, v in self._canon.items() if v == canon)
        return out

    def known_labels(self) -> set[str]:
        return set(self._canon) | set(self._cat_of)
