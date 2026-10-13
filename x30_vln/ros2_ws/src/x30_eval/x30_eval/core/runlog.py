"""Run directory layout: append-only episode log, infrastructure log, manifest, and the held-out guard."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

HELDOUT_LOCK = ".heldout_lock.json"


class HeldoutGuardError(RuntimeError):
    """Raised when a held-out evaluation is requested without the explicit, one-time-use permission."""


class RunDir:
    """results/<run>/ with episodes.jsonl (append-only, fsynced per line), infra_failures.jsonl, manifest.json,
    summary.json, traces/ and tracks/ (compressed per-episode recordings)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.episodes = self.path / "episodes.jsonl"
        self.infra = self.path / "infra_failures.jsonl"
        self.manifest = self.path / "manifest.json"
        self.summary = self.path / "summary.json"
        self.traces = self.path / "traces"
        self.tracks = self.path / "tracks"
        self.frames = self.path / "frames"

    def create(self) -> "RunDir":
        for d in (self.path, self.traces, self.tracks):
            d.mkdir(parents=True, exist_ok=True)
        return self

    @staticmethod
    def _append(path: Path, obj: dict) -> None:
        lead = ""
        if path.exists() and path.stat().st_size:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                lead = "" if f.read(1) == b"\n" else "\n"          # a crash left a truncated last line: start a fresh line
        with open(path, "a", encoding="utf-8") as f:
            f.write(lead + json.dumps(obj, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def append_result(self, res: dict) -> None:
        self._append(self.episodes, res)

    def append_infra(self, rec: dict) -> None:
        self._append(self.infra, rec)

    def read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue        # truncated last line after a crash: the episode is simply run again
        return out

    def results(self) -> list[dict]:
        return self.read_jsonl(self.episodes)

    def infra_failures(self) -> list[dict]:
        return self.read_jsonl(self.infra)

    def completed(self) -> set[tuple[str, int]]:
        """(episode_id, repeat) pairs with a final record (success/fail or exhausted infra retries are both final
        only when ``status`` is 'done'; exhausted infra failures are retried on resume)."""
        return {(r["episode_id"], int(r["repeat"])) for r in self.results() if r.get("status", "done") == "done"}

    def write_json(self, path: Path, obj: dict) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
        os.replace(tmp, path)

    def read_manifest(self) -> dict | None:
        return json.loads(self.manifest.read_text(encoding="utf-8")) if self.manifest.exists() else None


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


class HeldoutGuard:
    """Stops silent re-tuning on the held-out split.

    * running split ``heldout`` or ``all`` requires ``final=True`` (CLI ``--final``);
    * the first such run writes ``<results>/.heldout_lock.json``; any later run with a DIFFERENT run name needs
      ``allow_repeat=True`` (CLI ``--allow-repeat-heldout``) and is recorded as an additional evaluation, so reports
      can say "this is the n-th look at the held-out set";
    * resuming the same run name is always allowed (crash recovery);
    * runs registered with the same ``group`` (e.g. default and tuned configuration of one sweep, evaluated together
      in its final step) count as ONE held-out evaluation."""

    def __init__(self, results_root: str | Path):
        self.path = Path(results_root) / HELDOUT_LOCK

    def read(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"runs": []}

    def check_and_register(self, run_name: str, split: str, final: bool, allow_repeat: bool = False,
                           info: dict | None = None, group: str | None = None) -> None:
        if split == "dev":
            return
        if not final:
            raise HeldoutGuardError(
                f"split '{split}' includes the held-out episodes. They may only be run once, at the very end, with "
                f"--final (this writes {self.path}). Use --split dev for development and tuning.")
        lock = self.read()
        prior = [r for r in lock["runs"] if r["run"] != run_name and (group is None or r.get("group") != group)]
        if prior and not allow_repeat:
            names = ", ".join(r["run"] for r in prior)
            raise HeldoutGuardError(
                f"the held-out set was already evaluated by run(s): {names}. Another look needs "
                f"--allow-repeat-heldout and counts as an additional held-out evaluation (it is recorded). "
                f"Re-running it after changing the system is tuning on the test set.")
        if any(r["run"] == run_name for r in lock["runs"]):
            return
        same_group = [r for r in lock["runs"] if group is not None and r.get("group") == group]
        number = same_group[0]["evaluation_number"] if same_group else len({r.get("group") or r["run"] for r in lock["runs"]}) + 1
        lock["runs"].append({"run": run_name, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "split": split,
                             "evaluation_number": number, "group": group, **(info or {})})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
