"""Environment interface used by the runner. Implemented by RosEnv (real stack) and FakeEnv (tests, labelled fake)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.episodes import Episode
from ..core.recording import Recording, TaskResult


class Env(ABC):
    """One robot + stack under test. All times are seconds of the environment clock (sim time when simulated)."""

    name = "env"

    def check_health(self) -> None:
        """Raise InfraError if the bench is not usable (containers down, sim not publishing, services missing)."""

    @abstractmethod
    def reset_episode(self, ep: Episode, start: dict, seed: int, cfg: dict) -> None:
        """Teleport the robot to ``start`` {x,y,z,yaw}, reset semantic map and mapping state, wait until ready.
        Raise InfraError when this cannot be done."""

    @abstractmethod
    def begin(self, ep: Episode, mode: str, params: dict, trace_path: str | None) -> None:
        """Send the navigation goal, zero the task clock and start recording."""

    @abstractmethod
    def now(self) -> float:
        """Task time since ``begin`` (environment clock)."""

    @abstractmethod
    def advance(self, dt: float) -> None:
        """Let the environment run for ``dt`` seconds (real env: wait; fake env: step the simulation)."""

    @abstractmethod
    def task_result(self) -> TaskResult | None:
        """The system's result once it ended the task by itself, else None."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the running task and make the robot stop."""

    @abstractmethod
    def finish_recording(self) -> Recording:
        """Stop recording and return everything captured (including final map and GT object snapshots)."""

    def wall_limit_s(self, ep: Episode, cfg: dict) -> float | None:
        """Wall-clock guard for an episode; None disables it (fake env runs faster than real time)."""
        r = cfg["run"]
        return r["wall_limit_factor"] * ep.time_limit_s + r["wall_limit_extra_s"]

    def recover(self, cfg: dict) -> None:
        """Called after an infrastructure failure, before the retry (e.g. wait for services to come back)."""

    def world_info(self) -> dict:
        return {}

    def close(self) -> None:
        pass
