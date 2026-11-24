"""Real-time runner: maps real seconds to virtual time.

The simulation itself only knows ticks (virtual hours). The runner decides how
many ticks to execute per real second for the chosen scale, e.g.
"minute" = 1/60 tick per second (the clock display interpolates minutes),
"day" = 24 ticks per second, "year" = 1440 ticks per second (if the machine
can keep up; otherwise it runs as fast as it can and reports the actual rate).
Because the laws are written per tick, every scale yields the same history.
"""
from __future__ import annotations

import threading
import time

from ..timekeeping import time_scales
from .simulation import Simulation


class Runner:
    def __init__(self, sim: Simulation, scale: str = "hour"):
        self.sim = sim
        self.lock = threading.RLock()
        self.paused = True
        self.scale = scale
        self.custom_rate: float | None = None
        self._acc = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.effective_rate = 0.0
        self.version = 0  # bumped whenever the world is replaced (load / rewind)

    @property
    def rate(self) -> float:
        """Requested virtual hours per real second."""
        if self.custom_rate is not None:
            return self.custom_rate
        return time_scales(self.sim.cfg)[self.scale]

    def set_scale(self, scale: str) -> None:
        if scale not in time_scales(self.sim.cfg) and scale != "max":
            raise ValueError(scale)
        self.custom_rate = 1e9 if scale == "max" else None
        self.scale = scale if scale != "max" else self.scale

    def fractional_hour(self) -> float:
        return min(self._acc, 0.999)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def step(self, n: int = 1) -> None:
        with self.lock:
            for _ in range(n):
                self.sim.step()
            self.sim.history.flush()

    def _loop(self) -> None:
        last = time.perf_counter()
        window_ticks, window_start = 0, last
        while not self._stop.is_set():
            now = time.perf_counter()
            dt = now - last
            last = now
            if self.paused:
                time.sleep(0.05)
                self._acc = 0.0
                continue
            self._acc += dt * self.rate
            budget_end = now + 0.08
            with self.lock:
                while self._acc >= 1.0 and time.perf_counter() < budget_end:
                    self.sim.step()
                    self._acc -= 1.0
                    window_ticks += 1
            if self._acc > 50:  # cannot keep up: drop the backlog instead of spiralling
                self._acc = 0.0
            if now - window_start > 1.0:
                self.effective_rate = window_ticks / (now - window_start)
                window_ticks, window_start = 0, now
            time.sleep(0.005)
