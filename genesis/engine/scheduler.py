"""Event-driven part of the Simulation Engine: rare or long processes (storms,
cold snaps, droughts, eclipses) are scheduled on a priority queue instead of
being polled every tick."""
from __future__ import annotations

import heapq
import math


class Scheduler:
    def __init__(self):
        self.queue: list[tuple[int, int, str, dict]] = []
        self.seq = 0

    def at(self, tick: int, kind: str, **data) -> None:
        heapq.heappush(self.queue, (int(tick), self.seq, kind, data))
        self.seq += 1

    def after_poisson(self, now: int, rate: float, rng, kind: str, **data) -> None:
        if rate <= 0:
            return
        dt = max(1, int(math.ceil(rng.exponential(1.0 / rate))))
        self.at(now + dt, kind, **data)

    def due(self, tick: int):
        while self.queue and self.queue[0][0] <= tick:
            t, _, kind, data = heapq.heappop(self.queue)
            yield kind, data

    def upcoming(self, limit: int = 10):
        return [dict(tick=t, kind=k) for t, _, k, _ in sorted(self.queue)[:limit]]
