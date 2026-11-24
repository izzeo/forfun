"""Adaptive detail for the microscopic layer.

Normally microbes exist only as an aggregate density field (``world.pathogens``)
governed by logistic growth/decay. When someone *looks* at a cell at microscopic
scale (the observer, or an inhabitant with a microscope), that cell is expanded
into individual microbes sampled consistently from the aggregate state and
simulated with the same rates. Before each look the detail is re-synchronised to
the aggregate (individuals are kept, added or removed to match), so the two
descriptions never contradict each other. Passive observation is read-only: the
world evolves identically whether or not anybody watches (tested). A causal
process that needs individual detail may call ``step(writeback=True)``.
"""
from __future__ import annotations

import numpy as np

SCALE = 400.0  # individuals per unit of aggregate density


class MicroRegion:
    def __init__(self, world, cy: int, cx: int, seed: int, tick: int):
        self.cy, self.cx = int(cy), int(cx)
        self.rng = np.random.default_rng([seed, cy, cx, tick])
        self.created = tick
        self.last_tick = tick
        n = int(round(float(world.pathogens[cy, cx]) * SCALE))
        vir = float(world.virulence[cy, cx])
        self.pos = self.rng.random((n, 2)).astype(np.float32)
        self.virulence = np.clip(self.rng.normal(vir, 0.06, n), 0, 1).astype(np.float32)
        self.kind = np.zeros(n, dtype=np.int8)  # 0 pathogen
        nd = int(round(min(1.0, float(world.detritus[cy, cx])) * SCALE * 0.5))
        self.dpos = self.rng.random((nd, 2)).astype(np.float32)  # decomposers (benign)

    @property
    def count(self) -> int:
        return len(self.pos)

    def step(self, world, ticks: int = 1, writeback: bool = False) -> None:
        for _ in range(max(0, ticks)):
            t = float(world.temperature[self.cy, self.cx])
            m = float(world.moisture[self.cy, self.cx])
            light = float(world.light[self.cy, self.cx])
            warmwet = np.clip(t / 28.0, 0, 1.2) * m
            n = len(self.pos)
            cap = SCALE
            # division (logistic, same rates as the aggregate model)
            p_div = 0.02 * world.cfg.pathogen_virulence * warmwet * max(0.0, 1 - n / cap)
            div = self.rng.random(n) < p_div
            kids = self.pos[div] + self.rng.normal(0, 0.01, (int(div.sum()), 2)).astype(np.float32)
            kv = np.clip(self.virulence[div] + self.rng.normal(0, 0.02, int(div.sum())), 0, 1).astype(np.float32)
            # death
            p_die = 0.006 + 0.004 * light
            alive = self.rng.random(n) >= p_die
            self.pos = np.concatenate([self.pos[alive], kids]).astype(np.float32)
            self.virulence = np.concatenate([self.virulence[alive], kv]).astype(np.float32)
            self.kind = np.zeros(len(self.pos), dtype=np.int8)
            # brownian motion, reflecting walls
            self.pos = np.abs(self.pos + self.rng.normal(0, 0.01, self.pos.shape).astype(np.float32))
            self.pos = np.where(self.pos > 1, 2 - self.pos, self.pos)
            if len(self.dpos):
                self.dpos = np.clip(self.dpos + self.rng.normal(0, 0.006, self.dpos.shape), 0, 1).astype(np.float32)
        if not writeback:
            return
        # write back: detailed state defines the aggregate for this cell. Used only when a
        # causal process needs individual detail; passive observation never alters the world.
        world.pathogens[self.cy, self.cx] = np.float32(min(1.0, len(self.pos) / SCALE))
        if len(self.virulence):
            world.virulence[self.cy, self.cx] = np.float32(self.virulence.mean())

    def sync_from_aggregate(self, world, tick: int) -> None:
        """Catch up lazily: advance detail by the ticks elapsed since last look."""
        dt = tick - self.last_tick
        if dt > 200:  # too long: resample from aggregate rather than integrating detail
            fresh = MicroRegion(world, self.cy, self.cx, int(self.rng.integers(1 << 30)), tick)
            self.__dict__.update(fresh.__dict__)
        else:
            # aggregate may have changed through macro processes (diffusion, shedding)
            target = int(round(float(world.pathogens[self.cy, self.cx]) * SCALE))
            n = len(self.pos)
            if target > n:
                extra = target - n
                self.pos = np.concatenate([self.pos, self.rng.random((extra, 2)).astype(np.float32)])
                self.virulence = np.concatenate([self.virulence, np.full(extra, world.virulence[self.cy, self.cx], np.float32)])
            elif target < n:
                keep = self.rng.permutation(n)[:target]
                self.pos, self.virulence = self.pos[keep], self.virulence[keep]
            self.step(world, min(dt, 24))
        self.last_tick = tick

    def to_dict(self, limit: int = 1500) -> dict:
        return dict(cell=[self.cx, self.cy], count=self.count,
                    mean_virulence=float(self.virulence.mean()) if self.count else 0.0,
                    microbes=[[round(float(a), 4), round(float(b), 4), round(float(v), 3)]
                              for (a, b), v in zip(self.pos[:limit], self.virulence[:limit])],
                    decomposers=[[round(float(a), 4), round(float(b), 4)] for a, b in self.dpos[:limit // 2]])
