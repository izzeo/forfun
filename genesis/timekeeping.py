"""Virtual time.

One simulation tick is one virtual hour. Every process in the universe is
expressed in ticks, so all acceleration settings remain mutually consistent:
changing the time scale only changes how many ticks are executed per real
second, never the laws themselves.
"""
from __future__ import annotations

import math

from .config import WorldConfig

TICK_HOURS = 1.0


def time_scales(cfg: WorldConfig) -> dict[str, float]:
    """Virtual hours per real second for each named scale."""
    return {
        "minute": 1.0 / 60.0,      # 1 s = 1 virtual minute
        "hour": 1.0,               # 1 s = 1 virtual hour
        "day": float(cfg.day_ticks),
        "week": float(cfg.day_ticks * 7),
        "year": float(cfg.year_ticks),
    }


class Clock:
    def __init__(self, cfg: WorldConfig, tick: int = 0):
        self.cfg = cfg
        self.tick = tick

    def advance(self) -> None:
        self.tick += 1

    # -- derived quantities -------------------------------------------------
    @property
    def hour(self) -> int:
        return self.tick % self.cfg.day_ticks

    @property
    def day(self) -> int:
        return self.tick // self.cfg.day_ticks

    @property
    def year(self) -> int:
        return self.tick // self.cfg.year_ticks

    @property
    def day_fraction(self) -> float:
        return (self.tick % self.cfg.day_ticks) / self.cfg.day_ticks

    @property
    def year_fraction(self) -> float:
        return (self.tick % self.cfg.year_ticks) / self.cfg.year_ticks

    @property
    def season(self) -> str:
        return ["spring", "summer", "autumn", "winter"][int(self.year_fraction * 4) % 4]

    @property
    def moon_phase(self) -> float:
        period = self.cfg.moon_days * self.cfg.day_ticks
        return (self.tick % period) / period

    def sun_elevation(self) -> float:
        """-1..1, positive during day (noon at hour 12)."""
        return -math.cos(2 * math.pi * self.day_fraction)

    def is_night(self) -> bool:
        return self.sun_elevation() <= 0.0

    def info(self) -> dict:
        return dict(tick=self.tick, hour=self.hour, day=self.day % self.cfg.year_days,
                    year=self.year, season=self.season, night=self.is_night(),
                    moon_phase=round(self.moon_phase, 3),
                    label=f"Year {self.year}, day {self.day % self.cfg.year_days + 1}, {self.hour:02d}:00")
