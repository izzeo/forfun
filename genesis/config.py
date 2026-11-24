"""Configuration of a GENESIS universe."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass
class WorldConfig:
    seed: int = 1
    width: int = 96
    height: int = 96
    sea_level: float = 0.40
    # --- time: 1 simulation tick == 1 virtual hour -------------------------
    day_ticks: int = 24
    year_days: int = 60            # short orbital period: 1 year = 60 days = 1440 ticks
    moon_days: int = 20
    # --- level of detail ----------------------------------------------------
    chunk_size: int = 16
    lod_interval: int = 12         # inactive chunks are integrated in closed form every N ticks
    lod_enabled: bool = True
    # --- population ---------------------------------------------------------
    initial_population: int = 300
    # "primordial": simple first organisms (default). "advanced": an experimental
    # shortcut that starts from social, larger-brained omnivores so that later
    # stages (language, tools, settlements) can be studied in reasonable time.
    scenario: str = "primordial"
    max_population: int = 6000
    # --- bookkeeping cadence --------------------------------------------------
    snapshot_interval: int = 720
    max_snapshots: int = 40
    stats_interval: int = 24
    group_interval: int = 96
    species_interval: int = 240
    deliberation_interval: int = 12
    max_social_pairs: int = 400
    # --- world dials ----------------------------------------------------------
    plant_growth: float = 0.14
    pathogen_virulence: float = 1.0
    storm_rate: float = 1 / 400.0  # expected storms per tick
    eclipse_rate: float = 1 / 5000.0
    # --- morality / culture (see genesis.civilization.morality) ---------------
    moral_framework: str = "none"
    moral_education: float = 0.3
    # --- storage -----------------------------------------------------------
    db_path: str | None = None     # None -> in-memory SQLite
    record_events: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorldConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def year_ticks(self) -> int:
        return self.day_ticks * self.year_days
