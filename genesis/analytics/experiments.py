"""Comparative experiments, e.g. the same world raised under different moral
frameworks. Identical seeds give identical planets and founders, so
differences are attributable to the framework (plus chaotic divergence;
use several seeds)."""
from __future__ import annotations

import numpy as np

from ..config import WorldConfig
from ..engine.simulation import Simulation


def run_world(framework: str, seed: int, ticks: int, scenario: str = "advanced", population: int = 250,
              education: float = 0.5, **kw) -> dict:
    cfg = WorldConfig(seed=seed, scenario=scenario, initial_population=population, moral_framework=framework,
                      moral_education=education, record_events=False, **kw)
    sim = Simulation(cfg)
    sim.run(ticks)
    rows = sim.analytics.series
    pops = [r["population"] for r in rows]
    deaths = {}
    for r in rows:
        for c, v in r["deaths_by_cause"].items():
            deaths[c] = deaths.get(c, 0) + v
    total_deaths = max(1, sum(deaths.values()))
    shared = sum(r["social"].get("shared", 0) for r in rows)
    trades = sum(r["social"].get("trades", 0) for r in rows)
    rituals = sum(r["social"].get("rituals", 0) for r in rows)
    person_hours = max(1, sum(pops) * sim.cfg.stats_interval)
    last = rows[-1] if rows else {}
    return dict(
        framework=framework, seed=seed, ticks=ticks,
        final_population=len(sim.pop), mean_population=float(np.mean(pops)) if pops else 0.0,
        violent_death_share=round((deaths.get("violence", 0)) / total_deaths, 4),
        starvation_share=round(deaths.get("starvation", 0) / total_deaths, 4),
        sharing_per_1000h=round(1000 * shared / person_hours, 3),
        trades=trades, rituals=rituals, gini_energy=last.get("gini_energy"),
        mean_norms=last.get("norms"), groups=last.get("groups"), settlements=last.get("settlements"),
        skills=last.get("skills"), firsts=sorted(sim.analytics.firsts),
    )


def compare_frameworks(frameworks=("none", "quran_inspired", "utilitarian", "egoist"), seeds=(1, 2), ticks=3000,
                       scenario="advanced", population=250) -> list[dict]:
    out = []
    for fw in frameworks:
        runs = [run_world(fw, s, ticks, scenario, population) for s in seeds]
        agg = dict(framework=fw, runs=runs)
        for key in ("final_population", "mean_population", "violent_death_share", "starvation_share",
                    "sharing_per_1000h", "trades", "rituals"):
            agg[key] = round(float(np.mean([r[key] for r in runs])), 4)
        out.append(agg)
    return out
