"""Analytics: population time series, trait evolution, causes of death,
behaviour distribution and the timeline of emergent "firsts"."""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..intelligence import language
from ..physics import laws as L

TRACKED_TRAITS = ["size", "speed", "sense", "brain", "diet", "sociality", "curiosity", "communication",
                  "aggression", "lifespan", "sexual", "aquatic", "parental", "learn_rate"]


class Analytics:
    def __init__(self):
        self.series: list[dict] = []
        self.firsts: dict[str, dict] = {}
        self.action_counts = np.zeros(L.N_ACTIONS, np.int64)
        self.perf = dict(ticks_per_sec=0.0, ms_per_tick=0.0)

    def first(self, sim, key: str, text: str) -> None:
        if key in self.firsts:
            return
        self.firsts[key] = dict(key=key, tick=sim.clock.tick, text=text, date=sim.clock.info()["label"])
        sim.history.event(sim, "first", -1, -1, 0, 0, dict(key=key, text=text))

    def observe_actions(self, act: np.ndarray) -> None:
        if len(act):
            self.action_counts += np.bincount(act, minlength=L.N_ACTIONS)

    def record(self, sim) -> dict:
        pop, w, tick = sim.pop, sim.world, sim.clock.tick
        n = len(pop)
        sc = sim.stats_counter
        row = dict(tick=tick, population=n, births=sc["births"], deaths=sc["deaths"],
                   deaths_by_cause={c: sc["death_" + c] for c in L.DEATH_CAUSES if sc["death_" + c]},
                   species=sum(1 for s in sim.phylo.species.values() if s["extinct"] is None),
                   plant_biomass=round(float(w.biomass.sum()), 2),
                   pathogen_load=round(float(w.pathogens.mean()), 4),
                   lod_active=round(w.lod_stats["active_fraction"], 3),
                   actions={L.ACTIONS[k]: int(v) for k, v in enumerate(self.action_counts)})
        if n:
            row["traits"] = {t: round(float(pop.genome[:, GN.T[t]].mean()), 4) for t in TRACKED_TRAITS}
            row["mean_age_days"] = round(float(pop.age.mean() / sim.cfg.day_ticks), 2)
            row["max_generation"] = int(pop.generation.max())
            row["mean_generation"] = round(float(pop.generation.mean()), 2)
            row["skills"] = {L.SKILLS[s]: round(float((pop.skills[:, s] > 0.1).mean()), 4) for s in range(L.N_SKILLS)}
            row["emotions"] = [round(float(v), 3) for v in pop.emotion.mean(axis=0)]
            row["norms"] = [round(float(v), 3) for v in pop.norms.mean(axis=0)]
            row["beliefs"] = [round(float(v), 3) for v in pop.beliefs.mean(axis=0)]
            e = np.sort(pop.energy)
            row["gini_energy"] = round(float(1 - 2 * np.sum(np.cumsum(e) / max(e.sum(), 1e-9)) / n + 1 / n), 3)
            talkers = [int(a) for a in pop.id if int(a) in pop.lexicon]
            coh, _ = language.coherence(pop, talkers[:600])
            row["language_coherence"] = round(coh, 3)
            # reproduction, kinship, disease and property
            row["sexes"] = dict(female=int((pop.sex == 0).sum()), male=int((pop.sex == 1).sum()),
                                pregnant=int((pop.pregnant_until >= 0).sum()))
            row["kinship"] = dict(mean_inbreeding=round(float(pop.inbreeding.mean()), 4),
                                  kin_aversion=round(float(pop.kin_aversion.mean()), 3))
            infected = pop.infection > 0.05
            row["disease"] = dict(infected_share=round(float(infected.mean()), 4),
                                  mean_strain_virulence=round(float(pop.strain[infected].mean()), 3) if infected.any() else None)
            wealth = pop.inventory[:, L.R["food"]].astype(np.float64).copy()
            own = w.struct_owner[w.store_food > 0]
            food = w.store_food[w.store_food > 0]
            id2i = {int(a): k for k, a in enumerate(pop.id)}
            for o, f in zip(own.tolist(), food.tolist()):
                k = id2i.get(int(o))
                if k is not None:
                    wealth[k] += f
            ws = np.sort(wealth)
            row["gini_wealth"] = round(float(1 - 2 * np.sum(np.cumsum(ws) / max(ws.sum(), 1e-9)) / n + 1 / n), 3) if ws.sum() > 0 else 0.0
            from ..intelligence import selfmodel
            row["self_model"] = dict(mean_agency=round(float(pop.agency.mean()), 3),
                                     helpless_share=round(float((pop.agency < selfmodel.HELPLESS).mean()), 4),
                                     efficacy={d: round(float(pop.self_eff[:, k].mean()), 3) for k, d in enumerate(selfmodel.DOMAINS)},
                                     **selfmodel.calibration(pop, sim))
            row["speakers"] = len(talkers)
        row.update(sim.civ.summary())
        row["social"] = dict(sim.social_stats)
        self.series.append(row)
        sim.history.stats(tick, row)
        # reset interval counters
        for k in list(sc):
            sc[k] = 0
        self.action_counts[:] = 0
        for k in sim.social_stats:
            sim.social_stats[k] = 0
        return row
