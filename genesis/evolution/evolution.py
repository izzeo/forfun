"""Evolution Engine: reproduction, inheritance, mutation, speciation, phylogeny.

Nothing here decides *what* evolves. Selection is simply the differential
survival and reproduction that results from organisms interacting with the
world. This module only implements heredity (with variation) and bookkeeping.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..biology.population import make_name
from ..intelligence.brain import life_stage
from ..memory import memory as MEM
from ..physics import laws as L

A = L.A
SPECIES_THRESHOLD = 0.14


def reproduce(sim, pc, act) -> int:
    pop, rng, tick = sim.pop, sim.rng, sim.clock.tick
    n = len(pop)
    if n == 0 or n >= sim.cfg.max_population:
        return 0
    g = pop.genome
    cap = P.capacity(g)
    mature = life_stage(pop) >= 3
    ready = mature & (pop.energy > (0.3 + 0.35 * g[:, GN.T["repro_thresh"]]) * cap)
    cooldown = (tick - pop.last_repro) > 24 * (1 + 4 * g[:, GN.T["size"]] + 3 * g[:, GN.T["parental"]])
    want = (act == A["mate"]) & ready & cooldown
    if not want.any():
        return 0
    sexual_pref = g[:, GN.T["sexual"]]
    mates = pc.mate_idx
    used = np.zeros(n, bool)
    pa, pb = [], []
    for i in np.nonzero(want)[0]:
        if used[i]:
            continue
        j = int(mates[i])
        sexual = rng.random() < sexual_pref[i]
        if sexual:
            if j < 0 or used[j] or not ready[j] or not cooldown[j] or pc.mate_dist[i] > 1.6:
                continue
            # mate choice: the partner must not dislike the suitor
            aff = sim.social.affinity(int(pop.id[j]), int(pop.id[i]))
            if aff < -0.2 or rng.random() > (0.5 + 0.5 * sexual_pref[j]):
                continue
            used[i] = used[j] = True
            pa.append(i); pb.append(j)
        else:
            used[i] = True
            pa.append(i); pb.append(-1)
    if not pa:
        return 0
    if n + len(pa) > sim.cfg.max_population:
        k = sim.cfg.max_population - n
        pa, pb = pa[:k], pb[:k]
    make_offspring(sim, np.array(pa), np.array(pb))
    return len(pa)


def make_offspring(sim, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    pop, rng, tick, w = sim.pop, sim.rng, sim.clock.tick, sim.world
    m = len(pa)
    g = pop.genome
    sex = pb >= 0
    pbb = np.where(sex, pb, pa)
    # crossover (uniform) then mutation
    mask = rng.random((m, GN.G)) < 0.5
    child_g = np.where(mask & sex[:, None], g[pbb], g[pa])
    sig = (P.mutation_sigma(g[pa]) + P.mutation_sigma(g[pbb])) / 2
    child_g = child_g + rng.normal(0, 1, child_g.shape) * sig[:, None]
    # rare larger mutations
    big = rng.random(child_g.shape) < 0.002
    child_g = np.where(big, rng.random(child_g.shape), child_g)
    child_g = np.clip(child_g, 0, 1).astype(np.float32)
    wmask = rng.random((m, GN.N_W)) < 0.5
    child_w = np.where(wmask & sex[:, None], pop.weights[pbb], pop.weights[pa])
    child_w = (child_w + rng.normal(0, 1, child_w.shape) * (sig[:, None] * 2.0)).astype(np.float32)

    # energy investment (conserved): body + initial store come from the parents
    invest = 0.25 + 0.35 * (g[pa, GN.T["parental"]] + g[pbb, GN.T["parental"]]) / 2
    body = P.body(child_g) * 0.3        # newborns are small; the rest of the body grows later
    cap_c = P.capacity(child_g)
    e_child = np.minimum(invest * pop.energy[pa], cap_c * 0.6)
    total = body + e_child
    overhead = 0.1 * total
    share_a = np.where(sex, 0.5, 1.0)
    cost_a = (total + overhead) * share_a
    cost_b = (total + overhead) * (1 - share_a)
    ok = (pop.energy[pa] > cost_a + 0.05) & (np.where(sex, pop.energy[pbb] > cost_b + 0.05, True))
    pa, pb, pbb, sex = pa[ok], pb[ok], pbb[ok], sex[ok]
    child_g, child_w, e_child, cost_a, cost_b, overhead = child_g[ok], child_w[ok], e_child[ok], cost_a[ok], cost_b[ok], overhead[ok]
    m = len(pa)
    if m == 0:
        return np.zeros(0, np.int64)
    pop.energy[pa] -= cost_a
    np.subtract.at(pop.energy, pbb[sex], cost_b[sex])
    w.ledger.heat_out += float(overhead.sum())
    pop.last_repro[pa] = tick
    pop.last_repro[pbb[sex]] = tick
    np.add.at(pop.children, pa, 1)
    np.add.at(pop.children, pbb[sex], 1)

    # cultural & psychological inheritance (not genetic): norms, beliefs, personality noise
    norms = (pop.norms[pa] + pop.norms[pbb]) / 2 + rng.normal(0, 0.03, (m, 7))
    beliefs = 0.5 * (pop.beliefs[pa] + pop.beliefs[pbb]) / 2
    pers = np.stack([child_g[:, GN.T["curiosity"]], child_g[:, GN.T["patience"]], child_g[:, GN.T["sociality"]],
                     1 - child_g[:, GN.T["aggression"]], child_g[:, GN.T["fear_sens"]]], axis=1)
    pers = np.clip(pers + rng.normal(0, 0.12, pers.shape), 0, 1)
    fv = child_g[:, GN.FOOD_PREF0:GN.FOOD_PREF0 + L.N_PLANTS] * 1.0
    gen = np.maximum(pop.generation[pa], pop.generation[pbb]) + 1
    ids = pop.add(
        m,
        x=np.clip(pop.x[pa] + rng.normal(0, 0.3, m), 0, w.W - 1e-3),
        y=np.clip(pop.y[pa] + rng.normal(0, 0.3, m), 0, w.H - 1e-3),
        heading=rng.uniform(0, 2 * np.pi, m), energy=e_child, health=1.0, hydration=1.0,
        max_age=P.max_age(child_g) * rng.uniform(0.85, 1.15, m), generation=gen,
        parent_a=pop.id[pa], parent_b=np.where(sex, pop.id[pbb], -1), birth_tick=tick,
        species=pop.species[pa], group=pop.group[pa], genome=child_g, weights=child_w, growth=0.3,
        food_value=fv, personality=pers, norms=np.clip(norms, -1, 1), beliefs=beliefs,
        immunity=0.5 * (pop.immunity[pa] + pop.immunity[pbb]) * 0.3,
        hue_fear=0.3 * pop.hue_fear[pa],  # weak innate-like transmission via early exposure
        # upbringing: confident parents raise somewhat more confident young, and vice versa
        agency=np.clip(0.6 + 0.3 * ((pop.agency[pa] + pop.agency[pbb]) / 2 - 0.6), 0, 1),
    )
    sim.moral.educate_newborn(sim, ids, pa)
    n0 = len(pop) - m
    new_idx = np.arange(n0, n0 + m)
    MEM.record(pop, new_idx, MEM.M["born"], 1.0, pop.id[pa], tick)
    MEM.consolidate(pop, new_idx, MEM.M["born"], 1.0, pop.id[pa], tick, detail="was born", force=True)
    MEM.record(pop, pa, MEM.M["gave_birth"], 1.0, ids, tick)
    MEM.consolidate(pop, pa, MEM.M["gave_birth"], 0.8, ids, tick, detail="had a child")
    for k in range(m):
        cid = int(ids[k])
        pop.names[cid] = make_name(cid)
        # parents and child are bonded from the start
        a = int(pop.id[pa[k]])
        sim.social.bond(a, cid, 0.6)
        if sex[k]:
            b = int(pop.id[pbb[k]])
            sim.social.bond(b, cid, 0.6)
            sim.social.bond(a, b, 0.3)
            pop.partner[pa[k]] = b
            pop.partner[pbb[k]] = a
    sim.history.births(sim, new_idx)
    sim.stats_counter["births"] += m
    if sex.any():
        sim.analytics.first(sim, "sexual_reproduction", "Two organisms produced offspring together")
    return ids


class Phylogeny:
    """Species detected by genetic clustering; lineage of species is recorded."""

    def __init__(self):
        self.species: dict[int, dict] = {}
        self.next_id = 1

    def found_initial(self, sim):
        pop = sim.pop
        self.update(sim, force_initial=True)

    def update(self, sim, force_initial: bool = False):
        pop, tick = sim.pop, sim.clock.tick
        n = len(pop)
        if n == 0:
            for s in self.species.values():
                if s["extinct"] is None:
                    s["extinct"] = tick
            return
        traits = pop.genome[:, :GN.N_TRAITS]
        alive_ids = [sid for sid, s in self.species.items() if s["extinct"] is None]
        centroids = np.array([self.species[s]["centroid"] for s in alive_ids], np.float32) if alive_ids else np.zeros((0, GN.N_TRAITS), np.float32)
        assign = np.full(n, -1, np.int64)
        if len(centroids):
            d = np.sqrt(((traits[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2))
            best = d.argmin(axis=1)
            ok = d[np.arange(n), best] < SPECIES_THRESHOLD
            # prefer staying in own (current) species if still close
            cur = pop.species
            cur_pos = {s: k for k, s in enumerate(alive_ids)}
            cur_k = np.array([cur_pos.get(int(s), -1) for s in cur])
            has_cur = cur_k >= 0
            dc = np.where(has_cur, d[np.arange(n), np.maximum(cur_k, 0)], np.inf)
            stay = dc < SPECIES_THRESHOLD
            assign[ok] = np.array(alive_ids)[best[ok]]
            assign[stay] = cur[stay]
        # leader clustering for unassigned organisms -> new species
        rest = np.nonzero(assign < 0)[0]
        while len(rest):
            leader = rest[0]
            d = np.sqrt(((traits[rest] - traits[leader]) ** 2).mean(axis=1))
            members = rest[d < SPECIES_THRESHOLD]
            sid = self.next_id; self.next_id += 1
            parent = None if force_initial else int(pop.species[leader])
            self.species[sid] = dict(id=sid, parent=parent if parent in self.species else None, origin=tick, extinct=None,
                                     centroid=traits[members].mean(axis=0).tolist(), name=_species_name(sid),
                                     peak=0, count=0)
            assign[members] = sid
            rest = rest[d >= SPECIES_THRESHOLD]
            if not force_initial:
                sim.history.event(sim, "speciation", int(pop.id[leader]), -1, pop.x[leader], pop.y[leader],
                                  dict(species=sid, parent=parent, name=self.species[sid]["name"]))
                sim.analytics.first(sim, "speciation", f"New species {self.species[sid]['name']} diverged")
        pop.species = assign.astype(np.int32)
        counts = np.bincount(assign, minlength=self.next_id)
        for sid, s in self.species.items():
            c = int(counts[sid]) if sid < len(counts) else 0
            s["count"] = c
            s["peak"] = max(s["peak"], c)
            if c > 0:
                s["centroid"] = traits[assign == sid].mean(axis=0).tolist()
            elif s["extinct"] is None:
                s["extinct"] = tick
                sim.history.event(sim, "extinction", -1, -1, 0, 0, dict(species=sid, name=s["name"]))

    def tree(self) -> list[dict]:
        out = []
        for s in self.species.values():
            c = dict(s)
            cen = c.pop("centroid")
            c["traits"] = {t: round(cen[k], 3) for k, t in enumerate(GN.TRAITS)}
            out.append(c)
        return out


_GEN = ["Proto", "Aqua", "Terra", "Lumi", "Vora", "Sylva", "Cryo", "Thermo", "Noct", "Velo", "Magna", "Cogni"]
_EP = ["simplex", "fluvialis", "ferox", "socialis", "minor", "major", "sapiens", "curiosa", "agilis", "robusta", "pallida", "rubra"]


def _species_name(sid: int) -> str:
    x = (sid * 2654435761) & 0xFFFFFFFF
    return f"{_GEN[x % len(_GEN)]} {_EP[(x // 13) % len(_EP)]}-{sid}"
