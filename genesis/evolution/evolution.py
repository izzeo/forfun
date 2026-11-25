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


def gestation_hours(g_row) -> float:
    """Larger bodies and bigger brains take longer to grow in the womb."""
    return 24.0 * (2.0 + 6.0 * float(g_row[GN.T["size"]]) + 5.0 * float(g_row[GN.T["brain"]]))


def relatedness(pop, i: int, j: int) -> float:
    """Coefficient of relatedness from the recorded pedigree (parents and grandparents)."""
    ii, jj = int(pop.id[i]), int(pop.id[j])
    pi = {int(pop.parent_a[i]), int(pop.parent_b[i])} - {-1}
    pj = {int(pop.parent_a[j]), int(pop.parent_b[j])} - {-1}
    if jj in pi or ii in pj:
        return 0.5                                   # parent and child
    shared_parents = len(pi & pj)
    if shared_parents >= 2:
        return 0.5                                   # full siblings
    gi = set(int(x) for x in pop.grand[i]) - {-1}
    gj = set(int(x) for x in pop.grand[j]) - {-1}
    if jj in gi or ii in gj:
        return 0.25                                  # grandparent
    if shared_parents == 1:
        return 0.25                                  # half siblings
    if (pi & gj) or (pj & gi):
        return 0.25                                  # aunt/uncle and niece/nephew
    shared_g = len(gi & gj)
    if shared_g:
        return 0.0625 * shared_g                     # cousins (first cousins share two grandparents)
    return 0.0


def mate_appeal(pop, male: int, female: int) -> float:
    """Female choice: a healthy, respected male that looks like her own kind is preferred."""
    g = pop.genome
    q = float(pop.health[male]) * (0.6 + 0.4 * min(1.0, float(pop.prestige[male])))
    looks = 1.0 - 0.8 * abs(float(g[male, GN.T["hue"]]) - float(g[female, GN.T["hue"]]))
    return float(np.clip(q * looks, 0.0, 1.0))


def reproduce(sim, pc, act) -> int:
    """Mating (conception -> pregnancy) or, for primitive organisms, budding."""
    pop, rng, tick = sim.pop, sim.rng, sim.clock.tick
    n = len(pop)
    if n == 0 or n >= sim.cfg.max_population:
        return 0
    g = pop.genome
    cap = P.capacity(g)
    mature = life_stage(pop) >= 3
    ready = mature & (pop.energy > (0.3 + 0.35 * g[:, GN.T["repro_thresh"]]) * cap) & (pop.pregnant_until < 0)
    cooldown = (tick - pop.last_repro) > 24 * (1 + 4 * g[:, GN.T["size"]] + 3 * g[:, GN.T["parental"]])
    want = (act == A["mate"]) & ready & cooldown
    if not want.any():
        return 0
    sexual_pref = g[:, GN.T["sexual"]]
    mates = pc.mate_idx
    used = np.zeros(n, bool)
    buds = []
    conceived = 0
    for i in np.nonzero(want)[0]:
        if used[i]:
            continue
        if rng.random() < sexual_pref[i]:
            j = int(mates[i])
            if j < 0 or used[j] or not ready[j] or not cooldown[j] or pc.mate_dist[i] > 1.6 or pop.sex[i] == pop.sex[j]:
                continue
            if sim.social.affinity(int(pop.id[j]), int(pop.id[i])) < -0.2:
                continue
            r = relatedness(pop, i, j)
            if r > 0 and rng.random() < max(pop.kin_aversion[i], pop.kin_aversion[j]):
                continue                             # instinctive reluctance towards relatives
            female, male = (i, j) if pop.sex[i] == 0 else (j, i)
            if rng.random() > mate_appeal(pop, male, female) * (0.5 + 0.5 * sexual_pref[female]):
                continue                             # she declines
            used[i] = used[j] = True
            conceive(sim, female, male, r)
            conceived += 1
        else:
            used[i] = True
            buds.append(i)
    if buds:
        if n + len(buds) > sim.cfg.max_population:
            buds = buds[: max(0, sim.cfg.max_population - n)]
        if buds:
            make_offspring(sim, np.array(buds), np.full(len(buds), -1))
    return len(buds) + conceived


def conceive(sim, female: int, male: int, r: float) -> None:
    pop, tick = sim.pop, sim.clock.tick
    fid, mid = int(pop.id[female]), int(pop.id[male])
    pop.pregnant_until[female] = tick + int(gestation_hours(pop.genome[female]))
    pop.last_repro[female] = tick
    pop.last_repro[male] = tick
    pop.pending[fid] = dict(father=mid, g=pop.genome[male].copy(), w=pop.weights[male].copy(),
                            gen=int(pop.generation[male]), r=float(r), kin=float(pop.kin_aversion[male]),
                            parents=(int(pop.parent_a[male]), int(pop.parent_b[male])), conceived=tick)
    MEM.record(pop, np.array([female, male]), MEM.M["mated"], 0.6, np.array([mid, fid]), tick)
    sim.social.bond(fid, mid, 0.25)
    pop.partner[female] = mid
    pop.partner[male] = fid
    sim.stats_counter["conceptions"] += 1
    sim.analytics.first(sim, "pregnancy", "A female carried offspring through pregnancy")


def deliver(sim) -> int:
    """Births at the end of pregnancy; miscarriage if the mother is starving or very ill."""
    pop, tick = sim.pop, sim.clock.tick
    preg = np.nonzero(pop.pregnant_until >= 0)[0]
    if len(preg) == 0:
        return 0
    cap = P.capacity(pop.genome[preg])
    lost = preg[(pop.energy[preg] < 0.08 * cap) | (pop.health[preg] < 0.15)]
    for k in lost:
        pop.pending.pop(int(pop.id[k]), None)
        pop.pregnant_until[k] = -1
        pop.emotion[k, 3] = min(1.0, pop.emotion[k, 3] + 0.6)
        MEM.consolidate(pop, np.array([k]), MEM.M["lost_kin"], -0.9, -1, tick, detail="lost an unborn child", force=True)
        sim.stats_counter["miscarriages"] += 1
        sim.history.event(sim, "miscarriage", int(pop.id[k]), -1, pop.x[k], pop.y[k], {})
    due = [k for k in preg if pop.pregnant_until[k] >= 0 and pop.pregnant_until[k] <= tick and int(pop.id[k]) in pop.pending]
    if not due or len(pop) >= sim.cfg.max_population:
        for k in due:
            pop.pregnant_until[k] = tick + 24        # labour delayed
        return 0
    moms = np.array(due)
    info = [pop.pending.pop(int(pop.id[k])) for k in due]
    pop.pregnant_until[moms] = -1
    id2i = {int(a): k for k, a in enumerate(pop.id)}
    fathers = np.array([id2i.get(d["father"], -1) for d in info])
    ids = make_offspring(sim, moms, fathers, father_g=np.stack([d["g"] for d in info]),
                         father_w=np.stack([d["w"] for d in info]), father_info=info)
    return len(ids)


def make_offspring(sim, pa: np.ndarray, pb: np.ndarray, father_g=None, father_w=None, father_info=None) -> np.ndarray:
    """Create offspring of mothers ``pa``. ``father_g`` given -> sexual (father may be dead: pb == -1);
    otherwise asexual budding."""
    pop, rng, tick, w = sim.pop, sim.rng, sim.clock.tick, sim.world
    m = len(pa)
    g = pop.genome
    sexual = np.full(m, father_g is not None)
    fg = father_g if father_g is not None else g[pa]
    fw = father_w if father_w is not None else pop.weights[pa]
    pb = np.asarray(pb)
    alive_f = pb >= 0
    pbb = np.where(alive_f, pb, pa)                  # for cultural traits: living father, else the mother
    # crossover (uniform) then mutation
    mask = rng.random((m, GN.G)) < 0.5
    child_g = np.where(mask & sexual[:, None], fg, g[pa])
    sig = (P.mutation_sigma(g[pa]) + P.mutation_sigma(fg)) / 2
    child_g = child_g + rng.normal(0, 1, child_g.shape) * sig[:, None]
    big = rng.random(child_g.shape) < 0.002          # rare larger mutations
    child_g = np.where(big, rng.random(child_g.shape), child_g)
    child_g = np.clip(child_g, 0, 1).astype(np.float32)
    wmask = rng.random((m, GN.N_W)) < 0.5
    child_w = np.where(wmask & sexual[:, None], fw, pop.weights[pa])
    child_w = (child_w + rng.normal(0, 1, child_w.shape) * (sig[:, None] * 2.0)).astype(np.float32)

    # energy investment (conserved): the mother pays for the newborn's body and first energy
    invest = 0.25 + 0.35 * (g[pa, GN.T["parental"]] + fg[:, GN.T["parental"]]) / 2
    body = P.body(child_g) * 0.3                     # newborns are small; the rest of the body grows later
    cap_c = P.capacity(child_g)
    e_child = np.minimum(invest * pop.energy[pa], cap_c * 0.6)
    total = body + e_child
    overhead = 0.1 * total
    cost = total + overhead
    ok = pop.energy[pa] > cost + 0.05
    if not ok.all() and father_info is not None:
        # too weak to give birth now: the pregnancy fails rather than conjuring energy
        sim.stats_counter["miscarriages"] += int((~ok).sum())
    pa, pb, pbb, sexual, alive_f = pa[ok], pb[ok], pbb[ok], sexual[ok], alive_f[ok]
    fg = fg[ok]
    child_g, child_w, e_child, cost, overhead = child_g[ok], child_w[ok], e_child[ok], cost[ok], overhead[ok]
    if father_info is not None:
        father_info = [d for d, o in zip(father_info, ok) if o]
    m = len(pa)
    if m == 0:
        return np.zeros(0, np.int64)
    pop.energy[pa] -= cost
    w.ledger.heat_out += float(overhead.sum())
    pop.last_repro[pa] = tick
    np.add.at(pop.children, pa, 1)
    np.add.at(pop.children, pbb[alive_f], 1)

    # kinship bookkeeping and inbreeding
    if father_info is not None:
        r = np.array([d["r"] for d in father_info], np.float32)
        f_kin = np.array([d["kin"] for d in father_info], np.float32)
        f_gen = np.array([d["gen"] for d in father_info])
        f_par = np.array([d["parents"] for d in father_info], np.int64)
        f_ids = np.array([d["father"] for d in father_info], np.int64)
    else:
        r = np.zeros(m, np.float32)
        f_kin = pop.kin_aversion[pa]
        f_gen = pop.generation[pa]
        f_par = np.stack([pop.parent_a[pa], pop.parent_b[pa]], axis=1)
        f_ids = np.full(m, -1, np.int64)
    F = r / 2
    grand = np.concatenate([np.stack([pop.parent_a[pa], pop.parent_b[pa]], axis=1), f_par], axis=1)
    kin_av = np.clip((pop.kin_aversion[pa] + f_kin) / 2 + rng.normal(0, 0.05, m), 0, 1)

    # cultural & psychological inheritance (not genetic): norms, beliefs, personality noise
    norms = (pop.norms[pa] + pop.norms[pbb]) / 2 + rng.normal(0, 0.03, (m, 7))
    beliefs = 0.5 * (pop.beliefs[pa] + pop.beliefs[pbb]) / 2
    pers = np.stack([child_g[:, GN.T["curiosity"]], child_g[:, GN.T["patience"]], child_g[:, GN.T["sociality"]],
                     1 - child_g[:, GN.T["aggression"]], child_g[:, GN.T["fear_sens"]]], axis=1)
    pers = np.clip(pers + rng.normal(0, 0.12, pers.shape), 0, 1)
    fv = child_g[:, GN.FOOD_PREF0:GN.FOOD_PREF0 + L.N_PLANTS] * 1.0
    gen = np.maximum(pop.generation[pa], f_gen) + 1
    ids = pop.add(
        m,
        x=np.clip(pop.x[pa] + rng.normal(0, 0.3, m), 0, w.W - 1e-3),
        y=np.clip(pop.y[pa] + rng.normal(0, 0.3, m), 0, w.H - 1e-3),
        heading=rng.uniform(0, 2 * np.pi, m), energy=e_child, health=1.0, hydration=1.0,
        max_age=P.max_age(child_g) * rng.uniform(0.85, 1.15, m) * (1 - F), generation=gen,
        parent_a=pop.id[pa], parent_b=f_ids, birth_tick=tick,
        species=pop.species[pa], group=pop.group[pa], genome=child_g, weights=child_w, growth=0.3,
        food_value=fv, personality=pers, norms=np.clip(norms, -1, 1), beliefs=beliefs,
        immunity=0.5 * (pop.immunity[pa] + pop.immunity[pbb]) * 0.3 * (1 - 2 * F),
        immune_strain=pop.immune_strain[pa],      # maternal antibodies
        hue_fear=0.3 * pop.hue_fear[pa],          # weak innate-like transmission via early exposure
        agency=np.clip(0.6 + 0.3 * ((pop.agency[pa] + pop.agency[pbb]) / 2 - 0.6), 0, 1),
        sex=(rng.random(m) < 0.5).astype(np.int8), grand=grand, inbreeding=F, kin_aversion=kin_av,
    )
    sim.moral.educate_newborn(sim, ids, pa)
    n0 = len(pop) - m
    new_idx = np.arange(n0, n0 + m)
    MEM.record(pop, new_idx, MEM.M["born"], 1.0, pop.id[pa], tick)
    MEM.consolidate(pop, new_idx, MEM.M["born"], 1.0, pop.id[pa], tick, detail="was born", force=True)
    MEM.record(pop, pa, MEM.M["gave_birth"], 1.0, ids, tick)
    MEM.consolidate(pop, pa, MEM.M["gave_birth"], 0.8, ids, tick, detail="gave birth")
    for k in range(m):
        cid = int(ids[k])
        pop.names[cid] = make_name(cid)
        sim.social.bond(int(pop.id[pa[k]]), cid, 0.6)   # mother and child are bonded from the start
        if alive_f[k]:
            sim.social.bond(int(pop.id[pbb[k]]), cid, 0.4)
    if (F > 0).any():
        sim.stats_counter["inbred_births"] += int((F > 0).sum())
        sim.analytics.first(sim, "inbreeding", "A child was born to related parents (inbreeding)")
    sim.history.births(sim, new_idx)
    sim.stats_counter["births"] += m
    if sexual.any():
        sim.analytics.first(sim, "sexual_reproduction", "A child was born to two parents")
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
