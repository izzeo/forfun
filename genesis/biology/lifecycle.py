"""Biology Engine, part 2: metabolism, environment, disease, ageing and death.

Every organism that disappears does so for a recorded physical cause.
"""
from __future__ import annotations

import numpy as np

from ..memory import memory as MEM
from ..physics import laws as L
from ..physics.spatial import neighbor_pairs
from . import genome as GN
from .genome import Phenotype as P

A = L.A
C = {c: i for i, c in enumerate(L.DEATH_CAUSES)}


def metabolize(sim, pc, act, move_cost, dmg):
    """Energy use, thirst, temperature, drowning, fire, storms, disease, healing, ageing."""
    pop, w, rng, tick = sim.pop, sim.world, sim.rng, sim.clock.tick
    n = len(pop)
    g = pop.genome
    cy, cx = w.cell_of(pop.x, pop.y)
    resting = act == A["rest"]

    # ---- temperature & shelter --------------------------------------------
    T = w.temperature[cy, cx]
    topt = P.temp_opt(g)
    shelter = (w.struct[cy, cx] == L.STRUCT_SHELTER)
    tol = L.TEMP_TOLERANCE * (0.8 + 0.5 * g[:, GN.T["size"]]) + 6.0 * shelter
    cold_relief = 10.0 * (pc.warmth > 0.1)
    # huddling: bodies close together keep each other warm
    I, J, D = pc.pairs
    huddle = np.bincount(I[D < 1.0], minlength=n) if len(I) else np.zeros(n)
    cold_relief = cold_relief + 1.5 * np.minimum(huddle, 4)
    dev = np.where(T < topt, np.clip(topt - T - cold_relief, 0, None), T - topt)
    excess = np.clip(dev - tol, 0, None)
    temp_cost = 0.002 * dev / L.TEMP_TOLERANCE * P.body(g) ** 0.75   # thermoregulation
    dmg[:, C["exposure"]] += 0.002 * excess

    # ---- energy budget -------------------------------------------------------
    cost = P.basal(g) * np.where(resting, 0.7, 1.0) + move_cost + temp_cost
    paid = np.minimum(cost, pop.energy)
    pop.energy -= paid
    w.ledger.heat_out += float(paid.sum())
    starving = (cost - paid) > 1e-7

    # ---- growth: juveniles build body tissue from their energy store ------
    growing = pop.growth < 1.0
    if growing.any():
        gi = np.nonzero(growing)[0]
        body_full = P.body(g[gi])
        rate = 0.7 / np.maximum(P.maturity(g[gi]) * pop.max_age[gi], 24)
        amt = np.minimum(rate * body_full, 0.5 * np.clip(pop.energy[gi] - 0.15 * P.capacity(g[gi]), 0, None))
        amt = np.minimum(amt, (1 - pop.growth[gi]) * body_full)
        pop.energy[gi] -= amt
        pop.growth[gi] += (amt / body_full).astype(np.float32)

    # ---- water --------------------------------------------------------------
    aq = g[:, GN.T["aquatic"]]
    in_w = w.water[cy, cx]
    pop.hydration -= (0.006 * (1 + np.clip(T - 25, 0, None) / 15) * (1 - 0.8 * (in_w & (aq > 0.5)))).astype(np.float32)
    np.clip(pop.hydration, 0, 1, out=pop.hydration)
    dmg[:, C["dehydration"]] += np.where(pop.hydration <= 0, 0.015, 0)
    dmg[:, C["drowning"]] += np.where(w.deep[cy, cx], 0.06 * (1 - aq) ** 2, 0)
    dmg[:, C["dehydration"]] += np.where(~in_w, 0.03 * np.clip(aq - 0.6, 0, None), 0)

    # ---- fire & storms --------------------------------------------------------
    f = w.fire[cy, cx]
    burning = f > 0.3
    if burning.any():
        dmg[burning, C["fire"]] += 0.25 * f[burning] * (1 - 0.6 * (pop.skills[burning, L.SK["fire"]] > 0.2))
        MEM.consolidate(pop, np.nonzero(burning)[0], MEM.M["fire"], -0.8, 0, tick, detail="was caught in a fire")
        pop.emotion[burning, 0] = 1.0
    for s in w.storms:
        d = np.hypot(pop.x - s["cx"], pop.y - s["cy"])
        inside = (d < s["r"]) & ~shelter
        if inside.any():
            hit = inside & (rng.random(n) < 0.004 * s["intensity"])
            dmg[hit, C["storm"]] += rng.uniform(0.2, 0.9, int(hit.sum()))
            pop.emotion[inside, 0] = np.minimum(1, pop.emotion[inside, 0] + 0.05)
            sim.culture.unexplained_harm(sim, np.nonzero(hit)[0], 0.5, "storm")

    # ---- disease (hidden microscopic layer acting causally) ---------------
    P_here = w.pathogens[cy, cx]
    vir = w.virulence[cy, cx]
    hygiene = sim.civ.hygiene(sim)
    expo = rng.random(n) < 0.015 * P_here * (1 - pop.immunity) * (1 - hygiene)
    pop.infection[expo] += 0.15 * vir[expo]
    if n > 1:
        I, J, D = pc.pairs
        close = (D < 1.2) & (pop.infection[J] > 0.1)
        if close.any():
            tr = rng.random(int(close.sum())) < 0.01 * pop.infection[J[close]] * (1 - pop.immunity[I[close]]) * (1 - hygiene[I[close]])
            np.add.at(pop.infection, I[close][tr], 0.1)
    inf = pop.infection > 0
    if inf.any():
        ii = np.nonzero(inf)[0]
        pop.infection[ii] += (0.004 * vir[ii] - 0.008 * (pop.immunity[ii] + 0.3) - 0.002 * resting[ii]).astype(np.float32)
        pop.immunity[ii] = np.minimum(1, pop.immunity[ii] + 0.006 * pop.infection[ii])
        dmg[ii, C["disease"]] += 0.012 * np.clip(pop.infection[ii], 0, 1) * vir[ii]
        np.add.at(w.pathogens, (cy[ii], cx[ii]), 0.003 * np.clip(pop.infection[ii], 0, 1))
        newly_ill = ii[(pop.infection[ii] > 0.3) & (pop.infection[ii] < 0.32)]
        if len(newly_ill):
            MEM.consolidate(pop, newly_ill, MEM.M["ill"], -0.7, 0, tick, detail="fell ill")
            sim.culture.unexplained_harm(sim, newly_ill, 0.6, "disease")
    np.clip(pop.infection, 0, 1.5, out=pop.infection)
    pop.immunity *= np.float32(0.9998)

    # ---- apply damage, heal ------------------------------------------------
    total = dmg.sum(axis=1)
    pop.health -= total.astype(np.float32)
    cap = P.capacity(g)
    agefrac = pop.age / np.maximum(pop.max_age, 1)
    healthy = (pop.energy > 0.25 * cap) & (pop.hydration > 0.3) & (pop.infection < 0.05) & (total < 1e-4)
    regen = 0.004 * (1 + 2 * resting) * np.clip(1.2 - agefrac, 0.1, 1.0)
    pop.health = np.where(healthy, np.minimum(1.0, pop.health + regen), pop.health).astype(np.float32)

    # ---- ageing (Gompertz hazard) ---------------------------------------
    pop.age += 1
    hazard = 3e-6 * np.exp(9.0 * agefrac)
    old = (rng.random(n) < hazard) | (agefrac > 1.5)

    # ---- determine deaths and causes -------------------------------------
    cause = np.full(n, -1, np.int64)
    dead_h = pop.health <= 0
    cause[dead_h] = dmg[dead_h].argmax(axis=1)
    dead_s = (pop.energy <= 1e-6) & starving & (cause < 0)
    cause[dead_s] = C["starvation"]
    dead_o = old & (cause < 0)
    cause[dead_o] = C["old_age"]
    cause[sim.consumed & (cause < 0)] = C["predation"]
    return cause


def bury(sim, cause: np.ndarray, killers: dict[int, int]):
    """Remove the dead, returning their energy to the soil and recording the cause."""
    pop, w, tick = sim.pop, sim.world, sim.clock.tick
    dead = cause >= 0
    if not dead.any():
        return 0
    idx = np.nonzero(dead)[0]
    g = pop.genome
    body = pop.body_mass()[idx]
    E = np.where(sim.consumed[idx], 0.0, pop.energy[idx] + body + pop.inventory[idx, L.R["food"]])
    cy, cx = w.cell_of(pop.x[idx], pop.y[idx])
    np.add.at(w.detritus, (cy, cx), E.astype(np.float32))
    for k, i in enumerate(idx):
        c = L.DEATH_CAUSES[int(cause[i])]
        sim.stats_counter["deaths"] += 1
        sim.stats_counter["death_" + c] += 1
    sim.history.deaths(sim, idx, cause, killers)
    sim.psyche.on_deaths(sim, idx)
    sim.civ.on_deaths(sim, idx)
    for i in idx:
        pop.forget(int(pop.id[i]))
    pop.remove(dead)
    return len(idx)
