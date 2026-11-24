"""Biology Engine, part 1: executing chosen actions in the physical world.

Movement, foraging, drinking, resting, predation and mating. Every energy
transfer is booked so that the global energy ledger balances.
"""
from __future__ import annotations

import numpy as np

from ..memory import memory as MEM
from ..physics import laws as L
from . import genome as GN
from .genome import Phenotype as P

A = L.A
C = {c: i for i, c in enumerate(L.DEATH_CAUSES)}


def move(sim, pc, act):
    pop, w, rng = sim.pop, sim.world, sim.rng
    n = len(pop)
    g = pop.genome
    pop.heading += rng.normal(0, 0.6, n).astype(np.float32)
    d = np.stack([np.cos(pop.heading), np.sin(pop.heading)], axis=1).astype(np.float32)
    zero = np.zeros_like(d)

    def toward(idx):
        v = np.zeros((n, 2), np.float32)
        ok = idx >= 0
        v[ok, 0] = pop.x[idx[ok]] - pop.x[ok]
        v[ok, 1] = pop.y[idx[ok]] - pop.y[ok]
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        return np.where(nv > 0.3, v / (nv + 1e-6), 0.0), ok

    seek_better = pc.best_food > pc.food_here * 1.6 + 0.05
    d = np.where(((act == A["seek_food"]) & seek_better)[:, None], pc.food_dir, d)
    d = np.where(((act == A["seek_food"]) & ~seek_better)[:, None], d * 0.3, d)
    d = np.where(((act == A["flee"]) & (pc.threat > 0))[:, None], -pc.threat_dir, d)
    knows_water = np.linalg.norm(pc.water_dir, axis=1) > 0.1
    d = np.where(((act == A["drink"]) & ~pc.water_near & knows_water)[:, None], pc.water_dir, d)
    d = np.where(((act == A["drink"]) & pc.water_near)[:, None], zero, d)
    for a, idx in ((A["mate"], pc.mate_idx), (A["attack"], pc.prey_idx)):
        v, ok = toward(idx)
        d = np.where(((act == a) & ok)[:, None], v, d)
    # social: young organisms follow a parent, others approach friends/kin
    target = np.where(pc.parent_idx >= 0, pc.parent_idx, pc.friend_idx)
    v, ok = toward(target)
    d = np.where(((act == A["social"]) & ok)[:, None], v, d)
    # recall: intention target or remembered food site
    tick = sim.clock.tick
    use_int = pop.intent[:, 3] > tick
    tx = np.where(use_int, pop.intent[:, 0], pop.food_mem[:, 0])
    ty = np.where(use_int, pop.intent[:, 1], pop.food_mem[:, 1])
    rv = np.stack([tx - pop.x, ty - pop.y], axis=1)
    rn = np.linalg.norm(rv, axis=1, keepdims=True)
    has = (use_int | (pop.food_mem[:, 2] > 0)) & (rn[:, 0] > 0.5)
    d = np.where(((act == A["recall"]) & has)[:, None], rv / (rn + 1e-6), d)
    # thermotaxis: a local (felt) temperature gradient biases every movement
    T = w.temperature
    topt = P.temp_opt(g)
    cy, cx = pc.cy, pc.cx
    gx = T[cy, np.minimum(cx + 1, w.W - 1)] - T[cy, np.maximum(cx - 1, 0)]
    gy = T[np.minimum(cy + 1, w.H - 1), cx] - T[np.maximum(cy - 1, 0), cx]
    want = np.sign(topt - T[cy, cx])
    discomfort = np.clip(np.abs(topt - T[cy, cx]) / L.TEMP_TOLERANCE - 0.3, 0, 1)
    gv = np.stack([gx, gy], axis=1) * want[:, None]
    gv = gv / (np.linalg.norm(gv, axis=1, keepdims=True) + 1e-6)
    d = d + 0.6 * discomfort[:, None] * gv
    still = np.isin(act, [A["eat"], A["rest"], A["experiment"], A["work"]])
    d = np.where(still[:, None], zero, d)

    spd = P.speed(g) * (0.5 + 0.5 * pop.health)
    aq = g[:, GN.T["aquatic"]]
    in_w = w.water[pc.cy, pc.cx]
    spd = np.where(in_w, spd * (0.35 + 0.65 * aq), spd * (1 - 0.75 * aq))
    if sim.clock.is_night():
        spd = spd * 0.7
    step = d * spd[:, None]
    nx = np.clip(pop.x + step[:, 0], 0, w.W - 1e-3)
    ny = np.clip(pop.y + step[:, 1], 0, w.H - 1e-3)
    moved = np.hypot(nx - pop.x, ny - pop.y)
    mv = np.linalg.norm(d, axis=1) > 0.05
    pop.heading = np.where(mv, np.arctan2(d[:, 1], d[:, 0]), pop.heading).astype(np.float32)
    pop.x, pop.y = nx.astype(np.float32), ny.astype(np.float32)
    cost = L.MOVE_COST * P.body(g) * moved ** 2 * (1 + 0.5 * w.height_above[pc.cy, pc.cx])
    return cost.astype(np.float32)


def eat(sim, pc, act, gain_out):
    """Foraging. Returns per-organism energy gain and toxin damage."""
    pop, w = sim.pop, sim.world
    n = len(pop)
    g = pop.genome
    # "eat" or foraging that has arrived at the best visible patch
    seek_here = (act == A["seek_food"]) & ~(pc.best_food > pc.food_here * 1.6 + 0.05)
    eaters = np.nonzero((act == A["eat"]) | seek_here)[0]
    damage = np.zeros(n, np.float32)
    if len(eaters) == 0:
        return damage
    cy, cx = w.cell_of(pop.x[eaters], pop.y[eaters])
    t = w.ptype[cy, cx].astype(np.int64)
    size = g[eaters, GN.T["size"]]
    cap = P.capacity(g[eaters])
    room = np.clip(cap - pop.energy[eaters], 0, None)
    # accessibility of hidden or locked energy depends on skills
    sk = pop.skills[eaters]
    from ..intelligence.skills import tool_power, wear_tools
    tool = np.minimum(tool_power(pop, eaters), 1.5)
    unlock = np.zeros(len(eaters), np.float32)
    unlock = np.where(t == L.NUT_TREE, sk[:, L.SK["crack_nuts"]] * (0.6 + 0.4 * tool), unlock)
    unlock = np.where(t == L.TUBER, sk[:, L.SK["dig"]] * (0.6 + 0.4 * tool), unlock)
    unlock = np.minimum(unlock, 1.0)
    access = (1 - L.PLANT_LOCKED[t]) + L.PLANT_LOCKED[t] * unlock
    wear_tools(sim, eaters[(L.PLANT_LOCKED[t] > 0) & (tool > 0)], 0.5)
    reachable = w.biomass[cy, cx] * ((1 - L.PLANT_HIDDEN[t]) + L.PLANT_HIDDEN[t] * unlock)
    in_w = w.water[cy, cx]
    aq = g[eaters, GN.T["aquatic"]]
    habitat = np.where(in_w, aq, 1 - 0.7 * aq)
    reachable = reachable * np.clip(habitat * 1.2, 0, 1)
    # patience norm: leave a seed stock (sustainable harvesting)
    K = w.carrying()[cy, cx]
    keep = 0.25 * K * np.clip(pop.norms[eaters, 2], 0, 1)
    reachable = np.clip(reachable - keep, 0, None)
    diet = g[eaters, GN.T["diet"]]
    digest = 1 - diet * (1 - L.DIGEST_EFF_MISMATCH)
    cooked = (pop.skills[eaters, L.SK["fire"]] > 0.2) & (pc.warmth[eaters] > 0.1)
    eff = np.minimum(1.0, L.PLANT_NUTRITION[t] * access * digest * np.where(cooked, 1.25, 1.0))
    want = np.minimum(L.EAT_BITE * (0.5 + size), room / np.maximum(eff, 0.05))
    want = np.minimum(want, reachable)
    # several organisms in one cell share what is there
    cell = cy * w.W + cx
    demand = np.bincount(cell, weights=want, minlength=w.H * w.W)
    avail = w.biomass.ravel()
    scale = np.where(demand[cell] > avail[cell], avail[cell] / np.maximum(demand[cell], 1e-9), 1.0)
    amount = (want * scale).astype(np.float32)
    np.subtract.at(w.biomass.ravel(), cell, amount)
    np.clip(w.biomass, 0, None, out=w.biomass)
    gain = amount * eff
    # energy that could not be assimilated goes to detritus (dung)
    waste = amount - gain
    np.add.at(w.detritus.ravel(), cell, np.clip(waste, 0, None) * 0.5)
    sim.world.ledger.heat_out += float(np.clip(waste, 0, None).sum() * 0.5)
    tox = L.PLANT_TOXICITY[t]
    dmg = amount * tox * 1.2
    damage[eaters] = dmg
    pop.energy[eaters] += gain
    gain_out[eaters] += gain

    # taste learning (conditioned taste aversion / preference)
    lr = 0.08 + 0.35 * g[eaters, GN.T["learn_rate"]] * (0.2 + g[eaters, GN.T["brain"]])
    has = amount > 1e-4
    obs = np.where(has, gain / np.maximum(amount, 1e-6) - dmg / np.maximum(amount, 1e-6) * 1.0, 0)
    obs = np.clip(obs, -1.5, 2.0)
    ii = eaters[has]
    tt = t[has]
    novel = np.abs(pop.food_value[ii, tt] - obs[has]) > 0.5
    pop.food_value[ii, tt] += (lr[has] * (obs[has] - pop.food_value[ii, tt])).astype(np.float32)
    tick = sim.clock.tick
    MEM.record(pop, ii, MEM.M["ate"], gain[has], tt, tick)
    pois = dmg > 0.02
    if pois.any():
        pi = eaters[pois]
        MEM.record(pop, pi, MEM.M["poisoned"], -dmg[pois], t[pois], tick)
        MEM.consolidate(pop, pi, MEM.M["poisoned"], -dmg[pois] * 3, t[pois], tick,
                        detail="felt sick after eating " + "/".join(sorted({L.PLANTS[k]["name"] for k in t[pois]})))
        pop.emotion[pi, 0] = np.minimum(1, pop.emotion[pi, 0] + 0.3)
    if novel.any():
        MEM.consolidate(pop, ii[novel], MEM.M["first_taste"], np.abs(obs[has][novel] - 0.5) + 0.3, tt[novel], tick,
                        detail="discovered the taste of a plant")
    # place memory: good food remembered
    better = gain > np.maximum(pop.food_mem[eaters, 2] * 0.3, 0.02)
    bi = eaters[better]
    pop.food_mem[bi, 0] = pop.x[bi]
    pop.food_mem[bi, 1] = pop.y[bi]
    pop.food_mem[bi, 2] = np.maximum(pop.food_mem[bi, 2], obs[better] if better.any() else 0)
    return damage


def drink(sim, pc, act):
    pop, w = sim.pop, sim.world
    g = pop.genome
    dr = (act == A["drink"]) & pc.water_near
    aq_in = w.water[pc.cy, pc.cx] & (g[:, GN.T["aquatic"]] > 0.5)
    pop.hydration[dr | aq_in] = 1.0
    wi = np.nonzero(pc.water_near)[0]
    pop.water_mem[wi, 0] = pop.x[wi]
    pop.water_mem[wi, 1] = pop.y[wi]
    pop.water_mem[wi, 2] = 1.0
    return dr


def attack(sim, pc, act, gain_out, damage_by_cause):
    """Predation and violence. Returns list of (attacker_idx, target_idx, killed, cause)."""
    pop, w, rng = sim.pop, sim.world, sim.rng
    g = pop.genome
    att = np.nonzero((act == A["attack"]) & (pc.prey_idx >= 0) & (pc.prey_dist <= 1.3))[0]
    out = []
    if len(att) == 0:
        return out
    tgt = pc.prey_idx[att]
    from ..intelligence.skills import tool_power, wear_tools
    tool = tool_power(pop)
    strength = pop.body_mass() * (0.5 + 0.5 * pop.health) * (1 + 0.4 * tool) * (0.7 + 0.6 * g[:, GN.T["aggression"]])
    # cooperative hunting: allies attacking the same target add strength
    allies = np.bincount(tgt, minlength=len(pop))[tgt] - 1
    s_att = strength[att] * (1 + 0.35 * allies)
    s_def = strength[tgt] * (1 + 0.6 * P.speed(g[tgt]) / (P.speed(g[att]) + 1e-3))
    p = 1 / (1 + np.exp(-(s_att - s_def) * 2.5))
    hit = rng.random(len(att)) < p
    dmg = np.where(hit, 0.25 + 0.35 * s_att / (s_att + s_def), 0.03)
    gd = GN.genetic_distance(g[att], g[tgt])
    cause = np.where(gd > 0.1, C["predation"], C["violence"])
    np.add.at(damage_by_cause, (tgt, cause), dmg)
    wear_tools(sim, att, 1.0)
    pop.energy[att] -= 0.01 * P.body(g[att])
    w.ledger.heat_out += float((0.01 * P.body(g[att])).sum())
    # learning by the victim: fear of this appearance, remember the place
    from ..intelligence.perception import hue_bin
    hb = hue_bin(g[att])
    lr = 0.25 + 0.5 * g[tgt, GN.T["learn_rate"]]
    np.add.at(pop.hue_fear, (tgt, hb), lr * dmg)
    np.clip(pop.hue_fear, 0, 1, out=pop.hue_fear)
    pop.danger_mem[tgt, 0] = pop.x[tgt]
    pop.danger_mem[tgt, 1] = pop.y[tgt]
    pop.danger_mem[tgt, 2] = 1.0
    pop.emotion[tgt, 0] = np.minimum(1, pop.emotion[tgt, 0] + 0.6)
    pop.emotion[tgt, 2] = np.minimum(1, pop.emotion[tgt, 2] + 0.4 * (1 - g[tgt, GN.T["fear_sens"]]))
    tick = sim.clock.tick
    MEM.record(pop, tgt, MEM.M["attacked"], -dmg, pop.id[att], tick)
    MEM.consolidate(pop, tgt, MEM.M["attacked"], -dmg * 2, pop.id[att], tick, detail="was attacked")
    for k in range(len(att)):
        out.append((int(att[k]), int(tgt[k]), bool(hit[k]), int(cause[k])))
    return out


def feed_on_kills(sim, kills, gain_out):
    """Predators consume the bodies of prey that died this tick."""
    pop, w = sim.pop, sim.world
    g = pop.genome
    for a, t in kills:
        E = float(pop.energy[t] + pop.body_mass()[t] + pop.inventory[t, L.R["food"]])
        pop.energy[t] = 0
        pop.inventory[t, L.R["food"]] = 0
        sim.consumed[t] = True
        diet = float(g[a, GN.T["diet"]])
        eff = L.PREDATION_EFF * (L.DIGEST_EFF_MISMATCH + (1 - L.DIGEST_EFF_MISMATCH) * diet)
        room = float(P.capacity(g[a:a + 1])[0] - pop.energy[a])
        gain = max(0.0, min(room, eff * E))
        pop.energy[a] += gain
        gain_out[a] += gain
        rest = E - gain
        cy, cx = w.cell_of(pop.x[t:t + 1], pop.y[t:t + 1])
        w.detritus[cy[0], cx[0]] += rest * 0.7
        w.ledger.heat_out += rest * 0.3
        pop.kills[a] += 1
