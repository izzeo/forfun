"""Tool use, experimentation and the application of skills.

Discovery is a genuine trial-and-error process: an organism picks a verb and
a random subset of the objects physically present, and the *world's* recipe
table (physics.laws.RECIPES) decides whether anything happens. The organism
never reads the table — it only experiences the outcome.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..intelligence.brain import life_stage
from ..memory import memory as MEM
from ..physics import laws as L
from .perception import OBJ, OBJECTS

A = L.A
_RECIPE_ITEMS = [(name, L.VERBS.index(v), [OBJ[o] for o in objs], diff, cond, L.SK[L.RECIPE_SKILL[name]])
                 for name, (v, objs, diff, cond) in L.RECIPES.items()]


def experiment(sim, pc, act, reward) -> int:
    pop, rng, tick = sim.pop, sim.rng, sim.clock.tick
    ex = np.nonzero(act == A["experiment"])[0]
    if len(ex) == 0:
        return 0
    g = pop.genome
    night = sim.clock.is_night()
    st = life_stage(pop)
    found = 0
    for i in ex:
        present = np.nonzero(pc.materials[i])[0]
        if len(present) == 0:
            continue
        pv = np.exp(pop.explore_bias[i])
        verb = int(rng.choice(len(L.VERBS), p=pv / pv.sum()))
        chosen = set(int(o) for o in present[rng.random(len(present)) < 0.65])
        pop.energy[i] -= 0.002
        sim.world.ledger.heat_out += 0.002
        reward[i] += 0.01 * g[i, GN.T["curiosity"]]  # intrinsic reward of exploring
        for name, v, objs, diff, cond, sk in _RECIPE_ITEMS:
            if v != verb or not all(o in chosen for o in objs):
                continue
            if cond == "dry" and not pc.dry[i]:
                continue
            if cond == "night" and not night:
                continue
            if pop.skills[i, sk] > 0.05:
                pop.skills[i, sk] = min(1.0, pop.skills[i, sk] + 0.02)
                continue
            p = diff * (0.25 + g[i, GN.T["brain"]]) * (0.4 + g[i, GN.T["curiosity"]]) * (1.6 if st[i] <= 2 else 1.0)
            if rng.random() < p:
                learn_skill(sim, i, sk, 0.3, source="experiment", via=name)
                pop.explore_bias[i, verb] = min(3.0, pop.explore_bias[i, verb] + 0.4)
                reward[i] += 0.4
                found += 1
    return found


def learn_skill(sim, i: int, sk: int, level: float, source: str, via: str | None = None, teacher: int = -1):
    pop, tick = sim.pop, sim.clock.tick
    first_for_agent = pop.skills[i, sk] <= 0.05
    pop.skills[i, sk] = max(pop.skills[i, sk], level)
    if not first_for_agent:
        return
    name = L.SKILLS[sk]
    etype = MEM.M["discovered"] if source == "experiment" else MEM.M["learned"]
    MEM.record(pop, np.array([i]), etype, 1.0, sk, tick)
    MEM.consolidate(pop, np.array([i]), etype, 1.0, teacher, tick,
                    detail=(f"discovered how to {name.replace('_', ' ')}" if source == "experiment"
                            else f"learned {name.replace('_', ' ')} from {pop.names.get(teacher, 'another')}"), force=True)
    pop.personality[i, 0] = min(1.0, pop.personality[i, 0] + 0.02)
    if source == "experiment":
        is_new = sim.civ.discover_skill(sim, i, sk)
        sim.history.event(sim, "discovery" if is_new else "rediscovery", int(pop.id[i]), -1, pop.x[i], pop.y[i],
                          dict(skill=name, via=via))
        sim.analytics.first(sim, "skill:" + name, f"{pop.names.get(int(pop.id[i]), '?')} discovered {name.replace('_', ' ')}")
        if name == "microscope":
            sim.civ.discover_layer(sim, "microscopic", i)
        if name == "telescope":
            sim.civ.discover_layer(sim, "celestial", i)
    else:
        sim.analytics.first(sim, "skill_transmission", "Knowledge passed from one individual to another")
        sim.civ.discover_skill(sim, i, sk, transmitted=True)


def imitate(sim, I: np.ndarray, J: np.ndarray, boost: np.ndarray) -> int:
    """Social learning: I observes J. Vectorised over pairs and skills."""
    pop, rng = sim.pop, sim.rng
    if len(I) == 0:
        return 0
    g = pop.genome
    st = life_stage(pop)
    gap = pop.skills[J] - pop.skills[I]
    able = (pop.skills[J] > 0.15) & (gap > 0.1)
    p = (0.03 * (0.2 + g[I, GN.T["brain"]]) * (0.3 + g[I, GN.T["sociality"]])
         * np.where(st[I] <= 2, 2.0, 1.0) * boost)[:, None]
    teach = (0.5 + pop.norms[J, 3:4].clip(0, 1) + (st[J] == 4)[:, None] * 0.5)  # responsibility, elders teach
    hit = able & (rng.random(gap.shape) < p * teach)
    cnt = 0
    for k, sk in zip(*np.nonzero(hit)):
        i, j = int(I[k]), int(J[k])
        level = float(pop.skills[i, sk] + 0.5 * gap[k, sk])
        learn_skill(sim, i, int(sk), level, source="social", teacher=int(pop.id[j]))
        MEM.record(pop, np.array([j]), MEM.M["taught"], 0.5, pop.id[i], sim.clock.tick)
        cnt += 1
    return cnt


def work(sim, pc, act, reward) -> int:
    """Apply known skills: gather, craft, build, cultivate, store food."""
    pop, w, rng, tick = sim.pop, sim.world, sim.rng, sim.clock.tick
    wk = np.nonzero(act == A["work"])[0]
    if len(wk) == 0:
        return 0
    g = pop.genome
    sk = pop.skills
    inv = pop.inventory
    lim = L.CARRY_LIMIT
    R, S = L.R, L.SK
    cap = P.capacity(g)
    did = 0
    for i in wk:
        cy, cx = int(pc.cy[i]), int(pc.cx[i])
        mat = pc.materials[i]
        cold = w.temperature[cy, cx] < P.temp_opt(g[i:i + 1])[0] - 6 or sim.clock.is_night()
        land = not w.water[cy, cx]
        done = None
        # --- crafting / building, highest value first ---------------------
        if sk[i, S["fire"]] > 0.1 and inv[i, R["wood"]] >= 1 and cold and land and pc.warmth[i] < 0.1 and w.struct[cy, cx] == 0:
            inv[i, R["wood"]] -= 1
            _build(sim, i, cy, cx, L.STRUCT_HEARTH, hp=1.0)
            done = "hearth"
        elif sk[i, S["shelter"]] > 0.1 and inv[i, R["wood"]] >= 2 and land and w.struct[cy, cx] == 0 and _near_struct(w, cy, cx, L.STRUCT_SHELTER, 1) < 3:
            inv[i, R["wood"]] -= 2
            _build(sim, i, cy, cx, L.STRUCT_SHELTER, hp=1.0)
            done = "shelter"
        elif sk[i, S["knap_tool"]] > 0.1 and inv[i, R["tool"]] < 1 and mat[OBJ["stone2"]]:
            _take_stone(w, inv, i, cy, cx, 2)
            inv[i, R["tool"]] = 1
            done = "tool"
        elif sk[i, S["smelting"]] > 0.1 and inv[i, R["ore"]] >= 1 and mat[OBJ["fire"]] and inv[i, R["pottery"]] >= 1 and inv[i, R["metal"]] < 1:
            inv[i, R["ore"]] -= 1
            inv[i, R["metal"]] = 1
            done = "metal"
        elif sk[i, S["glassmaking"]] > 0.1 and mat[OBJ["sand"]] and mat[OBJ["fire"]] and inv[i, R["pottery"]] >= 1 and inv[i, R["glass"]] < 1:
            if inv[i, R["sand"]] >= 1:
                inv[i, R["sand"]] -= 1
            inv[i, R["glass"]] = 1
            done = "glass"
        elif sk[i, S["pottery"]] > 0.1 and mat[OBJ["clay"]] and mat[OBJ["fire"]] and inv[i, R["pottery"]] < 1:
            if inv[i, R["clay"]] >= 1:
                inv[i, R["clay"]] -= 1
            inv[i, R["pottery"]] = 1
            done = "pottery"
        elif sk[i, S["cultivate"]] > 0.1 and land and w.dug[cy, cx] and w.struct[cy, cx] == 0 and pop.energy[i] > 0.4 * cap[i]:
            # plant the crop this individual values most
            crops = np.array([L.BERRY, L.NUT_TREE, L.TUBER])
            best = int(crops[np.argmax(pop.food_value[i, crops])])
            seed = 0.08
            pop.energy[i] -= seed
            w.ptype[cy, cx] = best
            w.kmask[cy, cx] = 1.5 * (1.3 if sim.civ.knows_layer(sim, i, "celestial") else 1.0)
            w.biomass[cy, cx] += seed
            _build(sim, i, cy, cx, L.STRUCT_FARM, hp=1.0)
            done = "farm"
        elif sk[i, S["dig"]] > 0.1 and land and not w.dug[cy, cx]:
            w.dug[cy, cx] = True
            if w.ore[cy, cx] > 0.5 and inv[i, R["ore"]] < lim[R["ore"]]:
                w.ore[cy, cx] -= 1
                inv[i, R["ore"]] += 1
                sim.civ.discover_layer(sim, "subterranean", i)
                sim.analytics.first(sim, "ore_found", "Ore discovered underground by digging")
            if w.clay[cy, cx] and inv[i, R["clay"]] < lim[R["clay"]]:
                inv[i, R["clay"]] += 1
            done = "dig"
        elif sk[i, S["pottery"]] > 0.1 and inv[i, R["pottery"]] >= 1 and inv[i, R["food"]] < lim[R["food"]] and pop.energy[i] > 0.7 * cap[i] and w.biomass[cy, cx] > 0.3:
            take = min(0.3, float(w.biomass[cy, cx]) - 0.1)
            w.biomass[cy, cx] -= take
            inv[i, R["food"]] += take
            done = "store_food"
        elif pop.beliefs[i, 2] > 0.5 and pop.group[i] >= 0 and inv[i, R["wood"]] >= 1 and land and w.struct[cy, cx] == 0 and _near_struct(w, cy, cx, L.STRUCT_SHRINE, 5) == 0:
            inv[i, R["wood"]] -= 1
            _build(sim, i, cy, cx, L.STRUCT_SHRINE, hp=1.0)
            done = "shrine"
        # --- gathering ---------------------------------------------------
        elif w.stone[cy, cx] >= 1 and inv[i, R["stone"]] < lim[R["stone"]] and (sk[i, S["knap_tool"]] > 0.1 or sk[i, S["crack_nuts"]] > 0.1):
            w.stone[cy, cx] -= 1
            inv[i, R["stone"]] += 1
            done = "gather_stone"
        elif mat[OBJ["nut"]] and w.biomass[cy, cx] > 0.4 and inv[i, R["wood"]] < lim[R["wood"]] and (sk[i, S["shelter"]] > 0.1 or sk[i, S["fire"]] > 0.1):
            w.biomass[cy, cx] -= 0.1
            w.detritus[cy, cx] += 0.1
            inv[i, R["wood"]] += 1
            done = "gather_wood"
        elif w.sand[cy, cx] and sk[i, S["glassmaking"]] > 0.1 and inv[i, R["sand"]] < lim[R["sand"]]:
            inv[i, R["sand"]] += 1
            done = "gather_sand"
        elif w.clay[cy, cx] and sk[i, S["pottery"]] > 0.1 and inv[i, R["clay"]] < lim[R["clay"]]:
            inv[i, R["clay"]] += 1
            done = "gather_clay"
        if done:
            did += 1
            reward[i] += 0.05
            for s in _USES.get(done, ()):
                sk[i, S[s]] = min(1.0, sk[i, S[s]] + 0.01)
            if done in ("hearth", "shelter", "farm", "shrine"):
                MEM.record(pop, np.array([i]), MEM.M["built"], 0.4, 0, tick)
                MEM.consolidate(pop, np.array([i]), MEM.M["built"], 0.5, 0, tick, detail=f"built a {done}")
                sim.analytics.first(sim, "built:" + done, f"First {done} built")
    return did


_USES = {"hearth": ["fire"], "shelter": ["shelter"], "tool": ["knap_tool"], "metal": ["smelting"],
         "glass": ["glassmaking"], "pottery": ["pottery"], "farm": ["cultivate"], "dig": ["dig"],
         "store_food": ["pottery"]}


def _take_stone(w, inv, i, cy, cx, k):
    from_inv = min(inv[i, L.R["stone"]], k)
    inv[i, L.R["stone"]] -= from_inv
    rest = k - from_inv
    w.stone[cy, cx] = max(0.0, w.stone[cy, cx] - rest)


def _near_struct(w, cy, cx, kind, r):
    y0, y1 = max(0, cy - r), min(w.H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w.W, cx + r + 1)
    return int((w.struct[y0:y1, x0:x1] == kind).sum())


def _build(sim, i, cy, cx, kind, hp):
    w = sim.world
    w.struct[cy, cx] = kind
    w.struct_hp[cy, cx] = hp
    w.struct_group[cy, cx] = int(sim.pop.group[i])
    sim.history.event(sim, "built", int(sim.pop.id[i]), -1, cx, cy, dict(structure=L.STRUCTURES[kind]))
