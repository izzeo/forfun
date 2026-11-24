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
from ..physics import chemistry as CH
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
            if cond is not None and not physical_condition(sim, pc, i, cond):
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
    able = (pop.skills[J] > 0.12) & (gap > 0.05)
    p = (0.03 * (0.2 + g[I, GN.T["brain"]]) * (0.3 + g[I, GN.T["sociality"]])
         * np.where(st[I] <= 2, 2.0, 1.0) * boost)[:, None]
    teach = (0.5 + pop.norms[J, 3:4].clip(0, 1) + (st[J] == 4)[:, None] * 0.5)  # responsibility, elders teach
    hit = able & (rng.random(gap.shape) < p * teach)
    cnt = 0
    for k, sk in zip(*np.nonzero(hit)):
        i, j = int(I[k]), int(J[k])
        # copying a demonstrated procedure is fairly faithful; mastery then grows with practice
        level = float(pop.skills[i, sk] + 0.8 * gap[k, sk])
        learn_skill(sim, i, int(sk), level, source="social", teacher=int(pop.id[j]))
        MEM.record(pop, np.array([j]), MEM.M["taught"], 0.5, pop.id[i], sim.clock.tick)
        cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# physics of heat and materials, as experienced by one organism at one place
# ---------------------------------------------------------------------------
def heat_setup(w, cy, cx):
    """Best fire installation within one cell: FURNACE > KILN > OPEN (hearth or wildfire), else None."""
    y0, y1 = max(0, cy - 1), min(w.H, cy + 2)
    x0, x1 = max(0, cx - 1), min(w.W, cx + 2)
    st = w.struct[y0:y1, x0:x1]
    if (st == L.STRUCT_FURNACE).any():
        return CH.FURNACE
    if (st == L.STRUCT_KILN).any():
        return CH.KILN
    if (st == L.STRUCT_HEARTH).any() or (w.fire[y0:y1, x0:x1] > 0.1).any():
        return CH.OPEN
    return None


def fire_temperature(sim, i, cy, cx):
    """(temperature reached, fuel used) if organism i fires up what is here with what it carries."""
    w, inv = sim.world, sim.pop.inventory
    setup = heat_setup(w, cy, cx)
    if setup is None:
        return 0.0, "none", None
    if inv[i, L.R["charcoal"]] >= 1:
        fuel, m = "charcoal", 0.0
    elif inv[i, L.R["wood"]] >= 1:
        fuel, m = "wood", float(sim.pop.wood_moist[i])
    elif setup == CH.OPEN:
        fuel, m = "ambient", 0.3            # an existing hearth / wildfire burning damp wood
    else:
        return 0.0, "none", setup           # a cold kiln without fuel
    return CH.fire_temperature("wood" if fuel == "ambient" else fuel, m, setup), fuel, setup


def burn_fuel(sim, i, fuel):
    inv, w = sim.pop.inventory, sim.world
    if fuel == "charcoal":
        inv[i, L.R["charcoal"]] -= 1
        w.ledger.heat_out += CH.CHARCOAL_ENERGY
    elif fuel == "wood":
        inv[i, L.R["wood"]] -= 1
        w.ledger.heat_out += CH.WOOD_ENERGY


def _stone_rock(sim, i, cy, cx):
    w = sim.world
    return int(w.rock[cy, cx]) if w.stone[cy, cx] >= 1 else int(sim.pop.stone_rock[i])


def physical_condition(sim, pc, i, cond) -> bool:
    """Does the physics of this place allow the effect? (hidden from organisms)"""
    pop, w = sim.pop, sim.world
    cy, cx = int(pc.cy[i]), int(pc.cx[i])
    if cond == "knappable":
        return sim.rng.random() < CH.ROCK_KNAP[_stone_rock(sim, i, cy, cx)]
    if cond == "spark":
        tinder_dry = (pop.inventory[i, L.R["wood"]] >= 1 and pop.wood_moist[i] < 0.25) or bool(pc.dry[i])
        return tinder_dry and sim.rng.random() < CH.spark_chance(_stone_rock(sim, i, cy, cx))
    if cond == "clear_glass":
        return pop.glass_q[i] >= 0.6
    if cond.startswith("T:"):
        T, fuel, setup = fire_temperature(sim, i, cy, cx)
        kind = cond[2:]
        if kind == "ceramic":
            return T >= CH.T_CERAMIC
        if kind == "charcoal":
            return setup in (CH.KILN, CH.FURNACE) and T >= CH.T_CHARCOAL
        if kind == "smelt":
            inv = pop.inventory
            return (inv[i, L.R["copper_ore"]] >= 1 and T >= CH.T_COPPER) or (inv[i, L.R["ore"]] >= 1 and T >= CH.T_IRON)
        if kind == "glass":
            return T >= CH.T_GLASS_FLUXED
    return True


def tool_power(pop, idx=None) -> np.ndarray:
    """Effectiveness of the best tool each organism carries (0 = bare hands)."""
    sl = slice(None) if idx is None else idx
    inv = pop.inventory[sl]
    stone = np.where(inv[:, L.R["tool"]] >= 1, pop.tool_q[sl], 0)
    metal = np.where(inv[:, L.R["metal"]] >= 1, pop.metal_q[sl], 0)
    return np.maximum(stone, metal).astype(np.float32)


def wear_tools(sim, idx, uses: float = 1.0) -> None:
    """Tools wear with use; a worn-out tool breaks and its matter becomes debris."""
    pop, w = sim.pop, sim.world
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return
    inv = pop.inventory
    has_m = inv[idx, L.R["metal"]] >= 1
    has_s = (inv[idx, L.R["tool"]] >= 1) & ~has_m          # the best tool is the one in use
    mi, si = idx[has_m], idx[has_s]
    metal_wear = np.where(pop.metal_q[mi] >= CH.METALS["iron"]["effect"] - 1e-6, CH.METALS["iron"]["wear"], CH.METALS["copper"]["wear"])
    pop.metal_dur[mi] -= (metal_wear * uses).astype(np.float32)
    pop.tool_dur[si] -= np.float32(CH.TOOL_STONE_WEAR * uses)
    for arr_idx, res, dur, q, units in ((si, "tool", pop.tool_dur, pop.tool_q, L.MINERAL_UNITS["tool"]),
                                        (mi, "metal", pop.metal_dur, pop.metal_q, L.MINERAL_UNITS["metal"])):
        broken = arr_idx[dur[arr_idx] <= 0]
        for k in broken:
            inv[k, L.R[res]] -= 1
            cy, cx = w.cell_of(pop.x[k:k + 1], pop.y[k:k + 1])
            w.debris[cy[0], cx[0]] += units
            dur[k] = 1.0
            q[k] = 0.0
            sim.analytics.first(sim, "tool_broke", "A tool wore out and broke")


def work(sim, pc, act, reward) -> int:
    """Apply known skills: gather, craft, build, cultivate, store food. Matter and energy are conserved."""
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
    used_tool = []
    for i in wk:
        cy, cx = int(pc.cy[i]), int(pc.cx[i])
        mat = pc.materials[i]
        cold = w.temperature[cy, cx] < P.temp_opt(g[i:i + 1])[0] - 6 or sim.clock.is_night()
        land = not w.water[cy, cx]
        empty = land and w.struct[cy, cx] == 0
        done = None
        T, fuel, setup = fire_temperature(sim, i, cy, cx) if mat[OBJ["fire"]] or mat[OBJ["kiln"]] else (0.0, "none", None)
        # --- fire & building --------------------------------------------------
        if sk[i, S["fire"]] > 0.1 and inv[i, R["wood"]] >= 1 and cold and pc.warmth[i] < 0.1 and empty:
            burn_fuel(sim, i, "wood")
            _build(sim, i, cy, cx, L.STRUCT_HEARTH, hp=1.0)
            done = "hearth"
        elif sk[i, S["shelter"]] > 0.1 and inv[i, R["wood"]] >= 2 and empty and _near_struct(w, cy, cx, L.STRUCT_SHELTER, 1) < 3:
            inv[i, R["wood"]] -= 2
            w.struct_energy[cy, cx] += 2 * CH.WOOD_ENERGY
            _build(sim, i, cy, cx, L.STRUCT_SHELTER, hp=1.0)
            done = "shelter"
            used_tool.append(i)
        elif sk[i, S["kiln"]] > 0.1 and inv[i, R["clay"]] >= 2 and inv[i, R["stone"]] >= 2 and empty \
                and _near_struct(w, cy, cx, L.STRUCT_KILN, 3) + _near_struct(w, cy, cx, L.STRUCT_FURNACE, 3) == 0:
            inv[i, R["clay"]] -= 2
            inv[i, R["stone"]] -= 2
            w.struct_minerals[cy, cx] += 4
            _build(sim, i, cy, cx, L.STRUCT_KILN, hp=3.0)
            done = "kiln"
        elif sk[i, S["furnace"]] > 0.1 and w.struct[cy, cx] == L.STRUCT_KILN and inv[i, R["pottery"]] >= 1:
            inv[i, R["pottery"]] -= 1                   # a ceramic pipe (tuyere) to force air in
            w.struct_minerals[cy, cx] += 1
            w.struct[cy, cx] = L.STRUCT_FURNACE
            w.struct_hp[cy, cx] = 3.0
            sim.history.event(sim, "built", int(pop.id[i]), -1, cx, cy, dict(structure="furnace"))
            done = "furnace"
        # --- pyrotechnology: outcome decided by the temperature reached ----------------
        elif sk[i, S["charcoal"]] > 0.1 and inv[i, R["wood"]] >= 2 and setup in (CH.KILN, CH.FURNACE) \
                and inv[i, R["charcoal"]] < lim[R["charcoal"]]:
            inv[i, R["wood"]] -= 2
            inv[i, R["charcoal"]] += 1
            w.ledger.heat_out += 2 * CH.WOOD_ENERGY - CH.CHARCOAL_ENERGY
            done = "charcoal"
        elif sk[i, S["smelting"]] > 0.1 and inv[i, R["metal"]] < 1 and setup is not None and fuel != "none" \
                and ((inv[i, R["ore"]] >= 1 and T >= CH.T_IRON) or (inv[i, R["copper_ore"]] >= 1 and T >= CH.T_COPPER)):
            kind = "iron" if inv[i, R["ore"]] >= 1 and T >= CH.T_IRON else "copper"
            inv[i, R["ore" if kind == "iron" else "copper_ore"]] -= 1
            inv[i, R["metal"]] = 1
            pop.metal_q[i] = CH.METALS[kind]["effect"]
            pop.metal_dur[i] = 1.0
            burn_fuel(sim, i, fuel)
            done = "metal"
            sim.analytics.first(sim, "metal:" + kind, f"First {kind} smelted (fire reached {T:.0f} C)")
        elif sk[i, S["glassmaking"]] > 0.1 and inv[i, R["glass"]] < 1 and T >= CH.T_GLASS_FLUXED and fuel != "none" \
                and (inv[i, R["sand"]] >= 1 or w.sand_amt[cy, cx] >= 1) and (inv[i, R["salt"]] >= 1 or w.salt_amt[cy, cx] >= 1):
            purity = float(pop.sand_q[i]) if inv[i, R["sand"]] >= 1 else float(w.sand_purity[cy, cx])
            _take(w.sand_amt, inv, i, cy, cx, "sand")
            _take(w.salt_amt, inv, i, cy, cx, "salt")
            inv[i, R["glass"]] = 1
            w.debris[cy, cx] += 1                       # the flux ends up as slag
            pop.glass_q[i] = CH.glass_clarity(purity)
            burn_fuel(sim, i, fuel)
            done = "glass"
        elif sk[i, S["pottery"]] > 0.1 and inv[i, R["pottery"]] < 1 and T >= CH.T_CERAMIC and fuel != "none" \
                and (inv[i, R["clay"]] >= 1 or w.clay_amt[cy, cx] >= 1):
            _take(w.clay_amt, inv, i, cy, cx, "clay")
            inv[i, R["pottery"]] = 1
            burn_fuel(sim, i, fuel)
            done = "pottery"
        elif sk[i, S["knap_tool"]] > 0.1 and inv[i, R["tool"]] < 1 and mat[OBJ["stone2"]]:
            rock = _stone_rock(sim, i, cy, cx)
            _take_stone(w, inv, i, cy, cx, 2)
            inv[i, R["tool"]] = 1
            pop.tool_q[i] = CH.stone_tool_effect(rock)
            pop.tool_dur[i] = 1.0
            done = "tool"
        # --- land use -------------------------------------------------------------
        elif sk[i, S["cultivate"]] > 0.1 and land and w.dug[cy, cx] and w.struct[cy, cx] == 0 and pop.energy[i] > 0.4 * cap[i]:
            crops = np.array([L.BERRY, L.NUT_TREE, L.TUBER])
            best = int(crops[np.argmax(pop.food_value[i, crops])])
            seed = 0.08
            pop.energy[i] -= seed
            w.ptype[cy, cx] = best
            w.kmask[cy, cx] = 1.5 * (1.3 if sim.civ.knows_layer(sim, i, "celestial") else 1.0)
            w.biomass[cy, cx] += seed
            _build(sim, i, cy, cx, L.STRUCT_FARM, hp=1.0)
            done = "farm"
            used_tool.append(i)
        elif sk[i, S["dig"]] > 0.1 and land and not w.dug[cy, cx]:
            w.dug[cy, cx] = True
            for field, res in ((w.ore, "ore"), (w.copper, "copper_ore")):
                if field[cy, cx] >= 1 and inv[i, R[res]] < lim[R[res]]:
                    field[cy, cx] -= 1
                    inv[i, R[res]] += 1
                    sim.civ.discover_layer(sim, "subterranean", i)
                    sim.analytics.first(sim, "ore_found", "Ore discovered underground by digging")
            if w.clay_amt[cy, cx] >= 1 and inv[i, R["clay"]] < lim[R["clay"]]:
                w.clay_amt[cy, cx] -= 1
                inv[i, R["clay"]] += 1
            done = "dig"
            used_tool.append(i)
        elif sk[i, S["storage"]] > 0.1 and inv[i, R["food"]] >= 0.2 and (store := _near_store(w, cy, cx)) is not None \
                and w.store_food[store] < L.STORE_CAPACITY:
            amt = float(min(inv[i, R["food"]], L.STORE_CAPACITY - w.store_food[store]))
            inv[i, R["food"]] -= amt
            w.store_food[store] += amt
            pop.food_mem[i] = (store[1] + 0.5, store[0] + 0.5, 1.0)   # remember where the reserve is
            done = "deposit"
            sim.analytics.first(sim, "food_reserve", "Food was put aside in a storage pit for later")
        elif sk[i, S["storage"]] > 0.1 and empty and _near_struct(w, cy, cx, L.STRUCT_STORE, 4) == 0 \
                and (pop.group[i] >= 0 or _near_struct(w, cy, cx, L.STRUCT_SHELTER, 3) > 0 or w.river[cy, cx] or pc.water_near[i]):
            _build(sim, i, cy, cx, L.STRUCT_STORE, hp=1.0)
            done = "store"
            used_tool.append(i)
        elif (sk[i, S["storage"]] > 0.1 or (sk[i, S["pottery"]] > 0.1 and inv[i, R["pottery"]] >= 1)) \
                and inv[i, R["food"]] < (lim[R["food"]] if inv[i, R["pottery"]] >= 1 else 1.0) \
                and pop.energy[i] > 0.6 * cap[i] and w.biomass[cy, cx] > 0.3:
            take = min(0.3, float(w.biomass[cy, cx]) - 0.1)
            w.biomass[cy, cx] -= take
            inv[i, R["food"]] += take
            done = "store_food"
        elif pop.beliefs[i, 2] > 0.5 and pop.group[i] >= 0 and inv[i, R["wood"]] >= 1 and empty and _near_struct(w, cy, cx, L.STRUCT_SHRINE, 5) == 0:
            inv[i, R["wood"]] -= 1
            w.struct_energy[cy, cx] += CH.WOOD_ENERGY
            _build(sim, i, cy, cx, L.STRUCT_SHRINE, hp=1.0)
            done = "shrine"
        # --- gathering (from finite deposits) ------------------------------------
        elif w.stone[cy, cx] >= 1 and inv[i, R["stone"]] < lim[R["stone"]] and \
                (sk[i, S["knap_tool"]] > 0.1 or sk[i, S["crack_nuts"]] > 0.1 or sk[i, S["kiln"]] > 0.1):
            w.stone[cy, cx] -= 1
            inv[i, R["stone"]] += 1
            pop.stone_rock[i] = w.rock[cy, cx]
            done = "gather_stone"
        elif mat[OBJ["nut"]] and w.biomass[cy, cx] > 0.4 and inv[i, R["wood"]] < lim[R["wood"]] and \
                (sk[i, S["shelter"]] > 0.1 or sk[i, S["fire"]] > 0.1 or sk[i, S["charcoal"]] > 0.1
                 or (sk[i, S["knap_tool"]] > 0.1 and inv[i, R["tool"]] >= 1)):   # cutting sticks needs an edge
            w.biomass[cy, cx] -= CH.WOOD_ENERGY          # the wood keeps the energy of the biomass
            n0 = float(inv[i, R["wood"]])
            pop.wood_moist[i] = (pop.wood_moist[i] * n0 + CH.WOOD[CH.E["H2O"]]) / (n0 + 1)
            inv[i, R["wood"]] += 1
            done = "gather_wood"
            used_tool.append(i)
        elif w.sand_amt[cy, cx] >= 1 and sk[i, S["glassmaking"]] > 0.1 and inv[i, R["sand"]] < lim[R["sand"]]:
            w.sand_amt[cy, cx] -= 1
            inv[i, R["sand"]] += 1
            pop.sand_q[i] = w.sand_purity[cy, cx]
            done = "gather_sand"
        elif w.salt_amt[cy, cx] >= 1 and (sk[i, S["glassmaking"]] > 0.1 or sk[i, S["storage"]] > 0.1) and inv[i, R["salt"]] < lim[R["salt"]]:
            w.salt_amt[cy, cx] -= 1
            inv[i, R["salt"]] += 1
            done = "gather_salt"
        elif w.clay_amt[cy, cx] >= 1 and (sk[i, S["pottery"]] > 0.1 or sk[i, S["kiln"]] > 0.1) and inv[i, R["clay"]] < lim[R["clay"]]:
            w.clay_amt[cy, cx] -= 1
            inv[i, R["clay"]] += 1
            done = "gather_clay"
        if done:
            did += 1
            reward[i] += 0.05
            for s_ in _USES.get(done, ()):
                sk[i, S[s_]] = min(1.0, sk[i, S[s_]] + 0.01)
            if done in ("hearth", "shelter", "farm", "shrine", "store", "kiln", "furnace"):
                MEM.record(pop, np.array([i]), MEM.M["built"], 0.4, 0, tick)
                MEM.consolidate(pop, np.array([i]), MEM.M["built"], 0.5, 0, tick, detail=f"built a {done}")
                sim.analytics.first(sim, "built:" + done, f"First {done} built")
    wear_tools(sim, used_tool)
    w.clay = w.clay_amt >= 1
    w.sand = w.sand_amt >= 1
    return did


_USES = {"hearth": ["fire"], "shelter": ["shelter"], "tool": ["knap_tool"], "metal": ["smelting"],
         "glass": ["glassmaking"], "pottery": ["pottery"], "farm": ["cultivate"], "dig": ["dig"],
         "store_food": ["storage"], "deposit": ["storage"], "store": ["storage"], "kiln": ["kiln"],
         "furnace": ["furnace"], "charcoal": ["charcoal"]}


def _take(field, inv, i, cy, cx, res):
    """Use one unit of a material from the inventory, else from the deposit here."""
    if inv[i, L.R[res]] >= 1:
        inv[i, L.R[res]] -= 1
    else:
        field[cy, cx] -= 1


def _take_stone(w, inv, i, cy, cx, k):
    from_inv = min(inv[i, L.R["stone"]], k)
    inv[i, L.R["stone"]] -= from_inv
    rest = k - from_inv
    w.stone[cy, cx] = w.stone[cy, cx] - rest


def _near_store(w, cy, cx):
    """The closest storage pit within one cell, as (y, x), or None."""
    y0, y1 = max(0, cy - 1), min(w.H, cy + 2)
    x0, x1 = max(0, cx - 1), min(w.W, cx + 2)
    ys, xs = np.nonzero(w.struct[y0:y1, x0:x1] == L.STRUCT_STORE)
    if len(ys) == 0:
        return None
    return (int(ys[0] + y0), int(xs[0] + x0))


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
