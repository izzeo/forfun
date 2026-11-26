"""Creator mode: the researcher intervenes in the world.

Every intervention is (1) recorded in the world's history as an act of the
creator, so nobody mistakes the result for unaided emergence, and (2) booked
in the ledgers: energy or matter brought from outside goes to
``ledger.creator_in`` / ``world.creator_minerals`` so the conservation laws
still balance. Inhabitants who witness a sign without a visible cause treat it
like any other unexplained event: belief in unseen agency grows.

Note: interventions are not replayed when history is rewound.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..biology.population import make_name
from ..intelligence import brain
from ..memory import memory as MEM
from ..physics import laws as L

LAWS = {  # name: (config attribute, min, max)
    "plant_growth": ("plant_growth", 0.0, 0.5),
    "disease_virulence": ("pathogen_virulence", 0.0, 3.0),
    "storm_rate": ("storm_rate", 0.0, 0.05),
    "eclipse_rate": ("eclipse_rate", 0.0, 0.01),
}
EVENTS = ["storm", "drought", "rain", "wildfire", "plague", "abundance", "cold", "warm", "eclipse"]


class CreatorError(ValueError):
    pass


def _cells(w, x, y, r):
    yy, xx = np.mgrid[0:w.H, 0:w.W]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r * r


def _witnesses(sim, x, y, r):
    pop = sim.pop
    if not len(pop):
        return np.zeros(0, np.int64)
    return np.nonzero(np.hypot(pop.x - x, pop.y - y) <= r + 4)[0]


def _record(sim, kind, x, y, data):
    sim.history.event(sim, "intervention", -1, -1, x, y, dict(kind=kind, **data))
    sim.stats_counter["interventions"] += 1
    sim.interventions.append(dict(tick=sim.clock.tick, kind=kind, x=round(float(x), 1), y=round(float(y), 1), **data))
    sim.analytics.first(sim, "intervention", "The creator intervened in the world for the first time")


def _sign(sim, idx, harmful: bool, what: str):
    """Witnesses of a sign without visible cause."""
    if len(idx) == 0:
        return
    pop = sim.pop
    if harmful:
        sim.culture.unexplained_harm(sim, idx, 0.8, "creator")
    else:
        capable = idx[pop.genome[idx, GN.T["brain"]] > 0.25]
        pop.beliefs[capable, 0] = np.minimum(1, pop.beliefs[capable, 0] + 0.08)
        pop.norms[capable, 6] = np.minimum(1, pop.norms[capable, 6] + 0.02)     # gratitude
    MEM.consolidate(pop, idx, MEM.M["eclipse"], 0.8, -1, sim.clock.tick, detail=f"witnessed {what}", force=True)


def set_law(sim, name: str, value: float) -> dict:
    if name == "temperature":
        v = float(np.clip(value, -20, 20))
        sim.world.creator_temp = v
        _record(sim, "law", 0, 0, dict(law="temperature", value=v))
        return dict(law=name, value=v)
    if name not in LAWS:
        raise CreatorError(f"unknown law {name!r}; options: {sorted(LAWS) + ['temperature']}")
    attr, lo, hi = LAWS[name]
    v = float(np.clip(value, lo, hi))
    setattr(sim.cfg, attr, v)
    _record(sim, "law", 0, 0, dict(law=name, value=v))
    return dict(law=name, value=v)


def event(sim, kind: str, x: float, y: float, r: float = 6) -> dict:
    w, rng = sim.world, sim.rng
    x, y, r = float(x), float(y), float(np.clip(r, 1, 30))
    if kind not in EVENTS:
        raise CreatorError(f"unknown event {kind!r}; options: {EVENTS}")
    m = _cells(w, x, y, r)
    wit = _witnesses(sim, x, y, r)
    info = dict(radius=r)
    if kind == "storm":
        w.storms.append(dict(id=10_000_000 + sim.stats_counter["interventions"], cy=int(y), cx=int(x), r=int(r),
                             intensity=1.5))
        sim.scheduler.at(sim.clock.tick + 24, "storm_end", id=w.storms[-1]["id"])
        _sign(sim, wit, True, "a storm sent from nowhere")
    elif kind == "drought":
        w.moisture[m] *= 0.1
        w.base_moist[m] *= 0.5
        _sign(sim, wit, True, "the land dry up")
    elif kind == "rain":
        w.moisture[m] = 1.0
        w.clouds[m] = np.maximum(w.clouds[m], 0.9)
        _sign(sim, wit, False, "rain falling as if called")
    elif kind == "wildfire":
        ys, xs = np.nonzero(m)
        lit = sum(w.ignite(int(a), int(b)) for a, b in zip(ys, xs) if rng.random() < 0.3)
        info["cells_lit"] = int(lit)
        _sign(sim, wit, True, "fire fall from the sky")
    elif kind == "plague":
        w.pathogens[m] = np.maximum(w.pathogens[m], 0.9)
        w.virulence[m] = np.maximum(w.virulence[m], 0.8)
        _sign(sim, wit, True, "a sickness come over the land")
    elif kind == "abundance":
        K = w.carrying()
        add = np.where(m, np.clip(K - w.biomass, 0, None), 0).astype(np.float32)
        w.biomass += add
        w.ledger.creator_in = getattr(w.ledger, "creator_in", 0.0) + float(add.sum())
        info["energy_added"] = round(float(add.sum()), 2)
        _sign(sim, wit, False, "plants grow overnight")
    elif kind in ("cold", "warm"):
        w.creator_temp = float(getattr(w, "creator_temp", 0.0)) + (-5.0 if kind == "cold" else 5.0)
        info["global_offset"] = w.creator_temp
        _sign(sim, _witnesses(sim, w.W / 2, w.H / 2, max(w.W, w.H)), kind == "cold", f"the world turn {kind}")
    elif kind == "eclipse":
        sim._eclipse()
    _record(sim, kind, x, y, info)
    return dict(event=kind, **info, witnesses=int(len(wit)))


def spawn(sim, x: float, y: float, n: int = 10, kind: str = "advanced", copy_of: int | None = None) -> dict:
    """Create new organisms out of nothing (energy booked as coming from the creator)."""
    pop, w, rng = sim.pop, sim.world, sim.rng
    n = int(np.clip(n, 1, 200))
    if copy_of is not None:
        i = pop.index_of(int(copy_of))
        if i is None:
            raise CreatorError("no living organism with that id to copy")
        g = np.repeat(pop.genome[i:i + 1], n, axis=0)
        wts = np.repeat(pop.weights[i:i + 1], n, axis=0)
    else:
        g = GN.advanced_genomes(rng, n) if kind == "advanced" else GN.random_genomes(rng, n)
        wts = brain.instinct_weights(rng, n)
    cy, cx = int(np.clip(y, 0, w.H - 1)), int(np.clip(x, 0, w.W - 1))
    if w.water[cy, cx] and g[:, GN.T["aquatic"]].mean() < 0.5:
        raise CreatorError("that is water; land creatures would drown")
    energy = 0.6 * P.capacity(g)
    max_age = P.max_age(g) * rng.uniform(0.85, 1.15, n)
    pers = np.clip(np.stack([g[:, GN.T["curiosity"]], g[:, GN.T["patience"]], g[:, GN.T["sociality"]],
                             1 - g[:, GN.T["aggression"]], g[:, GN.T["fear_sens"]]], axis=1) + rng.normal(0, 0.12, (n, 5)), 0, 1)
    ids = pop.add(n, x=np.clip(x + rng.normal(0, 1.0, n), 0, w.W - 1e-3), y=np.clip(y + rng.normal(0, 1.0, n), 0, w.H - 1e-3),
                  heading=rng.uniform(0, 2 * np.pi, n), energy=energy, max_age=max_age,
                  age=(0.2 * max_age).astype(np.int32), genome=g, weights=wts,
                  food_value=g[:, GN.FOOD_PREF0:GN.FOOD_PREF0 + L.N_PLANTS], personality=pers,
                  birth_tick=sim.clock.tick, sex=(rng.random(n) < 0.5).astype(np.int8),
                  kin_aversion=rng.uniform(0.1, 0.5, n).astype(np.float32))
    for a in ids:
        pop.names[int(a)] = make_name(int(a))
    new_idx = np.arange(len(pop) - n, len(pop))
    w.ledger.creator_in = getattr(w.ledger, "creator_in", 0.0) + float((energy + pop.body_mass()[new_idx]).sum())
    sim.history.births(sim, new_idx)
    MEM.consolidate(pop, new_idx, MEM.M["born"], 1.0, -1, sim.clock.tick, detail="came into being", force=True)
    _record(sim, "spawn", x, y, dict(n=n, creature="copy" if copy_of is not None else kind, ids=[int(a) for a in ids[:20]]))
    return dict(spawned=n, ids=[int(a) for a in ids])


def act_on(sim, agent_id: int, what: str, skill: str | None = None) -> dict:
    pop, w = sim.pop, sim.world
    i = pop.index_of(int(agent_id))
    if i is None:
        raise CreatorError("no living organism with that id")
    x, y = float(pop.x[i]), float(pop.y[i])
    if what == "heal":
        cap = float(P.capacity(pop.genome[i:i + 1])[0])
        add = max(0.0, cap - float(pop.energy[i]))
        pop.energy[i] = cap
        pop.health[i] = 1.0
        pop.hydration[i] = 1.0
        pop.infection[i] = 0.0
        w.ledger.creator_in = getattr(w.ledger, "creator_in", 0.0) + add
        _sign(sim, np.array([i]), False, "itself healed without cause")
        _record(sim, "heal", x, y, dict(agent=int(agent_id)))
    elif what == "smite":
        cause = np.full(len(pop), -1, np.int64)
        cause[i] = L.DEATH_CAUSES.index("act_of_creator")
        others = _witnesses(sim, x, y, 2)
        others = others[others != i]
        if len(sim.consumed) != len(pop):
            sim.consumed = np.zeros(len(pop), bool)
        from ..biology import lifecycle
        _record(sim, "smite", x, y, dict(agent=int(agent_id)))
        lifecycle.bury(sim, cause, {})
        others = others - (others > i)          # indices shift after removal
        _sign(sim, others, True, "a companion struck down by no one")
    elif what == "reveal":
        if skill not in L.SK:
            raise CreatorError(f"unknown skill {skill!r}; options: {L.SKILLS}")
        from ..intelligence.skills import learn_skill
        learn_skill(sim, i, L.SK[skill], 0.6, source="experiment", via="revelation")
        MEM.consolidate(pop, np.array([i]), MEM.M["discovered"], 1.0, -1, sim.clock.tick,
                        detail=f"suddenly knew how to {skill.replace('_', ' ')}", force=True)
        _record(sim, "reveal", x, y, dict(agent=int(agent_id), skill=skill))
    else:
        raise CreatorError("what must be heal, smite or reveal")
    return dict(ok=True, agent=int(agent_id), act=what)


def place(sim, x: float, y: float, material: str, amount: float = 3) -> dict:
    w = sim.world
    cy, cx = int(np.clip(y, 0, w.H - 1)), int(np.clip(x, 0, w.W - 1))
    fields = dict(stone=w.stone, clay=w.clay_amt, sand=w.sand_amt, ore=w.ore, copper=w.copper, salt=w.salt_amt)
    if material not in fields:
        raise CreatorError(f"unknown material {material!r}; options: {sorted(fields)}")
    amount = float(np.clip(amount, 0, 20))
    fields[material][cy, cx] += amount
    w.creator_minerals = float(getattr(w, "creator_minerals", 0.0)) + amount
    w.clay = w.clay_amt >= 1
    w.sand = w.sand_amt >= 1
    _record(sim, "place", cx, cy, dict(material=material, amount=amount))
    return dict(placed=material, amount=amount)
