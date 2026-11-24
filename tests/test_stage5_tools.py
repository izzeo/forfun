"""Stage 5: experimentation, tools and knowledge transmission."""
import numpy as np

from genesis import Simulation
from genesis.biology import actions, genome as GN
from genesis.intelligence import skills
from genesis.intelligence.perception import perceive, OBJ
from genesis.physics import laws as L
from tests.conftest import small_cfg

A = L.A


def _sim(n=6):
    s = Simulation(small_cfg(seed=9, scenario="advanced", initial_population=n))
    s.step()
    s.pop.genome[:, GN.T["brain"]] = 0.9
    s.pop.genome[:, GN.T["curiosity"]] = 0.9
    return s


def _experiment(s, i, times):
    for _ in range(times):
        pc = perceive(s)
        act = np.full(len(s.pop), A["rest"]); act[i] = A["experiment"]
        s.pop.energy[i] = 3.0
        skills.experiment(s, pc, act, np.zeros(len(s.pop), np.float32))


def test_no_discovery_without_the_right_materials():
    s = _sim()
    w, p = s.world, s.pop
    # stand on bare open water-free land with nothing useful: no stone, no trees, no clay/sand
    ys, xs = np.nonzero(w.land & (w.stone == 0) & ~w.clay & ~w.sand & (w.ptype == L.GRASS))
    p.x[0], p.y[0] = xs[0] + 0.5, ys[0] + 0.5
    p.inventory[0] = 0
    _experiment(s, 0, 400)
    assert p.skills[0, L.SK["crack_nuts"]] == 0 and p.skills[0, L.SK["knap_tool"]] == 0
    assert p.skills[0, L.SK["pottery"]] == 0 and p.skills[0, L.SK["fire"]] == 0


def test_discovery_by_experiment_when_materials_present():
    s = _sim()
    w, p = s.world, s.pop
    x, y = 10, 10
    w.water[y, x] = False
    w.ptype[y, x] = L.NUT_TREE; w.kmask[y, x] = 1; w.biomass[y, x] = 1.0
    w.stone[y, x] = 3
    p.x[0], p.y[0] = x + 0.5, y + 0.5
    found = False
    for _ in range(60):
        _experiment(s, 0, 20)
        w.biomass[y, x] = 1.0
        if p.skills[0, L.SK["crack_nuts"]] > 0 or p.skills[0, L.SK["knap_tool"]] > 0:
            found = True
            break
    assert found, "trial and error with stones and nuts should eventually reveal an affordance"
    assert s.civ.skills_known, "the discovery is recorded"


def test_tool_skill_unlocks_energy():
    s = _sim()
    w, p = s.world, s.pop
    x, y = 12, 12
    w.water[y, x] = False
    gains = []
    for skill in (0.0, 1.0):
        w.ptype[y, x] = L.NUT_TREE; w.kmask[y, x] = 1; w.biomass[y, x] = 1.0
        p.x[0], p.y[0] = x + 0.5, y + 0.5
        p.energy[0] = 0.2
        p.skills[0, L.SK["crack_nuts"]] = skill
        pc = perceive(s)
        act = np.full(len(p), A["rest"]); act[0] = A["eat"]
        g = np.zeros(len(p), np.float32)
        actions.eat(s, pc, act, g)
        gains.append(float(g[0]))
    assert gains[1] > 2 * gains[0]


def test_skills_spread_by_imitation():
    s = _sim(4)
    p = s.pop
    p.skills[:] = 0
    p.skills[1, L.SK["knap_tool"]] = 0.9
    I = np.array([0] * 400); J = np.array([1] * 400)
    skills.imitate(s, I, J, np.ones(len(I), np.float32))
    assert p.skills[0, L.SK["knap_tool"]] > 0.1
    assert "skill_transmission" in s.analytics.firsts
