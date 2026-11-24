"""Stage 3: individual memory and learning that really changes behaviour."""
import numpy as np

from genesis import Simulation
from genesis.biology import actions
from genesis.intelligence import brain
from genesis.intelligence.perception import perceive
from genesis.memory import memory as MEM
from genesis.physics import laws as L
from tests.conftest import small_cfg

A = L.A


def _land_cell(s, k=0):
    ys, xs = np.nonzero(s.world.land & (s.world.height_above < 0.4))
    return int(xs[k * 37 % len(xs)]), int(ys[k * 37 % len(ys)])


def _make_patch(s, x, y, ptype, bio=0.8):
    w = s.world
    w.ptype[y, x] = ptype
    w.kmask[y, x] = 1.0
    w.biomass[y, x] = bio


def _eat_once(s, i):
    pc = perceive(s)
    act = np.full(len(s.pop), A["rest"])
    act[i] = A["eat"]
    gain = np.zeros(len(s.pop), np.float32)
    return actions.eat(s, pc, act, gain)[i], gain[i]


def _p_eat(s, i):
    pc = perceive(s)
    logits = brain.forward(s.pop.weights, pc.inputs) + s.pop.habit
    z = np.exp(logits[i] - logits[i].max())
    return float(z[A["eat"]] / z.sum()), pc


def test_taste_aversion_is_learned():
    s = Simulation(small_cfg(initial_population=6))
    s.step()
    i = 0
    x, y = _land_cell(s)
    _make_patch(s, x, y, L.NIGHTSHADE)
    s.pop.x[i], s.pop.y[i] = x + 0.5, y + 0.5
    s.pop.energy[i] = 0.5
    v0 = float(s.pop.food_value[i, L.NIGHTSHADE])
    h0 = float(s.pop.health[i])
    for _ in range(4):
        _make_patch(s, x, y, L.NIGHTSHADE)
        s.pop.energy[i] = 0.5
        _eat_once(s, i)
    assert s.pop.food_value[i, L.NIGHTSHADE] < v0 - 0.2, "eating a toxic plant must lower its learned value"
    assert s.pop.food_value[i, L.NIGHTSHADE] < s.pop.food_value[i, L.BERRY]
    assert any(m["type"] == "poisoned" for m in MEM.long_term(s.pop, int(s.pop.id[i]), s.clock.tick))


def test_learning_changes_behaviour_of_identical_twins():
    s = Simulation(small_cfg(initial_population=6))
    s.step()
    p = s.pop
    for f in ("genome", "weights", "food_value", "habit", "personality", "emotion"):
        getattr(p, f)[1] = getattr(p, f)[0]
    x, y = _land_cell(s)
    # twin 0 is poisoned a few times; twin 1 never is
    p.x[0], p.y[0] = x + 0.5, y + 0.5
    for _ in range(4):
        _make_patch(s, x, y, L.NIGHTSHADE)
        p.energy[0] = 0.5
        _eat_once(s, 0)
    # now both stand on identical nightshade patches with identical bodily state
    x2, y2 = _land_cell(s, 5)
    for k, (cx, cy) in enumerate([(x, y), (x2, y2)]):
        _make_patch(s, cx, cy, L.NIGHTSHADE)
        p.x[k], p.y[k] = cx + 0.5, cy + 0.5
        p.energy[k], p.health[k], p.hydration[k] = 0.8, 1.0, 1.0
    p.emotion[0] = p.emotion[1] = 0
    pe0, _ = _p_eat(s, 0)
    pe1, _ = _p_eat(s, 1)
    assert pe0 < pe1 * 0.8, (pe0, pe1)


def test_habits_form_from_reward():
    s = Simulation(small_cfg(initial_population=4))
    n = len(s.pop)
    a = np.full(n, A["work"])
    h0 = s.pop.habit[:, A["work"]].copy()
    for _ in range(30):
        brain.learn_habits(s, a, np.full(n, 0.5, np.float32))
    assert (s.pop.habit[:, A["work"]] > h0).all()
    for _ in range(80):
        brain.learn_habits(s, a, np.full(n, -0.5, np.float32))
    assert (s.pop.habit[:, A["work"]] < h0).all()


def test_short_and_long_term_memory():
    s = Simulation(small_cfg(initial_population=3))
    p = s.pop
    for t in range(12):
        MEM.record(p, np.array([0]), MEM.M["ate"], 0.1, t, t)
    stm = MEM.short_term(p, 0)
    assert len(stm) == 8 and stm[0]["tick"] == 11, "ring buffer keeps the most recent events"
    p.genome[0, 13] = 0.0  # tiny brain -> small long-term capacity
    for t in range(30):
        MEM.consolidate(p, np.array([0]), MEM.M["attacked"], -1.0, -1, t, detail="x", force=True)
    assert len(p.ltm[int(p.id[0])]) <= 6, "long-term memory capacity depends on brain size"
    # weak events are not consolidated
    MEM.consolidate(p, np.array([1]), MEM.M["ate"], 0.01, -1, 1)
    assert int(p.id[1]) not in p.ltm


def test_place_memory_guides_return():
    s = Simulation(small_cfg(initial_population=4))
    s.step()
    p = s.pop
    p.food_mem[0] = (p.x[0] + 4, p.y[0], 1.0)
    d0 = np.hypot(p.food_mem[0, 0] - p.x[0], p.food_mem[0, 1] - p.y[0])
    pc = perceive(s)
    act = np.full(len(p), A["rest"]); act[0] = A["recall"]
    actions.move(s, pc, act)
    d1 = np.hypot(p.food_mem[0, 0] - p.x[0], p.food_mem[0, 1] - p.y[0])
    assert d1 < d0
