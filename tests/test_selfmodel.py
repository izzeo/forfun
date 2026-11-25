"""Self-model: self-efficacy, learned helplessness and its recovery."""
import numpy as np

from genesis import Simulation
from genesis.intelligence import brain, selfmodel
from genesis.intelligence.perception import perceive
from genesis.physics import laws as L
from tests.conftest import small_cfg

A = L.A
D = selfmodel.D


def _sim(n=6):
    s = Simulation(small_cfg(seed=4, scenario="advanced", initial_population=n))
    s.step()
    return s


def _upd(s, act=None, attacks=(), gain=None, r_exp=None, r_work=None, harm=None, support=None):
    n = len(s.pop)
    z = np.zeros(n, np.float32)
    act = np.full(n, A["rest"]) if act is None else act
    selfmodel.update(s, act, list(attacks), z if gain is None else gain, z if r_exp is None else r_exp,
                     z if r_work is None else r_work, z if harm is None else harm, z if support is None else support)


def test_beliefs_track_own_outcomes():
    s = _sim()
    act = np.full(len(s.pop), A["rest"]); act[0] = A["attack"]; act[1] = A["attack"]
    for _ in range(60):
        _upd(s, act=act, attacks=[(0, 2, True, 0), (1, 3, False, 0)])
    e = s.pop.self_eff
    assert e[0, D["fight"]] > 0.8, "winning fights builds confidence"
    assert e[1, D["fight"]] < 0.2, "losing builds doubt"
    assert e[2, D["fight"]] < 0.5, "being overpowered also teaches"


def test_self_belief_changes_what_is_attempted():
    s = _sim()
    p = s.pop
    for f in ("genome", "weights", "habit", "personality", "emotion"):
        getattr(p, f)[1] = getattr(p, f)[0]
    p.self_eff[0, D["fight"]] = 0.95
    p.self_eff[1, D["fight"]] = 0.05
    pc = perceive(s)
    pc.inputs[1] = pc.inputs[0]
    logits = brain.forward(p.weights, pc.inputs) + p.habit
    selfmodel.bias(p, logits)
    assert logits[0, A["attack"]] > logits[1, A["attack"]] + 1.0


def test_learned_helplessness_and_recovery():
    s = _sim()
    p = s.pop
    a0 = float(p.agency[0])
    harm = np.zeros(len(p), np.float32); harm[0] = 0.3
    for _ in range(80):
        _upd(s, harm=harm)                       # repeated harm it cannot avoid
    assert p.agency[0] < selfmodel.HELPLESS < a0
    assert "learned_helplessness" in s.analytics.firsts
    pc = perceive(s)
    logits = brain.forward(p.weights, pc.inputs) + p.habit
    before = logits.copy()
    selfmodel.bias(p, logits)
    assert logits[0, A["experiment"]] - before[0, A["experiment"]] < logits[1, A["experiment"]] - before[1, A["experiment"]], \
        "the helpless stop trying"
    support = np.zeros(len(p), np.float32); support[0] = 1.5
    for _ in range(200):
        _upd(s, support=support)                 # others keep helping
    assert p.agency[0] >= 0.5
    assert "overcame_helplessness" in s.analytics.firsts


def test_vicarious_learning_from_similar_others():
    s = _sim()
    p = s.pop
    p.group[:] = 1
    p.self_eff[0, D["explore"]] = 0.1
    p.self_eff[1, D["explore"]] = 0.9
    for _ in range(50):
        selfmodel.vicarious(s, np.array([0]), np.array([1]))
    assert p.self_eff[0, D["explore"]] > 0.4
    p.group[2] = 2
    p.self_eff[2, D["explore"]] = 0.1
    selfmodel.vicarious(s, np.array([2]), np.array([1]))
    assert p.self_eff[2, D["explore"]] == np.float32(0.1), "only someone 'like me' is a model"


def test_population_is_mostly_not_helpless(run_advanced):
    a = run_advanced.pop.agency
    assert 0.3 < a.mean() < 0.85
    assert (a < selfmodel.HELPLESS).mean() < 0.3
