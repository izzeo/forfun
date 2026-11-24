"""Stage 8: individuality, emotions, life stages and moral education."""
import numpy as np

from genesis import Simulation
from genesis.biology import genome as GN, lifecycle
from genesis.civilization.morality import FRAMEWORKS, MoralSystem
from genesis.intelligence import brain
from genesis.intelligence.perception import perceive
from genesis.physics import laws as L
from tests.conftest import small_cfg


def test_personalities_are_unique(run_advanced):
    pers = run_advanced.pop.personality
    assert len(np.unique(np.round(pers, 6), axis=0)) == len(pers)


def test_grief_after_losing_a_bonded_companion():
    s = Simulation(small_cfg(seed=3, scenario="advanced", initial_population=6))
    s.step()
    p = s.pop
    a, b = int(p.id[0]), int(p.id[1])
    s.social.bond(a, b, 0.9)
    p.emotion[1, 3] = 0
    cause = np.full(len(p), -1)
    cause[0] = lifecycle.C["predation"]
    s.consumed = np.zeros(len(p), bool)
    lifecycle.bury(s, cause, {})
    k = p.index_of(b)
    assert p.emotion[k, 3] > 0.5
    assert any(m["type"] == "lost_kin" for m in p.ltm[b])


def test_life_stages_constrain_behaviour():
    s = Simulation(small_cfg(seed=3, scenario="advanced", initial_population=6))
    s.step()
    p = s.pop
    p.age[0] = 1  # infant
    st = brain.life_stage(p)
    assert st[0] == 0
    pc = perceive(s)
    for _ in range(50):
        act = brain.choose(s, pc)
        assert act[0] != L.A["mate"] and act[0] != L.A["work"]


def test_emotions_respond_to_threat():
    s = Simulation(small_cfg(seed=3, initial_population=6))
    s.step()
    pc = perceive(s)
    pc.threat[:] = 0
    pc.threat[0] = 0.9
    s.pop.emotion[:] = 0
    s.psyche.update(s, pc, np.zeros(len(s.pop), np.float32))
    assert s.pop.emotion[0, 0] > 0.3 and s.pop.emotion[1, 0] == 0


def test_moral_frameworks_are_optional_and_comparable():
    assert FRAMEWORKS["none"]["ideal"] is None
    assert "quran_inspired" in FRAMEWORKS
    q = Simulation(small_cfg(seed=21, scenario="advanced", initial_population=80, moral_framework="quran_inspired", moral_education=0.8))
    e = Simulation(small_cfg(seed=21, scenario="advanced", initial_population=80, moral_framework="egoist", moral_education=0.8))
    n = Simulation(small_cfg(seed=21, scenario="advanced", initial_population=80, moral_framework="none"))
    # identical worlds and founders
    assert np.array_equal(q.world.elev, e.world.elev) and np.array_equal(q.pop.genome, e.pop.genome)
    for s in (q, e, n):
        s.run(500)
    mercy = lambda s: float(s.pop.norms[:, 1].mean()) if len(s.pop) else 0.0
    born = lambda s: s.pop.generation > 0
    if born(q).any() and born(e).any():
        assert q.pop.norms[born(q), 1].mean() > e.pop.norms[born(e), 1].mean()
    assert abs(mercy(n)) < 0.3, "without education norms only drift from experience"


def test_moral_norms_change_sharing_willingness():
    s = Simulation(small_cfg(seed=3, scenario="advanced", initial_population=4))
    MoralSystem.experience(s.pop, [0], "mercy", 0.9)
    assert s.pop.norms[0, 1] >= 0.9 - 1e-6
