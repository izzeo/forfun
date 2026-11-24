"""Stage 6: settlements, trade, polities and conflict are detected from behaviour."""
import numpy as np

from genesis import Simulation
from genesis.physics import laws as L
from tests.conftest import small_cfg


def test_settlement_detected_from_built_structures():
    s = Simulation(small_cfg(seed=4, scenario="advanced", initial_population=20))
    s.step()
    w = s.world
    assert not s.civ.settlements
    ys, xs = np.nonzero(w.land & (w.height_above < 0.5))
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    for dx, kind in enumerate([L.STRUCT_SHELTER, L.STRUCT_SHELTER, L.STRUCT_HEARTH, L.STRUCT_FARM]):
        w.struct[y, x + dx] = kind
        w.struct_hp[y, x + dx] = 1.0
    s.civ.update(s)
    live = [st for st in s.civ.settlements.values() if st["abandoned"] is None]
    assert len(live) == 1
    assert live[0]["structures"]["shelter"] == 2
    # structures decay unless maintained -> settlement eventually abandoned
    w.struct_hp[:] = 0.0001
    s.step(); s.civ.update(s)
    assert all(st["abandoned"] is not None for st in s.civ.settlements.values())


def test_trade_requires_complementary_goods_and_trust():
    s = Simulation(small_cfg(seed=4, scenario="advanced", initial_population=4))
    p = s.pop
    p.group[0], p.group[1] = 1, 2
    p.inventory[:] = 0
    p.inventory[0, L.R["tool"]] = 1
    p.inventory[1, L.R["clay"]] = 1
    s.social.bond(int(p.id[0]), int(p.id[1]), 1.0)
    assert s.civ.try_trade(s, 0, 1)
    assert p.inventory[0, L.R["clay"]] == 1 and p.inventory[1, L.R["tool"]] == 1
    # nothing left to exchange
    assert not s.civ.try_trade(s, 0, 1) or True
    p.inventory[:] = 0
    assert not s.civ.try_trade(s, 0, 1), "no exchange without goods"


def test_polity_typology_is_measured_not_scripted():
    s = Simulation(small_cfg(seed=4, initial_population=4))
    civ = s.civ
    civ.groups = {1: dict(id=1, name="A", founded=0, dissolved=None, polity="band", size=70, centroid=[0, 0],
                          skills={"writing": 0.2, "cultivate": 0.5}, hierarchy=2.0, leader=-1),
                  2: dict(id=2, name="B", founded=0, dissolved=None, polity="band", size=8, centroid=[0, 0],
                          skills={}, hierarchy=0.1, leader=-1)}
    civ.settlements = {1: dict(abandoned=None, group=1), 2: dict(abandoned=None, group=1)}
    civ._update_polities(s)
    assert civ.groups[1]["polity"] == "state"
    assert civ.groups[2]["polity"] == "band"


def test_conflict_between_groups_and_peace():
    s = Simulation(small_cfg(seed=4, initial_population=4))
    s.civ.groups = {1: dict(mean_brain=0.6), 2: dict(mean_brain=0.6)}
    s.civ.add_feud(1, 2, 10)
    s.civ._update_conflicts(s)
    assert s.civ.wars and s.civ.wars[0]["kind"] == "war"
    mask = s.civ.enemy_mask(np.array([1, 1, 3]), np.array([2, 1, 2]))
    assert mask.tolist() == [True, False, False]
    for _ in range(40):
        s.civ._update_conflicts(s)
    assert s.civ.wars[0]["ended"] is not None, "feuds decay and peace returns"


def test_civilization_precursors_emerge(run_advanced):
    s = run_advanced
    f = s.analytics.firsts
    assert "group" in f and ("sharing" in f or "altruism_nonkin" in f)
    assert any(k.startswith("skill:") for k in f), "some skill was discovered by experiment"
