"""Rivers, lakes, food storage and upgrading saves from older versions."""
import pickle
import zlib

import numpy as np

from genesis import Simulation
from genesis.physics import hydrology, laws as L
from genesis.intelligence.perception import perceive
from tests.conftest import small_cfg


def test_rivers_flow_downhill_to_the_sea():
    s = Simulation(small_cfg(seed=2, width=96, height=96, initial_population=0))
    w = s.world
    assert w.river.sum() > 30, "relief produces a river network"
    sea = w.elev < s.cfg.sea_level
    filled = hydrology.priority_flood(w.elev, sea)
    acc, down, _ = hydrology.flow_routing(filled, sea, np.ones_like(w.elev))
    # follow one river from its highest cell: it must end in the sea, never climbing
    ys, xs = np.nonzero(w.river)
    i = int(np.argmax(filled[ys, xs]))
    cell = ys[i] * w.W + xs[i]
    for _ in range(w.H * w.W):
        nxt = down.ravel()[cell]
        if nxt < 0:
            break
        assert filled.ravel()[nxt] < filled.ravel()[cell]
        cell = nxt
    assert sea.ravel()[cell] or down.ravel()[cell] < 0 and sea.ravel()[cell]


def test_river_water_is_drinkable_inland():
    s = Simulation(small_cfg(seed=2, width=96, height=96, initial_population=4))
    s.step()
    w = s.world
    ys, xs = np.nonzero(w.river & (w.dist_water > 0) & ~w.coast | w.river)
    y, x = int(ys[0]), int(xs[0])
    s.pop.x[0], s.pop.y[0] = x + 0.5, y + 0.5
    s.pop.hydration[0] = 0.2
    pc = perceive(s)
    assert pc.water_near[0], "a river cell counts as water to drink"


def test_storage_pit_keeps_food_and_energy_is_conserved():
    s = Simulation(small_cfg(seed=3, scenario="advanced", initial_population=4))
    s.step()
    w, p = s.world, s.pop
    y, x = int(p.y[0]), int(p.x[0])
    w.struct[y, x] = L.STRUCT_STORE
    w.struct_hp[y, x] = 1.0
    w.store_food[y, x] = 2.0
    p.energy[0] = 0.1
    e0 = s.total_energy()
    in0, out0 = w.ledger.solar_in, w.ledger.heat_out
    s._eat_stores()
    assert p.energy[0] > 0.2, "a hungry organism eats from the pit"
    assert w.store_food[y, x] < 2.0
    e1 = s.total_energy()
    assert abs(e1 - (e0 + (w.ledger.solar_in - in0) - (w.ledger.heat_out - out0))) < 1e-4
    # spoilage inside a pit is slow
    before = float(w.store_food[y, x])
    s.world.update(s.clock, s.rng)
    assert float(w.store_food[y, x]) > before * 0.99


def test_old_saves_are_upgraded(tmp_path):
    s = Simulation(small_cfg(seed=5))
    s.run(30)
    d = s.state_dict()
    blob = pickle.loads(zlib.decompress(s.snapshot_bytes()))
    # simulate a save from before rivers, storage and the 13th skill existed
    for k in ("river", "lake", "flow", "fresh", "store_food"):
        blob["world"].pop(k, None)
    blob["pop"]["skills"] = blob["pop"]["skills"][:, :12]
    old = zlib.compress(pickle.dumps(blob))
    t = Simulation(small_cfg(seed=1))
    t.load_bytes(old)
    assert t.pop.skills.shape[1] == L.N_SKILLS
    assert hasattr(t.world, "river") and hasattr(t.world, "store_food")
    t.run(20)
    assert t.clock.tick == 50
