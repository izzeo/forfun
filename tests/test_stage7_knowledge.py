"""Stage 7: limited perception, hidden layers that still act, and their discovery."""
import numpy as np

from genesis import Simulation
from genesis.biology import genome as GN
from genesis.intelligence import skills
from genesis.intelligence.perception import perceive
from genesis.physics import laws as L
from genesis.physics.micro import SCALE
from tests.conftest import small_cfg


def test_perception_ignores_hidden_reality():
    s = Simulation(small_cfg(seed=8))
    s.run(5)
    a = perceive(s)
    # perturb everything an organism cannot sense: microbes, strains, ore, nutrients
    s.world.pathogens[:] = np.random.default_rng(0).random(s.world.pathogens.shape)
    s.world.virulence[:] = 0.99
    s.world.ore[:] = 9
    s.world.nutrients[:] = 1.7
    b = perceive(s)
    assert np.array_equal(a.inputs, b.inputs)
    assert np.array_equal(a.materials, b.materials)


def test_hidden_microbes_still_cause_disease():
    s = Simulation(small_cfg(seed=8, initial_population=60))
    s.world.pathogens[:] = 0.9
    s.world.virulence[:] = 1.0
    s.run(80)
    assert (s.pop.infection > 0).any() or s.history.death_causes().get("disease", 0) > 0
    assert "microscopic" not in s.civ.layers, "the cause is real but undiscovered"


def test_discovering_the_microscopic_layer_changes_behaviour():
    s = Simulation(small_cfg(seed=8, scenario="advanced", initial_population=10))
    s.step()
    h0 = s.civ.hygiene(s).copy()
    skills.learn_skill(s, 0, L.SK["microscope"], 0.5, source="experiment")
    assert "microscopic" in s.civ.layers
    assert s.civ.hygiene(s)[0] > h0[0], "knowing germs exist reduces infection risk"
    assert s.civ.knows_layer(s, 0, "microscopic")
    assert not s.civ.knows_layer(s, 5, "microscopic") or s.pop.group[5] == s.pop.group[0] >= 0


def test_micro_detail_is_consistent_with_aggregate_and_observation_is_passive():
    a = Simulation(small_cfg(seed=12))
    b = Simulation(small_cfg(seed=12))
    y, x = np.unravel_index(np.argmax(a.world.pathogens), a.world.pathogens.shape)
    v = a.micro_view(int(x), int(y))
    assert abs(v["count"] - a.world.pathogens[y, x] * SCALE) <= 1
    for _ in range(60):
        a.step(); b.step()
        a.micro_view(int(x), int(y))  # someone keeps looking through the microscope
    assert a.state_hash() == b.state_hash(), "watching must not change what happens"
    v = a.micro_view(int(x), int(y))
    assert abs(v["count"] - a.world.pathogens[y, x] * SCALE) <= 1


def test_celestial_layer_exists_before_it_is_known():
    s = Simulation(small_cfg(seed=8))
    sky = s.world.celestial.to_dict()
    assert sky["planets"] and sky["moons"]
    assert "celestial" not in s.civ.layers
