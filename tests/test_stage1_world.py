"""Stage 1: a working planet and simple organisms."""
import numpy as np

from genesis import Simulation
from genesis.physics import laws as L
from genesis.biology.population import FIELDS
from tests.conftest import small_cfg


def test_planet_has_land_water_and_climate(sim):
    w = sim.world
    assert 0.1 < w.water.mean() < 0.9, "planet should have both land and sea"
    # colder towards the poles
    eq = w.base_temp[w.H // 2].mean()
    pole = w.base_temp[0].mean()
    assert eq > pole + 10
    assert w.biomass.min() >= 0
    assert (w.stone > 0).any() and w.clay.any() and (w.ore > 0).any()


def test_day_night_cycle_and_weather():
    s = Simulation(small_cfg(initial_population=0))
    lights, temps = [], []
    for _ in range(48):
        s.step()
        lights.append(float(s.world.light.mean()))
        temps.append(float(s.world.temperature.mean()))
    lights = np.array(lights)
    assert lights.max() > 0.1 and lights.min() == 0.0, "light must follow day and night"
    assert max(temps) - min(temps) > 2, "temperature must vary over the day"
    assert s.world.clouds.max() > 0


def test_world_keeps_evolving_without_observers_or_life():
    s = Simulation(small_cfg(initial_population=0))
    b0 = s.world.biomass.copy()
    s.run(200)
    assert not np.allclose(b0, s.world.biomass), "causal history continues with nobody present"
    assert (s.world.biomass <= s.world.carrying() * 1.05 + 1e-3).all(), "plants stay within carrying capacity"


def test_organisms_have_required_attributes(sim):
    p = sim.pop
    for f in ("id", "energy", "max_age", "health", "food_mem", "stm", "habit", "food_value"):
        assert f in FIELDS
    assert len(np.unique(p.id)) == len(p), "unique identifiers"
    assert (p.energy > 0).all() and (p.health == 1).all()
    assert all(int(a) in p.names for a in p.id)


def test_organisms_move_within_the_world(sim):
    x0 = sim.pop.x.copy()
    sim.run(30)
    p = sim.pop
    assert (p.x >= 0).all() and (p.x < sim.world.W).all() and (p.y >= 0).all() and (p.y < sim.world.H).all()
    assert len(p) > 0


def test_lod_closed_form_matches_full_detail():
    """Lazy closed-form integration of empty regions approximates full per-tick detail."""
    a = Simulation(small_cfg(initial_population=0, lod_enabled=True))
    b = Simulation(small_cfg(initial_population=0, lod_enabled=False))
    a.run(240); b.run(240)
    ta, tb = a.world.biomass.sum(), b.world.biomass.sum()
    assert abs(ta - tb) / tb < 0.06
    assert a.world.lod_stats["active_fraction"] < 0.01  # nothing needed detail
