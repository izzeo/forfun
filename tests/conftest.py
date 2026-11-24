import numpy as np
import pytest

from genesis import Simulation, WorldConfig


def small_cfg(**kw):
    base = dict(seed=7, width=48, height=48, initial_population=120, snapshot_interval=100, stats_interval=24)
    base.update(kw)
    return WorldConfig(**base)


@pytest.fixture
def sim():
    return Simulation(small_cfg())


@pytest.fixture(scope="module")
def run_primordial():
    s = Simulation(small_cfg(seed=3))
    s.run(600)
    return s


@pytest.fixture(scope="module")
def run_advanced():
    s = Simulation(small_cfg(seed=2, scenario="advanced", initial_population=150))
    s.run(600)
    return s


def place(sim, i, x, y):
    sim.pop.x[i] = x
    sim.pop.y[i] = y


def find_cell(sim, cond):
    ys, xs = np.nonzero(cond)
    assert len(ys), "no cell matching condition"
    return int(xs[0]), int(ys[0])
