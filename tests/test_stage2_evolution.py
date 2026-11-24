"""Stage 2: energy, reproduction, death and evolution."""
import numpy as np

from genesis import Simulation
from genesis.biology import genome as GN
from tests.conftest import small_cfg


def test_energy_is_conserved():
    s = Simulation(small_cfg(seed=11))
    e0 = s.total_energy()
    in0, out0 = s.world.ledger.solar_in, s.world.ledger.heat_out
    s.run(300)
    e1 = s.total_energy()
    expected = e0 + (s.world.ledger.solar_in - in0) - (s.world.ledger.heat_out - out0)
    assert abs(e1 - expected) / e0 < 1e-3, (e1, expected)


def test_nobody_disappears_without_cause(run_primordial):
    s = run_primordial
    c = s.history.counts()
    assert c["dead_without_cause"] == 0
    assert c["ever_lived"] - c["dead"] == len(s.pop), "alive = ever born - dead"
    assert len(np.unique(s.pop.id)) == len(s.pop)
    assert sum(s.history.death_causes().values()) == c["dead"]


def test_generations_and_inheritance(run_primordial):
    s = run_primordial
    p = s.pop
    assert p.generation.max() >= 2, "several generations should have been born"
    kids = np.nonzero(p.parent_a > 0)[0]
    pid = {int(a): k for k, a in enumerate(p.id)}
    pairs = [(k, pid[int(p.parent_a[k])]) for k in kids if int(p.parent_a[k]) in pid]
    assert pairs, "some parents should still be alive"
    d_parent = np.mean([GN.genetic_distance(p.genome[k], p.genome[j]) for k, j in pairs])
    rnd = np.random.default_rng(0).permutation(len(p))
    d_random = np.mean(GN.genetic_distance(p.genome, p.genome[rnd]))
    assert d_parent < d_random, "offspring resemble their parents (heredity)"
    assert d_parent > 0, "mutation introduces variation"


def test_natural_selection_adapts_to_a_cold_world():
    """In a world held 12 degC colder, cold-tolerant genotypes should take over."""
    s = Simulation(small_cfg(seed=5, initial_population=160))
    n = len(s.pop)
    half = np.arange(n) < n // 2
    s.pop.genome[:, GN.T["temp_opt"]] = np.where(half, 0.25, 0.75)

    def chill(sim):
        sim.world.temp_offset = -12.0

    start = float(s.pop.genome[:, GN.T["temp_opt"]].mean())
    s.run(1500, callback=chill)
    assert len(s.pop) > 0
    assert s.pop.genome[:, GN.T["temp_opt"]].mean() < start - 0.1


def test_life_history_tradeoff_lifespan_vs_maturity():
    """Longer lifespan delays maturity: a real trade-off, not a free benefit."""
    from genesis.biology.genome import Phenotype as P
    g = np.full((2, GN.G), 0.5, np.float32)
    g[0, GN.T["lifespan"]], g[1, GN.T["lifespan"]] = 0.1, 0.9
    mat_hours = P.maturity(g) * P.max_age(g)
    assert mat_hours[1] > mat_hours[0]


def test_speciation_is_tracked(run_primordial):
    sp = run_primordial.phylo.species
    assert len(sp) >= 1
    assert all("origin" in v for v in sp.values())
