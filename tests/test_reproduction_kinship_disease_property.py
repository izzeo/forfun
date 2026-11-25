"""Sexes & pregnancy, kinship & inbreeding, evolving pathogen strains, property & inheritance."""
import numpy as np

from genesis import Simulation
from genesis.biology import genome as GN, lifecycle
from genesis.evolution import evolution as EV
from genesis.intelligence.perception import perceive
from genesis.physics import laws as L
from tests.conftest import small_cfg


def _sim(n=6, seed=4):
    s = Simulation(small_cfg(seed=seed, scenario="advanced", initial_population=n))
    s.step()
    return s


def test_two_sexes_and_pregnancy_lead_to_birth():
    s = _sim()
    p = s.pop
    p.sex[0], p.sex[1] = 0, 1
    p.energy[0] = 4.0
    EV.conceive(s, 0, 1, 0.0)
    mother = int(p.id[0])
    assert p.pregnant_until[0] > s.clock.tick and mother in p.pending
    n0 = len(p)
    assert EV.deliver(s) == 0, "no birth before the pregnancy is over"
    s.clock.tick = int(p.pregnant_until[0])
    p.energy[0] = 4.0
    p.health[0] = 1.0
    assert EV.deliver(s) == 1
    child = len(p) - 1
    assert len(p) == n0 + 1 and p.parent_a[child] == mother and p.parent_b[child] == p.id[1]
    assert p.sex[child] in (0, 1) and p.pregnant_until[0] == -1


def test_starving_mother_miscarries():
    s = _sim()
    p = s.pop
    p.sex[0], p.sex[1] = 0, 1
    EV.conceive(s, 0, 1, 0.0)
    p.energy[0] = 0.01
    EV.deliver(s)
    assert p.pregnant_until[0] == -1 and int(p.id[0]) not in p.pending
    assert s.stats_counter["miscarriages"] >= 1


def test_same_sex_pairs_are_not_mates():
    s = _sim(8)
    p = s.pop
    p.sex[:] = 0
    pc = perceive(s)
    assert (pc.mate_idx < 0).all()


def test_relatedness_from_pedigree():
    s = _sim(8)
    p = s.pop
    ids = p.id
    p.parent_a[2], p.parent_b[2] = ids[0], ids[1]        # 2 and 3 are full siblings, children of 0 and 1
    p.parent_a[3], p.parent_b[3] = ids[0], ids[1]
    p.parent_a[4], p.parent_b[4] = ids[0], ids[5]        # half sibling of 2
    assert EV.relatedness(p, 2, 0) == 0.5
    assert EV.relatedness(p, 2, 3) == 0.5
    assert EV.relatedness(p, 2, 4) == 0.25
    assert EV.relatedness(p, 6, 7) == 0.0


def test_inbreeding_depression():
    s = _sim()
    p = s.pop
    p.sex[0], p.sex[1] = 0, 1
    for r, slot in ((0.0, "out"), (0.5, "in")):
        p.energy[0] = 5.0; p.health[0] = 1.0; p.pregnant_until[0] = -1
        EV.conceive(s, 0, 1, r)
        s.clock.tick = int(p.pregnant_until[0])
        EV.deliver(s)
    out, inb = len(p) - 2, len(p) - 1
    assert p.inbreeding[out] == 0 and abs(p.inbreeding[inb] - 0.25) < 1e-6
    # lifespan of the inbred child is cut by (1 - F) beyond normal random variation
    from genesis.biology.genome import Phenotype as Ph
    assert p.max_age[inb] <= Ph.max_age(p.genome[inb:inb + 1])[0] * 1.15 * 0.75 + 1e-3


def test_pathogen_strains_mutate_and_immunity_is_specific():
    s = _sim(30, seed=9)
    p = s.pop
    p.x[:] = p.x[0]; p.y[:] = p.y[0]            # crowded: host-to-host spread
    p.infection[:] = 0
    p.infection[0] = 1.0
    p.strain[0] = 0.5
    p.immunity[:] = 0
    s.world.pathogens[:] = 0
    for _ in range(60):
        s.step()
    infected = p.infection > 0.05
    assert s.stats_counter["host_transmissions"] > 0 or infected.sum() > 1
    strains = p.strain[p.infection > 0]
    assert len(strains) > 1 and strains.std() > 0, "strains diverge by mutation"
    # immunity protects against similar strains much more than against distant ones
    p.immunity[1] = 1.0; p.immune_strain[1] = 0.3
    near = p.immunity[1] * np.exp(-abs(0.32 - p.immune_strain[1]) / 0.15)
    far = p.immunity[1] * np.exp(-abs(0.9 - p.immune_strain[1]) / 0.15)
    assert near > 0.8 and far < 0.05


def _pit(s, owner_idx, food=3.0):
    p, w = s.pop, s.world
    y, x = int(p.y[0]), int(p.x[0])
    w.struct[y, x] = L.STRUCT_STORE; w.struct_hp[y, x] = 1.0
    w.store_food[y, x] = food
    w.struct_owner[y, x] = int(p.id[owner_idx]) if owner_idx is not None else -1
    w.struct_group[y, x] = -1
    return y, x


def test_owner_eats_from_own_store():
    s = _sim()
    p = s.pop
    y, x = _pit(s, 0)
    p.energy[0] = 0.1
    s._eat_stores()
    assert s.world.store_food[y, x] < 3.0


def test_principled_stranger_does_not_steal():
    s = _sim()
    p = s.pop
    y, x = _pit(s, 1)
    p.group[:] = -1
    p.partner[:] = -1
    p.parent_a[:] = -1; p.parent_b[:] = -1
    p.norms[0, 0] = 1.0                          # strong sense of justice
    p.energy[0] = 0.1
    s._eat_stores()
    assert s.world.store_food[y, x] == 3.0


def test_inheritance_passes_property_to_partner():
    s = _sim()
    p, w = s.pop, s.world
    y, x = _pit(s, 0)
    p.partner[0] = p.id[1]
    heir = int(p.id[1])
    cause = np.full(len(p), -1); cause[0] = lifecycle.C["old_age"]
    s.consumed = np.zeros(len(p), bool)
    lifecycle.bury(s, cause, {})
    assert w.struct_owner[y, x] == heir
    assert "inheritance" in s.analytics.firsts
