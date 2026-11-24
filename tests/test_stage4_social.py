"""Stage 4: social behaviour, communication and groups."""
import numpy as np

from genesis import Simulation
from genesis.biology import genome as GN
from genesis.intelligence import language
from genesis.intelligence.perception import perceive
from tests.conftest import small_cfg


def _talkers(n=12, seed=4):
    s = Simulation(small_cfg(seed=seed, scenario="advanced", initial_population=n))
    s.step()
    p = s.pop
    p.genome[:, GN.T["communication"]] = 0.9
    p.genome[:, GN.T["brain"]] = 0.8
    # everyone stands together, so they share attention
    p.x[:] = p.x[0]; p.y[:] = p.y[0]
    return s


def test_naming_game_converges_to_shared_words():
    s = _talkers()
    pc = perceive(s)
    rng = np.random.default_rng(0)
    ids = [int(a) for a in s.pop.id]
    coh0, _ = language.coherence(s.pop, ids)
    for _ in range(600):
        i, j = rng.choice(len(s.pop), 2, replace=False)
        language.converse(s, pc, int(i), int(j))
    coh, vocab = language.coherence(s.pop, ids)
    assert coh > 0.8, coh
    assert coh > coh0
    assert vocab, "a shared vocabulary exists"
    # words are invented, not predefined
    assert all(v["word"].isalpha() for v in vocab.values())


def test_alarm_call_transfers_danger_knowledge():
    s = _talkers(4)
    pc = perceive(s)
    p = s.pop
    pc.threat[0] = 0.9
    pc.threat_idx[0] = 3
    p.danger_mem[1] = 0
    for _ in range(5):
        language.converse(s, pc, 0, 1)
    assert p.danger_mem[1, 2] > 0, "the hearer learned about a danger it did not see itself"


def test_relationships_groups_and_sharing_emerge(run_advanced):
    s = run_advanced
    assert any(len(r) for r in s.pop.relations.values()), "relationships form from interactions"
    assert len(s.civ.groups) > 0, "stable groups are detected"
    shared = sum(r["social"].get("shared", 0) for r in s.analytics.series)
    care = sum(r["social"].get("care", 0) for r in s.analytics.series)
    assert shared + care > 0, "food is shared / offspring cared for"
    assert "communication" in s.analytics.firsts


def test_affinity_is_individual_and_asymmetric(sim):
    a, b = int(sim.pop.id[0]), int(sim.pop.id[1])
    sim.social.adjust(a, b, 0.5)
    assert sim.social.affinity(a, b) == 0.5
    assert sim.social.affinity(b, a) == 0.0
