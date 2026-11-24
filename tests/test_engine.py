"""Simulation engine: determinism, save/load, rewind, time scales, performance, history."""
import time

import numpy as np

from genesis import Simulation
from genesis.engine.runner import Runner
from genesis.timekeeping import time_scales
from tests.conftest import small_cfg


def test_deterministic_given_seed():
    a = Simulation(small_cfg(seed=31)); b = Simulation(small_cfg(seed=31))
    a.run(150); b.run(150)
    assert a.state_hash() == b.state_hash()
    c = Simulation(small_cfg(seed=32)); c.run(150)
    assert c.state_hash() != a.state_hash()


def test_save_and_load_roundtrip(tmp_path):
    a = Simulation(small_cfg(seed=33))
    a.run(120)
    path = tmp_path / "w.pkl"
    a.save(str(path))
    a.run(80)
    h = a.state_hash()
    b = Simulation(small_cfg(seed=1))
    b.load(str(path))
    assert b.clock.tick == 120
    b.run(80)
    assert b.state_hash() == h, "a loaded universe continues exactly as the original"


def test_rewind_replays_identical_history():
    s = Simulation(small_cfg(seed=34, snapshot_interval=100))
    hashes = {}
    for _ in range(320):
        s.step()
        if s.clock.tick in (150, 250):
            hashes[s.clock.tick] = s.state_hash()
    events_before = s.history.events(since=0, limit=100000)
    s.rewind(150)
    assert s.clock.tick == 150 and s.state_hash() == hashes[150]
    assert all(e["tick"] <= 150 for e in s.history.events(since=0, limit=100000)), "abandoned future erased"
    s.run(100)
    assert s.state_hash() == hashes[250]


def test_time_scales_are_consistent():
    cfg = small_cfg()
    sc = time_scales(cfg)
    assert sc["minute"] * 60 == sc["hour"] == 1.0
    assert sc["day"] == cfg.day_ticks and sc["year"] == cfg.year_ticks
    r = Runner(Simulation(small_cfg(initial_population=10)))
    for name in ("minute", "hour", "day", "year"):
        r.set_scale(name)
        assert r.rate == sc[name]


def test_runner_advances_in_real_time():
    s = Simulation(small_cfg(initial_population=20))
    r = Runner(s, scale="day")
    r.paused = False
    r.start()
    time.sleep(1.2)
    r.stop()
    assert 5 <= s.clock.tick <= 60


def test_history_is_persisted(run_primordial):
    h = run_primordial.history
    c = h.counts()
    assert c["ever_lived"] > 120
    dead = h.db.execute("SELECT id FROM organisms WHERE death_tick IS NOT NULL LIMIT 1").fetchone()[0]
    rec = h.organism(dead)
    assert rec["cause"] and rec["final"] is not None
    assert any(e["kind"] == "death" for e in h.biography(dead))
    lin = h.lineage(dead)
    assert lin["ancestors"]["id"] == dead


def test_performance_scales_with_population():
    s = Simulation(small_cfg(seed=35, width=96, height=96, initial_population=1500))
    s.run(3)
    t = time.perf_counter()
    s.run(10)
    ms = (time.perf_counter() - t) / 10 * 1000
    assert ms < 250, f"{ms:.1f} ms per tick with ~{len(s.pop)} organisms"
