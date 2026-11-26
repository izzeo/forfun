"""Observer API and live stream."""
from fastapi.testclient import TestClient

from genesis import Simulation
from genesis.api.server import create_app
from tests.conftest import small_cfg


def _client():
    sim = Simulation(small_cfg(seed=40, initial_population=60))
    return TestClient(create_app(sim, autostart=False)), sim


def test_endpoints():
    c, sim = _client()
    assert c.post("/api/control", json={"action": "step", "value": 30}).json()["tick"] == 30
    st = c.get("/api/state").json()
    assert st["clock"]["tick"] == 30 and st["population"] == len(sim.pop)
    w = c.get("/api/world").json()
    assert len(w["elevation"]) == sim.world.W * sim.world.H
    f = c.get("/api/field/pathogens").json()
    assert f["hidden_layer"] == "microscopic" and f["known_to_inhabitants"] is False
    rows = c.get("/api/agents").json()["rows"]
    aid = int(rows[0][0])
    d = c.get(f"/api/agent/{aid}").json()
    for k in ("long_term_memory", "short_term_memory", "personality", "emotions", "norms", "relations", "biography"):
        assert k in d
    assert c.get(f"/api/agent/{aid}/lineage").json()["ancestors"]["id"] == aid
    for path in ("/api/stats", "/api/species", "/api/groups", "/api/settlements", "/api/events", "/api/firsts",
                 "/api/deaths", "/api/knowledge", "/api/morality", "/api/micro?x=5&y=5"):
        assert c.get(path).status_code == 200, path
    assert c.get("/").status_code == 200


def test_controls_and_stream(tmp_path):
    c, sim = _client()
    assert c.post("/api/control", json={"action": "scale", "value": "year"}).json()["scale"] == "year"
    p = str(tmp_path / "s.pkl")
    c.post("/api/control", json={"action": "step", "value": 10})
    assert c.post("/api/control", json={"action": "save", "value": p}).json()["ok"]
    c.post("/api/control", json={"action": "step", "value": 5})
    c.post("/api/control", json={"action": "load", "value": p})
    assert c.get("/api/state").json()["clock"]["tick"] == 10
    with c.websocket_connect("/ws") as ws:
        f = ws.receive_json()
        assert f["type"] == "frame" and "fields" in f and len(f["agents"]) == f["population"]


def test_creator_mode_is_recorded_and_booked():
    c, sim = _client()
    c.post("/api/control", json={"action": "step", "value": 5})
    x, y = float(sim.pop.x[0]), float(sim.pop.y[0])
    r = c.post("/api/creator", json={"action": "event", "kind": "abundance", "x": x, "y": y, "radius": 4}).json()
    assert r["energy_added"] >= 0
    n0 = len(sim.pop)
    c.post("/api/creator", json={"action": "spawn", "x": x, "y": y, "n": 3, "kind": "advanced"})
    assert len(sim.pop) == n0 + 3
    aid = int(sim.pop.id[0])
    assert c.post("/api/creator", json={"action": "act", "agent": aid, "kind": "reveal", "skill": "fire"}).status_code == 200
    assert c.post("/api/creator", json={"action": "law", "name": "plant_growth", "value": 0.2}).json()["value"] == 0.2
    assert c.post("/api/creator", json={"action": "event", "kind": "nonsense", "x": 1, "y": 1}).status_code == 400
    info = c.get("/api/creator").json()
    assert info["count"] == 4 and info["creator_energy"] > 0
    assert any(e["kind"] == "intervention" for e in sim.history.events(["intervention"]))
    # energy balance still closes when the creator's contributions are counted
    e = sim.total_energy()
    L = sim.world.ledger
    assert abs(e - (sim.initial_energy + L.solar_in - L.heat_out + L.creator_in)) / sim.initial_energy < 1e-3
