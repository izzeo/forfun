"""FastAPI server: observer API + WebSocket stream for the visualisation."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..civilization.morality import FRAMEWORKS, VIRTUES
from ..config import WorldConfig
from ..engine.runner import Runner
from ..engine.simulation import Simulation
from ..intelligence import language
from ..intelligence.brain import life_stage
from ..memory import memory as MEM
from ..physics import laws as L
from ..timekeeping import time_scales

WEB = Path(__file__).resolve().parent.parent / "web"
STAGES = ["infant", "child", "adolescent", "adult", "elder"]
HIDDEN_FIELDS = {"pathogens": "microscopic", "ore": "subterranean", "nutrients": "microscopic"}


def _r(a, k=3):
    return np.round(np.asarray(a, dtype=np.float64), k).tolist()


class CreatorAct(BaseModel):
    action: str                       # law | event | spawn | act | place
    name: str | None = None
    value: float | None = None
    x: float | None = None
    y: float | None = None
    radius: float | None = 6
    n: int | None = 10
    kind: str | None = None
    agent: int | None = None
    skill: str | None = None
    material: str | None = None


class Control(BaseModel):
    action: str
    value: str | float | int | None = None
    seed: int | None = None
    scenario: str | None = None
    framework: str | None = None
    population: int | None = None


def create_app(sim: Simulation | None = None, cfg: WorldConfig | None = None, autostart: bool = True) -> FastAPI:
    state = dict(runner=Runner(sim or Simulation(cfg or WorldConfig())))

    @asynccontextmanager
    async def lifespan(app):
        if autostart:
            state["runner"].set_scale(getattr(app.state, "initial_scale", "day"))
            state["runner"].start()
        yield
        state["runner"].stop()

    app = FastAPI(title="GENESIS observer", version="0.1.0", lifespan=lifespan)

    def R() -> Runner:
        return state["runner"]

    def S() -> Simulation:
        return state["runner"].sim

    # ------------------------------------------------------------------ static
    @app.middleware("http")
    async def no_cache_ui(request, call_next):
        # the UI changes with the code; never let the browser serve a stale copy
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(WEB / "index.html"))

    # ------------------------------------------------------------------ world
    @app.get("/api/world")
    def world():
        s = S()
        w = s.world
        return dict(width=w.W, height=w.H, sea_level=s.cfg.sea_level, elevation=_r(w.elev.ravel(), 3),
                    water=w.water.ravel().astype(int).tolist(), ptype=w.ptype.ravel().tolist(),
                    river=(w.river | w.lake).ravel().astype(int).tolist(),
                    sand=w.sand.ravel().astype(int).tolist() if hasattr(w, "sand") else [],
                    rock=w.rock.ravel().tolist() if hasattr(w, "rock") else [],
                    stony=(w.stone >= 1).ravel().astype(int).tolist(),
                    plants=[dict(name=p["name"], color=p["color"]) for p in L.PLANTS],
                    celestial=w.celestial.to_dict(), reality_layers=L.REALITY_LAYERS,
                    known_layers=s.civ.layers, config=s.cfg.to_dict(), skills=L.SKILLS, actions=L.ACTIONS,
                    causes=L.DEATH_CAUSES, structures=L.STRUCTURES, virtues=VIRTUES,
                    scales=time_scales(s.cfg))

    @app.get("/api/field/{name}")
    def field(name: str):
        s = S()
        try:
            f = s.world.field(name)
        except KeyError:
            raise HTTPException(404, "unknown field")
        layer = HIDDEN_FIELDS.get(name)
        return dict(name=name, values=_r(f.ravel(), 3), min=float(f.min()), max=float(f.max()),
                    hidden_layer=layer, known_to_inhabitants=(layer is None or layer in s.civ.layers))

    @app.get("/api/state")
    def get_state():
        s, r = S(), R()
        last = s.analytics.series[-1] if s.analytics.series else {}
        return dict(clock=s.clock.info(), population=len(s.pop), paused=r.paused, scale=r.scale,
                    requested_rate=r.rate, effective_rate=round(r.effective_rate, 2), ms_per_tick=round(s.last_step_ms, 2),
                    civ=s.civ.summary(), latest=last, snapshots=[t for t, _ in s.snapshots], storms=s.world.storms,
                    moral_framework=s.moral.framework, scenario=s.cfg.scenario, seed=s.cfg.seed,
                    lod=s.world.lod_stats, energy=dict(total=round(s.total_energy(), 2), **s.world.ledger.to_dict()),
                    matter=dict(minerals=round(s.total_minerals(), 2)),
                    climate=dict(co2=round(float(getattr(s.world, "co2", 1.0)), 4),
                                 greenhouse_c=round(float(getattr(s.world, "greenhouse", 0.0)), 3)))

    def compact_agents(s):
        p = s.pop
        if len(p) == 0:
            return []
        g = p.genome
        return np.column_stack([p.id, np.round(p.x, 2), np.round(p.y, 2), p.species, p.group, p.last_action,
                                np.round(g[:, GN.T["size"]], 2), np.round(g[:, GN.T["hue"]], 2),
                                np.round(g[:, GN.T["brain"]], 2), life_stage(p),
                                np.round(g[:, GN.T["aquatic"]], 2), np.round(g[:, GN.T["diet"]], 2),
                                np.round(p.health, 2)]).tolist()

    @app.get("/api/agents")
    def agents():
        return dict(columns=["id", "x", "y", "species", "group", "action", "size", "hue", "brain", "stage",
                             "aquatic", "diet", "health"],
                    rows=compact_agents(S()))

    def agent_detail(s, aid: int):
        p = s.pop
        i = p.index_of(aid)
        rec = s.history.organism(aid)
        if i is None:
            if rec is None:
                raise HTTPException(404, "no such organism")
            return dict(id=aid, alive=False, record=rec, biography=s.history.biography(aid))
        g = p.genome[i]
        cap = float(P.capacity(g[None])[0])
        rel = p.relations.get(aid, {})
        top_rel = sorted(rel.items(), key=lambda kv: -abs(kv[1]))[:12]
        lex = p.lexicon.get(aid, {})
        sp = s.phylo.species.get(int(p.species[i]), {})
        grp = s.civ.groups.get(int(p.group[i]))
        return dict(
            id=aid, alive=True, name=p.names.get(aid, "?"), x=float(p.x[i]), y=float(p.y[i]),
            age_hours=int(p.age[i]), age_days=round(int(p.age[i]) / s.cfg.day_ticks, 1),
            max_age_days=round(float(p.max_age[i]) / s.cfg.day_ticks, 1), stage=STAGES[int(life_stage(p)[i])],
            generation=int(p.generation[i]), species=dict(id=int(p.species[i]), name=sp.get("name")),
            group=(dict(id=grp["id"], name=grp["name"], polity=grp["polity"]) if grp else None),
            parents=[int(p.parent_a[i]), int(p.parent_b[i])], partner=int(p.partner[i]),
            partner_name=p.names.get(int(p.partner[i])), children=int(p.children[i]), kills=int(p.kills[i]),
            energy=round(float(p.energy[i]), 3), capacity=round(cap, 3), health=round(float(p.health[i]), 3),
            hydration=round(float(p.hydration[i]), 3), infection=round(float(p.infection[i]), 3),
            immunity=round(float(p.immunity[i]), 3), growth=round(float(p.growth[i]), 3),
            action=L.ACTIONS[int(p.last_action[i])], prestige=round(float(p.prestige[i]), 3),
            traits={t: round(float(g[k]), 3) for k, t in enumerate(GN.TRAITS)},
            personality=dict(zip(["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"], _r(p.personality[i]))),
            emotions=dict(zip(["fear", "joy", "anger", "sadness", "affection"], _r(p.emotion[i]))),
            norms=dict(zip(VIRTUES, _r(p.norms[i]))),
            beliefs=dict(zip(["spirits", "naturalism", "ritual"], _r(p.beliefs[i]))),
            body=dict(sex=["female", "male"][int(p.sex[i])] if p.sex[i] >= 0 else "?",
                      pregnant=bool(p.pregnant_until[i] >= 0),
                      due_in_hours=int(p.pregnant_until[i] - s.clock.tick) if p.pregnant_until[i] >= 0 else None,
                      inbreeding=round(float(p.inbreeding[i]), 3), kin_aversion=round(float(p.kin_aversion[i]), 3),
                      strain_virulence=round(float(p.strain[i]), 3) if p.infection[i] > 0.05 else None),
            self_model=dict(agency=round(float(p.agency[i]), 3), helpless=bool(p.agency[i] < 0.25),
                            efficacy=dict(zip(["fight", "forage", "explore", "craft", "social"], _r(p.self_eff[i])))),
            skills={L.SKILLS[k]: round(float(v), 3) for k, v in enumerate(p.skills[i]) if v > 0.01},
            inventory={L.RESOURCES[k]: round(float(v), 2) for k, v in enumerate(p.inventory[i]) if v > 0.01},
            materials=dict(wood_moisture=round(float(p.wood_moist[i]), 3), stone_tool=round(float(p.tool_q[i]), 2),
                           stone_tool_durability=round(float(p.tool_dur[i]), 3), metal_tool=round(float(p.metal_q[i]), 2),
                           glass_clarity=round(float(p.glass_q[i]), 2)),
            food_values={L.PLANTS[k]["name"]: round(float(v), 3) for k, v in enumerate(p.food_value[i])},
            innate_food={L.PLANTS[k]["name"]: round(float(g[GN.FOOD_PREF0 + k]), 3) for k in range(L.N_PLANTS)},
            habits={L.ACTIONS[k]: round(float(v), 3) for k, v in enumerate(p.habit[i])},
            fear_of_appearance=_r(p.hue_fear[i]),
            places=dict(food=_r(p.food_mem[i], 2), danger=_r(p.danger_mem[i], 2), water=_r(p.water_mem[i], 2)),
            short_term_memory=MEM.short_term(p, i),
            long_term_memory=MEM.long_term(p, aid, s.clock.tick)[-60:],
            relations=[dict(id=o, name=p.names.get(o, "?"), affinity=round(v, 3)) for o, v in top_rel],
            lexicon={language.CONCEPTS[c]: language.top_word(lex, c) for c in lex},
            biography=s.history.biography(aid),
        )

    @app.get("/api/agent/{aid}")
    def agent(aid: int):
        s = S()
        with R().lock:
            return agent_detail(s, aid)

    @app.get("/api/agent/{aid}/lineage")
    def lineage(aid: int, up: int = 4, down: int = 2):
        return S().history.lineage(aid, up=min(up, 8), down=min(down, 4))

    @app.get("/api/stats")
    def stats(since: int = 0, limit: int = 2000):
        rows = [r for r in S().analytics.series if r["tick"] >= since]
        step = max(1, len(rows) // limit)
        return dict(rows=rows[::step])

    @app.get("/api/species")
    def species():
        return dict(species=S().phylo.tree())

    @app.get("/api/groups")
    def groups():
        out = []
        for gr in S().civ.groups.values():
            d = {k: v for k, v in gr.items() if not k.startswith("_")}
            out.append(d)
        return dict(groups=out, feuds=[dict(groups=list(k), score=round(v, 2)) for k, v in S().civ.feuds.items()],
                    wars=S().civ.wars, trade=[dict(groups=list(k), count=v) for k, v in S().civ.trade.items()])

    @app.get("/api/settlements")
    def settlements():
        return dict(settlements=[{k: v for k, v in st.items() if not k.startswith("_")} for st in S().civ.settlements.values()])

    @app.get("/api/events")
    def events(kinds: str | None = None, since: int = 0, limit: int = 200):
        return dict(events=S().history.events(kinds.split(",") if kinds else None, since, limit))

    @app.get("/api/firsts")
    def firsts():
        return dict(firsts=sorted(S().analytics.firsts.values(), key=lambda f: f["tick"]))

    @app.get("/api/deaths")
    def deaths():
        s = S()
        return dict(causes=s.history.death_causes(), counts=s.history.counts(),
                    recent=s.history.events(["death"], 0, 100))

    @app.get("/api/knowledge")
    def knowledge():
        s = S()
        return dict(layers=s.civ.layers, all_layers=L.REALITY_LAYERS, skills_known=s.civ.skills_known)

    @app.get("/api/micro")
    def micro(x: int, y: int):
        s = S()
        if not (0 <= x < s.world.W and 0 <= y < s.world.H):
            raise HTTPException(400, "outside world")
        with R().lock:
            return s.micro_view(x, y)

    @app.get("/api/morality")
    def morality():
        return dict(frameworks={k: dict(description=v["description"], ideal=v["ideal"]) for k, v in FRAMEWORKS.items()},
                    virtues=VIRTUES, current=S().moral.framework, strength=S().moral.strength)

    @app.get("/api/creator")
    def creator_info():
        from ..engine import creator as CR
        s = S()
        laws = {k: getattr(s.cfg, attr) for k, (attr, lo, hi) in CR.LAWS.items()}
        laws["temperature"] = float(getattr(s.world, "creator_temp", 0.0))
        return dict(laws=laws, ranges={k: [lo, hi] for k, (attr, lo, hi) in CR.LAWS.items()} | {"temperature": [-20, 20]},
                    events=CR.EVENTS, skills=L.SKILLS, materials=["stone", "clay", "sand", "ore", "copper", "salt"],
                    interventions=s.interventions[-100:], count=len(s.interventions),
                    creator_energy=round(float(getattr(s.world.ledger, "creator_in", 0.0)), 2),
                    creator_minerals=round(float(getattr(s.world, "creator_minerals", 0.0)), 2))

    @app.post("/api/creator")
    def creator_act(c: CreatorAct):
        from ..engine import creator as CR
        s, r = S(), R()
        try:
            with r.lock:
                if c.action == "law":
                    return CR.set_law(s, c.name, float(c.value))
                if c.action == "event":
                    return CR.event(s, c.kind, c.x, c.y, c.radius or 6)
                if c.action == "spawn":
                    return CR.spawn(s, c.x, c.y, c.n or 10, c.kind or "advanced", c.agent)
                if c.action == "act":
                    return CR.act_on(s, int(c.agent), c.kind, c.skill)
                if c.action == "place":
                    return CR.place(s, c.x, c.y, c.material, c.value or 3)
        except (CR.CreatorError, TypeError) as e:
            raise HTTPException(400, str(e))
        raise HTTPException(400, "unknown creator action")

    @app.post("/api/control")
    def control(c: Control):
        r = R()
        a = c.action
        if a == "pause":
            r.paused = True
        elif a == "resume":
            r.paused = False
        elif a == "step":
            r.step(int(c.value or 1))
        elif a == "scale":
            try:
                r.set_scale(str(c.value))
            except ValueError:
                raise HTTPException(400, "unknown scale")
        elif a == "save":
            path = str(c.value or "genesis_save.pkl")
            with r.lock:
                S().save(path)
            return dict(ok=True, path=os.path.abspath(path))
        elif a == "load":
            with r.lock:
                S().load(str(c.value or "genesis_save.pkl"))
                r.version += 1
        elif a == "rewind":
            with r.lock:
                try:
                    t = S().rewind(int(c.value))
                except ValueError as e:
                    raise HTTPException(400, str(e))
                r.version += 1
            return dict(ok=True, tick=t)
        elif a == "snapshot":
            with r.lock:
                S().take_snapshot()
        elif a == "new":
            cfg = WorldConfig(seed=int(c.seed if c.seed is not None else np.random.randint(1 << 30)),
                              scenario=c.scenario or "primordial", moral_framework=c.framework or "none",
                              initial_population=int(c.population or 300))
            with r.lock:
                r.sim = Simulation(cfg)
                r.version += 1
        elif a == "framework":
            from ..civilization.morality import MoralSystem
            with r.lock:
                S().cfg.moral_framework = str(c.value)
                S().moral = MoralSystem(str(c.value), S().cfg.moral_education)
        else:
            raise HTTPException(400, "unknown action")
        return dict(ok=True, paused=r.paused, scale=r.scale, tick=S().clock.tick)

    # ------------------------------------------------------------------ live stream
    def build_frame(k: int, last_version: int) -> dict:
        s, r = S(), R()
        with r.lock:
            frame = dict(type="frame", clock=s.clock.info(), minute=int(r.fractional_hour() * 60),
                         population=len(s.pop), agents=compact_agents(s), paused=r.paused,
                         effective_rate=round(r.effective_rate, 2), storms=s.world.storms,
                         version=r.version)
            if k % 8 == 0 or r.version != last_version:
                w = s.world
                frame["fields"] = dict(
                    plants=np.clip(w.biomass * 200, 0, 255).astype(np.uint8).ravel().tolist(),
                    clouds=np.clip(w.clouds * 255, 0, 255).astype(np.uint8).ravel().tolist(),
                    struct=w.struct.ravel().tolist(),
                    fire=np.clip(w.fire * 255, 0, 255).astype(np.uint8).ravel().tolist(),
                    ptype=w.ptype.ravel().tolist(),
                    temp=np.clip((w.temperature + 30) * 4, 0, 255).astype(np.uint8).ravel().tolist())
                frame["sun"] = s.clock.sun_elevation()
        return frame

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        k = 0
        last_version = -1
        try:
            while True:
                # built in a worker thread: waiting for the simulation lock must never block the server
                frame = await asyncio.to_thread(build_frame, k, last_version)
                last_version = frame["version"]
                await sock.send_json(frame)
                k += 1
                await asyncio.sleep(0.25)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app
