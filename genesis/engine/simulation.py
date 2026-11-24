"""Simulation Engine: owns the universe and advances it one tick (one virtual
hour) at a time, in a fixed causal order. Supports snapshots, save/load and
deterministic rewind (replay from the nearest snapshot)."""
from __future__ import annotations

import pickle
import time
import zlib
from collections import Counter

import numpy as np

from ..analytics.analytics import Analytics
from ..biology import actions, lifecycle
from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..biology.population import Population, make_name
from ..civilization.civ import Civilization
from ..civilization.culture import Culture
from ..civilization.morality import MoralSystem
from ..civilization.social import Social
from ..config import WorldConfig
from ..evolution.evolution import Phylogeny, reproduce
from ..intelligence import brain, skills
from ..intelligence.perception import perceive
from ..intelligence.psyche import Psyche
from ..memory import memory as MEM
from ..physics import laws as L
from ..physics.micro import MicroRegion
from ..physics.world import World
from ..storage.history import History
from ..timekeeping import Clock
from .scheduler import Scheduler

SNAPSHOT_VERSION = 1


class Simulation:
    def __init__(self, cfg: WorldConfig | None = None):
        self.cfg = cfg or WorldConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.clock = Clock(self.cfg)
        self.world = World(self.cfg, self.rng)
        self.pop = Population()
        self.history = History(self.cfg.db_path)
        self.history.enabled = self.cfg.record_events
        self.analytics = Analytics()
        self.civ = Civilization()
        self.phylo = Phylogeny()
        self.social = Social(self)
        self.culture = Culture()
        self.psyche = Psyche()
        self.moral = MoralSystem(self.cfg.moral_framework, self.cfg.moral_education)
        self.scheduler = Scheduler()
        self.stats_counter: Counter = Counter()
        self.social_stats: Counter = Counter()
        self.snapshots: list[tuple[int, bytes]] = []
        self.micro: dict[tuple[int, int], MicroRegion] = {}
        self.consumed = np.zeros(0, bool)
        self.last_step_ms = 0.0
        self._lex_cache = (-1, None)
        self._genesis()

    # ------------------------------------------------------------------ setup
    def _genesis(self) -> None:
        cfg, rng, w = self.cfg, self.rng, self.world
        n = cfg.initial_population
        cand = np.argwhere(w.land & (w.biomass > 0.2) & (w.base_temp > 8) & (w.dist_water < 6))
        if len(cand) == 0:
            cand = np.argwhere(w.land)
        pick = cand[rng.integers(0, len(cand), n)]
        if cfg.scenario == "advanced":
            g = GN.advanced_genomes(rng, n)
        elif cfg.scenario == "primordial":
            g = GN.random_genomes(rng, n)
        else:
            raise ValueError(f"unknown scenario {cfg.scenario!r}")
        max_age = P.max_age(g) * rng.uniform(0.85, 1.15, n)
        pers = np.clip(np.stack([g[:, GN.T["curiosity"]], g[:, GN.T["patience"]], g[:, GN.T["sociality"]],
                                 1 - g[:, GN.T["aggression"]], g[:, GN.T["fear_sens"]]], axis=1)
                       + rng.normal(0, 0.12, (n, 5)), 0, 1)
        ids = self.pop.add(
            n, x=pick[:, 1] + rng.random(n), y=pick[:, 0] + rng.random(n), heading=rng.uniform(0, 2 * np.pi, n),
            energy=0.6 * P.capacity(g), max_age=max_age, age=(rng.uniform(0.15 if cfg.scenario == "advanced" else 0.0, 0.45, n) * max_age).astype(np.int32),
            genome=g, weights=brain.instinct_weights(rng, n), food_value=g[:, GN.FOOD_PREF0:GN.FOOD_PREF0 + L.N_PLANTS],
            personality=pers, birth_tick=0)
        for a in ids:
            self.pop.names[int(a)] = make_name(int(a))
        self.phylo.found_initial(self)
        self.history.births(self, np.arange(n))
        self.scheduler.after_poisson(0, cfg.storm_rate, rng, "storm")
        self.scheduler.after_poisson(0, cfg.eclipse_rate, rng, "eclipse")
        self.scheduler.after_poisson(0, 1 / 3000.0, rng, "cold_snap")
        self.history.event(self, "genesis", -1, -1, 0, 0, dict(seed=cfg.seed, population=n))
        self.history.flush()
        self.initial_energy = self.total_energy()

    # ------------------------------------------------------------------ helpers
    def lexicon_sizes(self) -> np.ndarray:
        if self._lex_cache[0] == self.clock.tick and len(self._lex_cache[1]) == len(self.pop):
            return self._lex_cache[1]
        lex = self.pop.lexicon
        arr = np.array([len(lex.get(int(a), ())) for a in self.pop.id], dtype=np.int32)
        self._lex_cache = (self.clock.tick, arr)
        return arr

    def total_energy(self) -> float:
        pop = self.pop
        agents = float(pop.energy.sum(dtype=np.float64) + pop.body_mass().sum(dtype=np.float64)
                       + pop.inventory[:, L.R["food"]].sum(dtype=np.float64)) if len(pop) else 0.0
        return self.world.total_energy() + agents

    # ------------------------------------------------------------------ events
    def _handle_events(self) -> None:
        tick, rng, w = self.clock.tick, self.rng, self.world
        for kind, data in self.scheduler.due(tick):
            if kind == "storm":
                land = np.argwhere(w.land)
                cy, cx = land[rng.integers(len(land))]
                s = dict(id=self.scheduler.seq, cy=int(cy), cx=int(cx), r=int(rng.integers(5, 14)),
                         intensity=float(rng.uniform(0.5, 1.5)))
                w.storms.append(s)
                self.scheduler.at(tick + int(rng.integers(6, 36)), "storm_end", id=s["id"])
                self.scheduler.after_poisson(tick, self.cfg.storm_rate, rng, "storm")
                self.history.event(self, "storm", -1, -1, cx, cy, dict(radius=s["r"], intensity=round(s["intensity"], 2)))
            elif kind == "storm_end":
                w.storms = [s for s in w.storms if s["id"] != data["id"]]
            elif kind == "lightning":
                pass
            elif kind == "eclipse":
                self._eclipse()
                self.scheduler.after_poisson(tick, self.cfg.eclipse_rate, rng, "eclipse")
            elif kind == "cold_snap":
                w.temp_offset = -float(rng.uniform(5, 12))
                self.scheduler.at(tick + int(rng.integers(48, 240)), "cold_snap_end")
                self.history.event(self, "cold_snap", -1, -1, 0, 0, dict(offset=round(w.temp_offset, 1)))
            elif kind == "cold_snap_end":
                w.temp_offset = 0.0
                self.scheduler.after_poisson(tick, 1 / 3000.0, rng, "cold_snap")
        # lightning inside storms can start wildfires
        for s in w.storms:
            if rng.random() < 0.05 * s["intensity"]:
                cy = int(np.clip(s["cy"] + rng.integers(-s["r"], s["r"] + 1), 0, w.H - 1))
                cx = int(np.clip(s["cx"] + rng.integers(-s["r"], s["r"] + 1), 0, w.W - 1))
                if w.ignite(cy, cx):
                    self.history.event(self, "wildfire", -1, -1, cx, cy, {})

    def _eclipse(self) -> None:
        pop = self.pop
        self.history.event(self, "eclipse", -1, -1, 0, 0, {})
        if len(pop) == 0:
            return
        idx = np.nonzero(pop.genome[:, GN.T["brain"]] > 0.25)[0]
        pop.emotion[idx, 0] = np.minimum(1, pop.emotion[idx, 0] + 0.5)
        MEM.consolidate(pop, idx, MEM.M["eclipse"], 0.8, -1, self.clock.tick, detail="saw the sun go dark")
        self.culture.unexplained_harm(self, idx, 0.4, "eclipse")

    # ------------------------------------------------------------------ main loop
    def step(self) -> None:
        t0 = time.perf_counter()
        cfg, pop, w = self.cfg, self.pop, self.world
        self.clock.advance()
        tick = self.clock.tick
        self._handle_events()
        w.set_active_from_positions(pop.x, pop.y)
        w.update(self.clock, self.rng)

        n = len(pop)
        if n:
            self.consumed = np.zeros(n, bool)
            pc = perceive(self)
            if tick % cfg.deliberation_interval == 0:
                brain.deliberate(self, pc)
            act = brain.choose(self, pc)
            pop.last_action = act.astype(np.int8)
            self.analytics.observe_actions(act)
            e0 = pop.energy.copy()
            h0 = pop.health.copy()
            reward = np.zeros(n, np.float32)
            gain = np.zeros(n, np.float32)
            dmg = np.zeros((n, len(L.DEATH_CAUSES)), np.float32)

            move_cost = actions.move(self, pc, act)
            toxin = actions.eat(self, pc, act, gain)
            dmg[:, lifecycle.C["poisoning"]] += toxin
            actions.drink(self, pc, act)
            attacks = actions.attack(self, pc, act, gain, dmg)
            skills.experiment(self, pc, act, reward)
            skills.work(self, pc, act, reward)
            self._eat_stores()
            st = self.social.interact(pc, act, attacks)
            self.social_stats.update(st)
            cause = lifecycle.metabolize(self, pc, act, move_cost, dmg)
            # predators feed on the prey they killed
            killers = {}
            kills = []
            for a, t, hit, c in attacks:
                if cause[t] in (lifecycle.C["predation"], lifecycle.C["violence"]) and t not in killers and cause[a] < 0:
                    killers[t] = int(pop.id[a])
                    if pop.genome[a, GN.T["diet"]] > 0.3:
                        kills.append((a, t))
                        self.analytics.first(self, "predation", "First organism killed and ate another")
                    else:
                        self.analytics.first(self, "violence", "First lethal violence between organisms")
            actions.feed_on_kills(self, kills, gain)
            # learning from the consequences of this hour's actions
            de = pop.energy - e0
            dh = pop.health - h0
            reward += (de / P.capacity(pop.genome)) * 4 + dh * 3
            brain.learn_habits(self, act, reward)
            self.psyche.update(self, pc, reward)
            self._decay_memories()
            births = reproduce(self, pc, act)
            if births:
                cause = np.concatenate([cause, np.full(len(pop) - len(cause), -1, np.int64)])
                self.consumed = np.concatenate([self.consumed, np.zeros(len(pop) - len(self.consumed), bool)])
            lifecycle.bury(self, cause, killers)

        if tick % cfg.group_interval == 0:
            self.civ.update(self)
        if tick % cfg.species_interval == 0:
            self.phylo.update(self)
        if tick % cfg.stats_interval == 0:
            self.analytics.record(self)
            self.history.flush()
        if tick % cfg.snapshot_interval == 0:
            self.take_snapshot()
        self.last_step_ms = (time.perf_counter() - t0) * 1000

    def _eat_stores(self) -> None:
        pop = self.pop
        cap = P.capacity(pop.genome)
        hungry = (pop.energy < 0.3 * cap) & (pop.inventory[:, L.R["food"]] > 0)
        if hungry.any():
            amt = np.minimum(pop.inventory[hungry, L.R["food"]], 0.2)
            pop.inventory[hungry, L.R["food"]] -= amt
            pop.energy[hungry] += amt * 0.9
            self.world.ledger.heat_out += float(amt.sum() * 0.1)
        # food spoils without pottery
        spoil = (pop.inventory[:, L.R["food"]] > 0) & (pop.inventory[:, L.R["pottery"]] < 1)
        if spoil.any():
            amt = pop.inventory[spoil, L.R["food"]] * 0.01
            pop.inventory[spoil, L.R["food"]] -= amt
            cy, cx = self.world.cell_of(pop.x[spoil], pop.y[spoil])
            np.add.at(self.world.detritus, (cy, cx), amt)

    def _decay_memories(self) -> None:
        pop = self.pop
        pop.food_mem[:, 2] *= 0.997
        pop.danger_mem[:, 2] *= 0.995
        pop.water_mem[:, 2] *= 0.999
        pop.hue_fear *= 0.9998

    def run(self, ticks: int, callback=None) -> None:
        for _ in range(ticks):
            self.step()
            if callback:
                callback(self)
        self.history.flush()

    # ------------------------------------------------------------------ micro LOD
    def micro_view(self, cx: int, cy: int) -> dict:
        key = (int(cy), int(cx))
        reg = self.micro.get(key)
        if reg is None:
            reg = MicroRegion(self.world, cy, cx, self.cfg.seed, self.clock.tick)
            self.micro[key] = reg
            if len(self.micro) > 16:
                self.micro.pop(next(iter(self.micro)))
        else:
            reg.sync_from_aggregate(self.world, self.clock.tick)
        d = reg.to_dict()
        d["aggregate_density"] = float(self.world.pathogens[cy, cx])
        d["known_to_inhabitants"] = "microscopic" in self.civ.layers
        return d

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        self.history.flush()
        return dict(version=SNAPSHOT_VERSION, cfg=self.cfg.to_dict(), tick=self.clock.tick,
                    rng=self.rng.bit_generator.state, world=self.world.__dict__, pop=self.pop.state_dict(),
                    civ=self.civ.state_dict(), phylo=self.phylo.__dict__, analytics=self.analytics.__dict__,
                    scheduler=self.scheduler.__dict__, stats_counter=self.stats_counter,
                    social_stats=self.social_stats, initial_energy=self.initial_energy)

    def snapshot_bytes(self) -> bytes:
        return zlib.compress(pickle.dumps(self.state_dict(), protocol=pickle.HIGHEST_PROTOCOL), 3)

    def load_bytes(self, blob: bytes) -> None:
        d = pickle.loads(zlib.decompress(blob))
        self.cfg = WorldConfig.from_dict(d["cfg"])
        self.clock = Clock(self.cfg, d["tick"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = d["rng"]
        self.world = World.__new__(World)
        self.world.__dict__.update(d["world"])
        self.world.cfg = self.cfg
        self.pop = Population()
        self.pop.load_state(d["pop"])
        self.civ = Civilization(); self.civ.load_state(d["civ"])
        self.phylo = Phylogeny(); self.phylo.__dict__.update(d["phylo"])
        self.analytics = Analytics(); self.analytics.__dict__.update(d["analytics"])
        self.scheduler = Scheduler(); self.scheduler.__dict__.update(d["scheduler"])
        self.stats_counter = d["stats_counter"]
        self.social_stats = d["social_stats"]
        self.initial_energy = d["initial_energy"]
        self.moral = MoralSystem(self.cfg.moral_framework, self.cfg.moral_education)
        self.social = Social(self)
        self.micro = {}
        self._lex_cache = (-1, None)

    def take_snapshot(self) -> None:
        self.snapshots.append((self.clock.tick, self.snapshot_bytes()))
        if len(self.snapshots) > self.cfg.max_snapshots:
            # keep the first snapshot (genesis era) and thin out the oldest of the rest
            self.snapshots.pop(1)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(current=self.snapshot_bytes(), snapshots=self.snapshots), f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.load_bytes(d["current"])
        self.snapshots = d["snapshots"]
        self.history.truncate_after(self.clock.tick)

    def rewind(self, target_tick: int) -> int:
        """Return to an earlier moment. Replays deterministically from the nearest snapshot."""
        target_tick = max(0, int(target_tick))
        cands = [(t, b) for t, b in self.snapshots if t <= target_tick]
        if not cands:
            raise ValueError("no snapshot at or before that tick")
        t, blob = max(cands, key=lambda tb: tb[0])
        self.history.truncate_after(t)
        self.load_bytes(blob)
        self.snapshots = [(st, b) for st, b in self.snapshots if st <= t]
        while self.clock.tick < target_tick:
            self.step()
        self.history.flush()
        return self.clock.tick

    def state_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for arr in (self.pop.id, self.pop.x, self.pop.y, self.pop.energy, self.pop.genome, self.world.biomass,
                    self.world.pathogens, self.pop.skills, self.pop.habit):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(self.clock.tick).encode())
        return h.hexdigest()
