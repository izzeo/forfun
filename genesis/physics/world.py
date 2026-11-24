"""Physics Engine: the planet.

Terrain, climate, day/night, seasons, weather, flora, decay, a hidden
microbial layer, wildfire, mineral resources and a sky. All fields are
NumPy arrays on a (height, width) grid; one cell is roughly one hectare.

Level-of-detail: the grid is split into chunks. Chunks that contain or border
organisms (or that the observer is looking at) are integrated every tick.
Other chunks are advanced lazily in closed form every ``lod_interval`` ticks
(analytic logistic solution), which conserves the energy ledger while saving
most of the cost of remote, empty land.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import WorldConfig
from . import laws as L


def _value_noise(rng: np.random.Generator, h: int, w: int, octaves: int = 5, base: int = 4) -> np.ndarray:
    out = np.zeros((h, w), dtype=np.float64)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = base * (2 ** o)
        g = rng.random((n + 1, n + 1))
        ys = np.linspace(0, n, h, endpoint=False)
        xs = np.linspace(0, n, w, endpoint=False)
        y0 = ys.astype(int); x0 = xs.astype(int)
        fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
        fy = fy * fy * (3 - 2 * fy); fx = fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0]; b = g[y0][:, x0 + 1]; c = g[y0 + 1][:, x0]; d = g[y0 + 1][:, x0 + 1]
        out += amp * (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy)
        total += amp
        amp *= 0.5
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def _blur(f: np.ndarray, k: float = 0.2) -> np.ndarray:
    """Cheap 4-neighbour diffusion with reflecting borders."""
    p = np.pad(f, 1, mode="edge")
    return f * (1 - 4 * k) + k * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])


class EnergyLedger:
    """Tracks energy entering (sunlight) and leaving (heat) the biosphere."""

    def __init__(self):
        self.solar_in = 0.0
        self.heat_out = 0.0

    def to_dict(self):
        return dict(solar_in=self.solar_in, heat_out=self.heat_out)


class Celestial:
    """The sky. Fully simulated, but unknown to inhabitants until observed."""

    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.axial_tilt_deg = float(18 + rng.random() * 12)
        self.moons = [dict(name="moon", period_days=cfg.moon_days, radius_rel=0.27)]
        n_planets = int(rng.integers(2, 5))
        self.planets = [dict(name=f"wanderer-{i + 1}", period_days=float(cfg.year_days * (0.4 + 3.5 * rng.random())),
                             brightness=float(0.2 + rng.random())) for i in range(n_planets)]
        self.star_count = int(4000 + rng.integers(0, 3000))

    def to_dict(self):
        return dict(axial_tilt_deg=self.axial_tilt_deg, moons=self.moons, planets=self.planets,
                    star_count=self.star_count)


class World:
    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.cfg = cfg
        H, W = cfg.height, cfg.width
        self.H, self.W = H, W
        lat = np.linspace(-1, 1, H)[:, None] * np.ones((1, W))
        self.lat = lat.astype(np.float32)

        # --- terrain ---------------------------------------------------------
        noise = _value_noise(rng, H, W, octaves=6, base=3)
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        elev = 0.65 * noise + 0.35 * (1 - np.clip(r, 0, 1.2) / 1.2)
        elev = (elev - elev.min()) / (elev.max() - elev.min())
        self.elev = elev.astype(np.float32)
        self.water = self.elev < cfg.sea_level
        self.deep = self.elev < cfg.sea_level - 0.12
        land = ~self.water

        # distance to water (rough): repeated dilation
        dist = np.where(self.water, 0.0, 30.0)
        for _ in range(12):
            p = np.pad(dist, 1, mode="edge")
            dist = np.minimum(dist, np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]) + 1)
        self.dist_water = dist.astype(np.float32)
        # "coast": land cell next to water (drinkable)
        self.coast = land & (self.dist_water <= 1.0)

        height_above = np.clip((self.elev - cfg.sea_level) / (1 - cfg.sea_level), 0, 1)
        self.height_above = height_above.astype(np.float32)
        self.base_temp = (27.0 - 26.0 * np.abs(self.lat) - 22.0 * height_above).astype(np.float32)
        self.base_moist = np.clip(1.0 - self.dist_water / 14.0 + 0.3 * (_value_noise(rng, H, W, 4, 3) - 0.5), 0, 1).astype(np.float32)

        # --- flora: dominant plant type per cell -----------------------------
        veg = _value_noise(rng, H, W, octaves=4, base=6)
        ptype = np.full((H, W), L.GRASS, dtype=np.int8)
        warm = self.base_temp > 8
        moist = self.base_moist > 0.45
        ptype[land & warm & moist & (veg > 0.55)] = L.NUT_TREE
        ptype[land & warm & moist & (veg > 0.35) & (veg <= 0.55)] = L.BERRY
        ptype[land & warm & moist & (veg <= 0.22)] = L.NIGHTSHADE
        ptype[land & (self.base_moist < 0.3) & (veg > 0.5)] = L.TUBER
        ptype[self.water] = L.ALGAE
        self.ptype = ptype
        kmask = np.ones((H, W), dtype=np.float32)
        kmask[self.deep] = 0.0
        kmask[land & (height_above > 0.75)] = 0.25
        kmask[land & (self.base_temp < 0)] *= 0.4
        self.kmask = kmask
        self.biomass = (L.PLANT_K[ptype] * kmask * (0.3 + 0.5 * rng.random((H, W)))).astype(np.float32)

        # --- matter & hidden layers ------------------------------------------
        self.detritus = np.zeros((H, W), dtype=np.float32)
        self.nutrients = np.full((H, W), 0.5, dtype=np.float32)
        warmwet = np.clip(self.base_temp / 25.0, 0, 1) * self.base_moist
        self.pathogens = (0.15 * warmwet * rng.random((H, W))).astype(np.float32)  # hidden (microscopic)
        self.virulence = (0.5 + 0.5 * _value_noise(rng, H, W, 3, 2)).astype(np.float32)  # hidden strain property
        self.moisture = self.base_moist.copy()
        self.clouds = (0.3 * _value_noise(rng, H, W, 3, 2)).astype(np.float32)
        self.fire = np.zeros((H, W), dtype=np.float32)
        self.temperature = self.base_temp.copy()
        self.light = np.zeros((H, W), dtype=np.float32)
        self.wind = np.array([1.0, 0.3], dtype=np.float64)
        self._wind_acc = np.zeros(2)
        self.temp_offset = 0.0  # cold snaps / droughts set this

        rock = _value_noise(rng, H, W, 4, 5)
        self.stone = np.where(land & ((height_above > 0.35) | (rock > 0.72)), 1 + 4 * rock, 0).astype(np.float32)
        self.clay = land & (self.dist_water <= 2) & (self.elev < cfg.sea_level + 0.08) & (rock < 0.5)
        self.sand = land & (self.dist_water <= 1) & ~self.clay
        ore_n = _value_noise(rng, H, W, 4, 7)
        self.ore = np.where(land & (height_above > 0.3) & (ore_n > 0.7), 3 * ore_n, 0).astype(np.float32)  # hidden underground
        self.dug = np.zeros((H, W), dtype=bool)

        self.struct = np.zeros((H, W), dtype=np.int8)
        self.struct_hp = np.zeros((H, W), dtype=np.float32)
        self.struct_group = np.full((H, W), -1, dtype=np.int32)

        self.celestial = Celestial(cfg, rng)
        self.ledger = EnergyLedger()

        # --- LOD --------------------------------------------------------------
        cs = cfg.chunk_size
        self.ch, self.cw = math.ceil(H / cs), math.ceil(W / cs)
        self.active_chunks = np.ones((self.ch, self.cw), dtype=bool)
        self.observed_chunks = np.zeros((self.ch, self.cw), dtype=bool)
        self.chunk_last = np.zeros((self.ch, self.cw), dtype=np.int64)
        self._light_acc = 0.0
        self._light_n = 0
        self.lod_stats = dict(active_fraction=1.0)
        self.storms: list[dict] = []

    # ------------------------------------------------------------------
    @property
    def land(self) -> np.ndarray:
        return ~self.water

    def cell_of(self, x: np.ndarray, y: np.ndarray):
        cx = np.clip(x.astype(np.int64), 0, self.W - 1)
        cy = np.clip(y.astype(np.int64), 0, self.H - 1)
        return cy, cx

    def set_active_from_positions(self, x: np.ndarray, y: np.ndarray) -> None:
        cs = self.cfg.chunk_size
        act = np.zeros((self.ch, self.cw), dtype=bool)
        if len(x):
            act[np.clip((y // cs).astype(int), 0, self.ch - 1), np.clip((x // cs).astype(int), 0, self.cw - 1)] = True
            p = np.pad(act, 1)
            act = act | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] | p[2:, 2:] | p[:-2, 2:] | p[2:, :-2]
        # NOTE: observation never changes the level of detail of causal processes;
        # only the presence of organisms (who interact with plants) does.
        if not self.cfg.lod_enabled:
            act[:] = True
        self.active_chunks = act
        self.lod_stats["active_fraction"] = float(act.mean())

    def _chunk_mask(self, chunks: np.ndarray) -> np.ndarray:
        cs = self.cfg.chunk_size
        return np.kron(chunks, np.ones((cs, cs), dtype=bool))[: self.H, : self.W]

    # ------------------------------------------------------------------
    def update(self, clock, rng: np.random.Generator) -> None:
        cfg = self.cfg
        sun = max(0.0, clock.sun_elevation())
        yf = clock.year_fraction
        season = math.sin(2 * math.pi * yf)
        tilt = self.celestial.axial_tilt_deg / 23.0

        # ---- weather: wind, clouds, rain --------------------------------
        self.wind += rng.normal(0, 0.03, 2)
        self.wind = np.clip(self.wind, -1.5, 1.5)
        self._wind_acc += self.wind * 0.25
        sh = np.trunc(self._wind_acc).astype(int)
        if sh[0] or sh[1]:
            self.clouds = np.roll(self.clouds, (int(sh[1]), int(sh[0])), axis=(0, 1))
            self._wind_acc -= sh
        evap = np.where(self.water, 0.012, 0.003 * self.moisture) * np.clip(self.temperature / 25.0, 0, 1.5) * (0.3 + sun)
        self.clouds += evap.astype(np.float32)
        for s in self.storms:
            yy, xx = s["cy"], s["cx"]
            y0, y1 = max(0, yy - s["r"]), min(self.H, yy + s["r"])
            x0, x1 = max(0, xx - s["r"]), min(self.W, xx + s["r"])
            self.clouds[y0:y1, x0:x1] += 0.05 * s["intensity"]
        thresh = 0.55 - 0.2 * self.height_above
        rain = np.clip(self.clouds - thresh, 0, None) * 0.25
        self.clouds = np.clip(_blur(self.clouds - rain, 0.05) * 0.996, 0, 1.2).astype(np.float32)
        self.rain = rain.astype(np.float32)
        self.moisture = np.clip(self.moisture + rain * 2.0 - 0.004 * (self.moisture - self.base_moist * 0.8) - 0.002 * sun, 0, 1).astype(np.float32)
        self.moisture[self.water] = 1.0

        # ---- light & temperature -----------------------------------------
        insol = sun * (1 - 0.35 * np.abs(self.lat)) * (1 + 0.35 * tilt * season * self.lat)
        self.light = np.clip(insol * (1 - 0.65 * np.clip(self.clouds, 0, 1)), 0, 1.5).astype(np.float32)
        seasonal = 11.0 * tilt * season * self.lat
        seasonal = np.where(self.water, seasonal * 0.5, seasonal)
        diurnal = 5.0 * clock.sun_elevation() * (1 - 0.5 * np.clip(self.clouds, 0, 1))
        self.temperature = (self.base_temp + seasonal + diurnal + self.temp_offset).astype(np.float32)
        self.temperature[self.fire > 0.1] += 15.0 * self.fire[self.fire > 0.1]

        # ---- flora (LOD) ----------------------------------------------------
        self._light_acc += sun * (1 - 0.35 * 0.4)
        self._light_n += 1
        act = self._chunk_mask(self.active_chunks)
        self._grow(act, dt=1.0, light=self.light)
        if clock.tick % cfg.lod_interval == 0:
            inact = ~act
            if inact.any():
                mean_light = self._light_acc / max(1, self._light_n)
                self._grow_closed_form(inact, dt=float(cfg.lod_interval), mean_light=mean_light)
            self._light_acc = 0.0
            self._light_n = 0

        # ---- decay -> nutrients ---------------------------------------------
        dec = self.detritus * L.DETRITUS_DECAY * np.clip(0.3 + self.temperature / 30.0, 0.1, 1.5) * (0.5 + self.moisture)
        dec = np.minimum(dec, self.detritus)
        self.detritus -= dec
        self.ledger.heat_out += float(dec.sum())
        self.nutrients = np.clip(self.nutrients + dec * 0.8 - 0.0004 * (self.nutrients - 0.5), 0, 2.0).astype(np.float32)

        # ---- microscopic layer (aggregate model) ---------------------------
        warmwet = np.clip(self.temperature / 28.0, 0, 1.2) * self.moisture
        P = self.pathogens
        dP = (0.02 * cfg.pathogen_virulence * warmwet * P * (1 - P)
              + 0.01 * np.clip(self.detritus, 0, 1) - 0.006 * P - 0.004 * self.light * P)
        self.pathogens = np.clip(_blur(P + dP, 0.03), 0, 1).astype(np.float32)

        # ---- fire --------------------------------------------------------------
        if self.fire.any():
            self._update_fire(rng)

        # ---- structures decay -------------------------------------------------
        built = self.struct > 0
        if built.any():
            self.struct_hp[built] -= 0.0008 * (1 + 3 * self.rain[built])
            hearth = self.struct == L.STRUCT_HEARTH
            self.struct_hp[hearth] -= 0.03 * (1 + 5 * self.rain[hearth])  # fuel burns out
            gone = built & (self.struct_hp <= 0)
            self.struct[gone] = 0
            self.struct_group[gone] = -1

    # ------------------------------------------------------------------
    def _growth_rate(self, light):
        t = self.temperature
        ftemp = np.clip(1 - ((t - 22.0) / 22.0) ** 2, 0, 1)
        fmoist = np.where(self.water, 1.0, np.clip(self.moisture * 1.6, 0, 1))
        return self.cfg.plant_growth * light * ftemp * fmoist

    def carrying(self):
        fert = np.clip(0.6 + 0.8 * self.nutrients, 0.5, 1.8)
        return (L.PLANT_K[self.ptype] * self.kmask * fert).astype(np.float32)

    def _grow(self, mask, dt, light):
        K = self.carrying()
        B = self.biomass
        g = self._growth_rate(light) * dt
        grow = np.where(B < K, g * (B + 0.02) * (1 - B / np.maximum(K, 1e-6)), 0.0)
        die = np.where(B > K, 0.02 * dt * (B - K), 0.0)
        grow = np.where(mask & (K > 0), grow, 0.0)
        die = np.where(mask, die, 0.0)
        self.biomass = (B + grow - die).astype(np.float32)
        self.detritus += die.astype(np.float32)
        self.nutrients = np.clip(self.nutrients - 0.05 * grow, 0, 2).astype(np.float32)
        self.ledger.solar_in += float(grow.sum())

    def _grow_closed_form(self, mask, dt, mean_light):
        """Analytic logistic integration B(t)=K/(1+((K-B0)/B0)e^{-rt}) for lazily updated chunks."""
        K = self.carrying()
        B0 = np.maximum(self.biomass, 1e-4)
        ml = np.full_like(self.light, mean_light) * (1 - 0.65 * np.clip(self.clouds, 0, 1))
        r = self._growth_rate(ml)
        Kc = np.maximum(K, 1e-6)
        Bt = Kc / (1 + ((Kc - B0) / B0) * np.exp(-r * dt))
        Bt = np.where(B0 > K, K + (B0 - K) * np.exp(-0.02 * dt), Bt)
        Bt = np.where(K > 0, Bt, B0 * np.exp(-0.02 * dt))
        new = np.where(mask, Bt, self.biomass).astype(np.float32)
        delta = new - self.biomass
        self.ledger.solar_in += float(np.clip(delta, 0, None).sum())
        self.detritus += np.clip(-delta, 0, None).astype(np.float32)
        self.nutrients = np.clip(self.nutrients - 0.05 * np.clip(delta, 0, None), 0, 2).astype(np.float32)
        self.biomass = new

    def _update_fire(self, rng):
        F = self.fire
        burn = np.minimum(self.biomass, 0.15 * F)
        self.biomass -= burn
        self.ledger.heat_out += float(burn.sum() * 0.8)
        self.detritus += burn * 0.2
        p = np.pad(F, 1)
        neigh = np.maximum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        fuel = self.biomass * (1 - self.moisture) * self.land
        ignite = (neigh > 0.3) & (fuel > 0.25) & (rng.random(F.shape) < 0.08 * neigh)
        F = np.where(ignite, np.maximum(F, 0.6), F)
        F = F * 0.93 - self.rain * 3.0
        F[(self.biomass < 0.05)] *= 0.5
        F = np.clip(F, 0, 1).astype(np.float32)
        F[F < 0.05] = 0
        # burnt shelters
        hit = (F > 0.5) & (self.struct > 0)
        self.struct_hp[hit] -= 0.1
        self.fire = F

    # ------------------------------------------------------------------
    def ignite(self, cy: int, cx: int, strength: float = 0.9) -> bool:
        if self.water[cy, cx] or self.biomass[cy, cx] < 0.2 or self.moisture[cy, cx] > 0.7:
            return False
        self.fire[cy, cx] = max(self.fire[cy, cx], strength)
        return True

    def total_energy(self) -> float:
        return float(self.biomass.sum(dtype=np.float64) + self.detritus.sum(dtype=np.float64))

    # observer-facing field accessor ------------------------------------
    def field(self, name: str) -> np.ndarray:
        if name == "elevation":
            return self.elev
        if name == "plants":
            return self.biomass
        if name == "temperature":
            return self.temperature
        if name == "clouds":
            return self.clouds
        if name == "moisture":
            return self.moisture
        if name == "pathogens":
            return self.pathogens
        if name == "ore":
            return self.ore
        if name == "nutrients":
            return self.nutrients
        if name == "fire":
            return self.fire
        if name == "light":
            return self.light
        if name == "structures":
            return self.struct.astype(np.float32)
        raise KeyError(name)
