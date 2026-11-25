"""Perception: what each organism can know about the world *right now*.

Only visible, local information is read here: plant colour categories and the
visible part of their biomass, open water, fire, structures, and the visible
appearance (size, hue) of nearby organisms. Hidden physics (toxicity, locked
nutrients, pathogens, ore, other creatures' diet) is deliberately NOT read.
Tests verify that perturbing hidden fields leaves percepts unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..physics import laws as L
from ..physics.spatial import neighbor_pairs, nearest_by

N_DIR = 8
_ANG = np.arange(N_DIR) * (2 * np.pi / N_DIR)
DIRS = np.stack([np.cos(_ANG), np.sin(_ANG)], axis=1).astype(np.float32)

OBJECTS = ["stone", "stone2", "nut", "wood", "wood2", "tool", "ground", "seed", "dug", "clay", "fire",
           "ore", "pottery", "sand", "glass", "metal", "small_thing", "sky", "words", "charcoal", "kiln", "salt"]
OBJ = {n: i for i, n in enumerate(OBJECTS)}
HUE_BINS = 8


@dataclass
class Percept:
    cy: np.ndarray
    cx: np.ndarray
    inputs: np.ndarray
    food_here: np.ndarray
    best_food: np.ndarray
    food_dir: np.ndarray
    water_near: np.ndarray
    water_dir: np.ndarray
    threat: np.ndarray
    threat_idx: np.ndarray
    threat_dir: np.ndarray
    prey_idx: np.ndarray
    prey_dist: np.ndarray
    mate_idx: np.ndarray
    mate_dist: np.ndarray
    friend_idx: np.ndarray
    friend_dist: np.ndarray
    parent_idx: np.ndarray
    pairs: tuple
    materials: np.ndarray
    dry: np.ndarray
    sense: np.ndarray
    warmth: np.ndarray


def hue_bin(genome: np.ndarray) -> np.ndarray:
    return np.minimum((genome[:, GN.T["hue"]] * HUE_BINS).astype(np.int64), HUE_BINS - 1)


def perceive(sim) -> Percept:
    pop, w, clock = sim.pop, sim.world, sim.clock
    n = len(pop)
    g = pop.genome
    cy, cx = w.cell_of(pop.x, pop.y)
    night = clock.is_night()

    # warmth / light from fire and hearths
    hearth = (w.struct == L.STRUCT_HEARTH)
    warm_field = np.maximum(w.fire, hearth.astype(np.float32))
    pw = np.pad(warm_field, 1)
    warm_near = np.maximum.reduce([warm_field, pw[:-2, 1:-1], pw[2:, 1:-1], pw[1:-1, :-2], pw[1:-1, 2:]])
    warmth = warm_near[cy, cx]

    sense = P.sense(g) * (0.55 + 0.45 * (warmth > 0.1) if night else 1.0)
    sense = np.asarray(sense, dtype=np.float32)

    # ---- plants (visible) ---------------------------------------------------
    vis_bio = w.biomass * (1 - L.PLANT_HIDDEN[w.ptype])
    aq = g[:, GN.T["aquatic"]]

    def value_at(yy, xx):
        t = w.ptype[yy, xx]
        b = vis_bio[yy, xx]
        fv = np.take_along_axis(pop.food_value, t.astype(np.int64).reshape(n, -1), axis=1).reshape(t.shape)
        wat = w.water[yy, xx]
        habitat = np.where(wat, aq[:, None] if t.ndim == 2 else aq, (1 - 0.7 * aq[:, None]) if t.ndim == 2 else (1 - 0.7 * aq))
        return b * np.clip(fv, 0, None) * habitat

    food_here = value_at(cy, cx) if n else np.zeros(0, np.float32)
    rr = np.concatenate([sense[:, None] * 0.5, sense[:, None]], axis=1)  # two rings
    sx = pop.x[:, None, None] + DIRS[None, None, :, 0] * rr[:, :, None]
    sy = pop.y[:, None, None] + DIRS[None, None, :, 1] * rr[:, :, None]
    sx = sx.reshape(n, -1); sy = sy.reshape(n, -1)
    syi, sxi = w.cell_of(sx, sy)
    vals = value_at(syi, sxi) if n else np.zeros((0, 2 * N_DIR), np.float32)
    dist_pen = np.tile(np.concatenate([np.full(N_DIR, 1.0), np.full(N_DIR, 0.8)]), (n, 1))
    vals_p = vals * dist_pen
    best = vals_p.argmax(axis=1) if n else np.zeros(0, np.int64)
    best_food = vals_p[np.arange(n), best] if n else np.zeros(0, np.float32)
    dvec = np.stack([sx[np.arange(n), best] - pop.x, sy[np.arange(n), best] - pop.y], axis=1) if n else np.zeros((0, 2))
    food_dir = dvec / (np.linalg.norm(dvec, axis=1, keepdims=True) + 1e-6)

    # ---- water --------------------------------------------------------------
    water_near = (w.coast | w.water)[cy, cx]
    wmask = w.water[syi, sxi] if n else np.zeros((0, 2 * N_DIR), bool)
    has_w = wmask.any(axis=1)
    wi = np.where(wmask, 1, 0).argmax(axis=1) if n else np.zeros(0, np.int64)
    wvec = np.stack([sx[np.arange(n), wi] - pop.x, sy[np.arange(n), wi] - pop.y], axis=1) if n else np.zeros((0, 2))
    water_dir = np.where(has_w[:, None], wvec / (np.linalg.norm(wvec, axis=1, keepdims=True) + 1e-6), 0.0)
    # remembered water
    mem = pop.water_mem
    use_mem = ~has_w & (mem[:, 2] > 0)
    mv = np.stack([mem[:, 0] - pop.x, mem[:, 1] - pop.y], axis=1)
    water_dir = np.where(use_mem[:, None], mv / (np.linalg.norm(mv, axis=1, keepdims=True) + 1e-6), water_dir)

    # ---- other organisms (visible: position, size, hue, identity of kin) ------
    I, J, D = neighbor_pairs(pop.x, pop.y, radius=float(max(2.0, sense.max() if n else 2.0)))
    within = D <= sense[I]
    I, J, D = I[within], J[within], D[within]
    size = g[:, GN.T["size"]]
    hb = hue_bin(g)
    fear_learned = pop.hue_fear[I, hb[J]] if len(I) else np.zeros(0, np.float32)
    big = np.clip((size[J] / (size[I] + 1e-3) - 1.6) * 0.4, 0, 0.4)
    thr_score = np.clip(fear_learned + big, 0, 1) * (1 - D / (sense[I] + 1e-3))
    kin = ((pop.parent_a[I] == pop.id[J]) | (pop.parent_b[I] == pop.id[J]) |
           (pop.parent_a[J] == pop.id[I]) | (pop.parent_b[J] == pop.id[I]) |
           ((pop.parent_a[I] == pop.parent_a[J]) & (pop.parent_a[I] >= 0)))
    same_group = (pop.group[I] == pop.group[J]) & (pop.group[I] >= 0)
    thr_score = np.where(kin | same_group, 0.0, thr_score)
    m = thr_score > 0.05
    threat_idx, _ = nearest_by(I[m], J[m], -thr_score[m], n)
    threat = np.zeros(n, np.float32)
    np.maximum.at(threat, I[m], thr_score[m])
    tv = np.zeros((n, 2), np.float32)
    has_t = threat_idx >= 0
    tv[has_t, 0] = pop.x[threat_idx[has_t]] - pop.x[has_t]
    tv[has_t, 1] = pop.y[threat_idx[has_t]] - pop.y[has_t]
    # remembered danger contributes too
    dm = pop.danger_mem
    dd = np.hypot(dm[:, 0] - pop.x, dm[:, 1] - pop.y)
    mem_thr = np.where((dm[:, 2] > 0) & (dd < sense), dm[:, 2] * (1 - dd / (sense + 1e-3)), 0)
    use_dm = (mem_thr > threat) & (mem_thr > 0.05)
    tv[use_dm, 0] = dm[use_dm, 0] - pop.x[use_dm]
    tv[use_dm, 1] = dm[use_dm, 1] - pop.y[use_dm]
    threat = np.maximum(threat, mem_thr).astype(np.float32)
    threat_dir = tv / (np.linalg.norm(tv, axis=1, keepdims=True) + 1e-6)

    gd = GN.genetic_distance(g[I], g[J]) if len(I) else np.zeros(0, np.float32)
    diet = g[:, GN.T["diet"]]
    preym = (diet[I] > 0.35) & (size[J] < size[I] * 1.15) & (gd > 0.14) & ~kin
    enemies = sim.civ.enemy_mask(pop.group[I], pop.group[J]) if len(I) else np.zeros(0, bool)
    preym |= enemies & ~kin
    prey_idx, prey_dist = nearest_by(I[preym], J[preym], D[preym], n)

    mature = pop.age >= P.maturity(g) * pop.max_age
    ready = mature & (pop.energy > (0.3 + 0.35 * g[:, GN.T["repro_thresh"]]) * P.capacity(g)) & (pop.pregnant_until < 0)
    # a mate is a mature member of the other sex of one's own kind (kin are not excluded here:
    # avoiding relatives is a heritable instinct that evolution may or may not strengthen)
    matem = ready[I] & mature[J] & (gd < 0.12) & (pop.sex[I] != pop.sex[J]) & (pop.pregnant_until[J] < 0)
    mate_idx, mate_dist = nearest_by(I[matem], J[matem], D[matem], n)
    fm = kin | same_group
    friend_idx, friend_dist = nearest_by(I[fm], J[fm], D[fm], n)
    friends_n = np.bincount(I[fm], minlength=n).astype(np.float32)
    pm = (pop.parent_a[I] == pop.id[J]) | (pop.parent_b[I] == pop.id[J])
    parent_idx, _ = nearest_by(I[pm], J[pm], D[pm], n)

    # ---- materials (for experiments / work) --------------------------------
    inv = pop.inventory
    mat = np.zeros((n, len(OBJECTS)), dtype=bool)
    cs = w.stone[cy, cx]
    mat[:, OBJ["stone"]] = (cs >= 1) | (inv[:, L.R["stone"]] >= 1)
    mat[:, OBJ["stone2"]] = (cs >= 2) | ((cs >= 1) & (inv[:, L.R["stone"]] >= 1)) | (inv[:, L.R["stone"]] >= 2)
    tree = (w.ptype[cy, cx] == L.NUT_TREE) & (w.biomass[cy, cx] > 0.1)
    mat[:, OBJ["nut"]] = tree
    mat[:, OBJ["wood"]] = (tree & (w.biomass[cy, cx] > 0.3)) | (inv[:, L.R["wood"]] >= 1)
    mat[:, OBJ["wood2"]] = inv[:, L.R["wood"]] >= 2
    mat[:, OBJ["tool"]] = inv[:, L.R["tool"]] >= 1
    mat[:, OBJ["ground"]] = w.land[cy, cx]
    mat[:, OBJ["seed"]] = np.isin(w.ptype[cy, cx], [L.BERRY, L.NUT_TREE, L.TUBER]) & (w.biomass[cy, cx] > 0.1)
    mat[:, OBJ["dug"]] = w.dug[cy, cx]
    mat[:, OBJ["clay"]] = w.clay[cy, cx] | (inv[:, L.R["clay"]] >= 1)
    mat[:, OBJ["fire"]] = warmth > 0.1
    mat[:, OBJ["ore"]] = (inv[:, L.R["ore"]] >= 1) | (inv[:, L.R["copper_ore"]] >= 1)
    mat[:, OBJ["charcoal"]] = inv[:, L.R["charcoal"]] >= 1
    mat[:, OBJ["salt"]] = (inv[:, L.R["salt"]] >= 1) | (w.salt_amt[cy, cx] >= 1)
    kilnish = (w.struct == L.STRUCT_KILN) | (w.struct == L.STRUCT_FURNACE)
    pk = np.pad(kilnish, 1)
    near_kiln = kilnish | pk[:-2, 1:-1] | pk[2:, 1:-1] | pk[1:-1, :-2] | pk[1:-1, 2:]
    mat[:, OBJ["kiln"]] = near_kiln[cy, cx]
    mat[:, OBJ["pottery"]] = inv[:, L.R["pottery"]] >= 1
    mat[:, OBJ["sand"]] = w.sand[cy, cx] | (inv[:, L.R["sand"]] >= 1)
    mat[:, OBJ["glass"]] = inv[:, L.R["glass"]] >= 1
    mat[:, OBJ["metal"]] = inv[:, L.R["metal"]] >= 1
    mat[:, OBJ["small_thing"]] = water_near | (w.detritus[cy, cx] > 0.05)
    mat[:, OBJ["sky"]] = night
    mat[:, OBJ["words"]] = sim.lexicon_sizes() >= 6
    dry = (w.moisture[cy, cx] < 0.5) & (w.rain[cy, cx] < 0.01)

    # ---- assemble NN inputs ------------------------------------------------
    cap = P.capacity(g)
    tdev = np.abs(w.temperature[cy, cx] - P.temp_opt(g)) - 10.0 * (warmth > 0.1)
    fmem = pop.food_mem
    fmd = np.hypot(fmem[:, 0] - pop.x, fmem[:, 1] - pop.y)
    inp = np.zeros((n, GN.N_IN), np.float32)
    inp[:, 0] = 1.0
    inp[:, 1] = 1 - np.clip(pop.energy / cap, 0, 1)
    inp[:, 2] = 1 - pop.health
    inp[:, 3] = 1 - pop.hydration
    inp[:, 4] = np.clip(food_here, 0, 1.5)
    inp[:, 5] = np.clip(best_food, 0, 1.5)
    inp[:, 6] = water_near
    inp[:, 7] = threat
    inp[:, 8] = ready * np.where(mate_idx >= 0, 1.0, 1.0 - g[:, GN.T["sexual"]])
    inp[:, 9] = np.clip(friends_n / 5.0, 0, 1)
    inp[:, 10] = (prey_idx >= 0)
    inp[:, 11] = np.clip(tdev / L.TEMP_TOLERANCE, 0, 2)
    inp[:, 12] = 1.0 if night else 0.0
    inp[:, 13] = mat.sum(axis=1) / 6.0
    inp[:, 14] = pop.emotion[:, 0]
    inp[:, 15] = np.where(fmem[:, 2] > 0, np.clip(fmem[:, 2], 0, 1.5) * (fmd > 1.5), 0)

    return Percept(cy=cy, cx=cx, inputs=inp, food_here=food_here.astype(np.float32), best_food=best_food.astype(np.float32),
                   food_dir=food_dir.astype(np.float32), water_near=water_near, water_dir=water_dir.astype(np.float32),
                   threat=threat, threat_idx=threat_idx, threat_dir=threat_dir.astype(np.float32),
                   prey_idx=prey_idx, prey_dist=prey_dist, mate_idx=mate_idx, mate_dist=mate_dist,
                   friend_idx=friend_idx, friend_dist=friend_dist, parent_idx=parent_idx,
                   pairs=(I, J, D), materials=mat, dry=dry, sense=sense, warmth=warmth)
