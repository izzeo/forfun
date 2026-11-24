"""Fresh water: rivers and lakes derived from the terrain.

Rain falls on the land (more on mountains: orographic precipitation) and flows
downhill along the steepest descent (D8 flow routing). Cells that collect
enough upstream runoff become rivers; land depressions that cannot drain
collect water into lakes. Nothing is placed by hand: the river network is a
consequence of the relief.
"""
from __future__ import annotations

import numpy as np

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def flow_routing(elev: np.ndarray, sea: np.ndarray, precip: np.ndarray):
    """Return (accumulated runoff, downstream index or -1, sink mask)."""
    H, W = elev.shape
    flat = elev.ravel()
    down = np.full(H * W, -1, np.int64)
    sink = np.zeros(H * W, bool)
    for y in range(H):
        for x in range(W):
            i = y * W + x
            if sea[y, x]:
                continue
            best, bi = flat[i], -1
            for dy, dx in _NB:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W:
                    j = yy * W + xx
                    if flat[j] < best:
                        best, bi = flat[j], j
            down[i] = bi
            sink[i] = bi < 0
    acc = np.where(sea.ravel(), 0.0, precip.ravel()).astype(np.float64)
    for i in np.argsort(-flat, kind="stable"):  # high to low
        j = down[i]
        if j >= 0 and not sea.ravel()[i]:
            acc[j] += acc[i]
    return acc.reshape(H, W), down.reshape(H, W), sink.reshape(H, W)


def priority_flood(elev: np.ndarray, sea: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Fill depressions so that every land cell drains to the sea (Barnes et al., 2014)."""
    import heapq
    H, W = elev.shape
    filled = elev.astype(np.float64).copy()
    done = np.zeros((H, W), bool)
    heap = []
    border = sea.copy()
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    for y, x in zip(*np.nonzero(border)):
        heapq.heappush(heap, (filled[y, x], int(y), int(x)))
        done[y, x] = True
    while heap:
        e, y, x = heapq.heappop(heap)
        for dy, dx in _NB:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and not done[yy, xx]:
                done[yy, xx] = True
                filled[yy, xx] = max(filled[yy, xx], e + eps)
                heapq.heappush(heap, (filled[yy, xx], yy, xx))
    return filled


def compute(elev: np.ndarray, sea: np.ndarray, lat: np.ndarray, sea_level: float, quantile: float = 0.945):
    """Rivers (bool), lakes (bool) and flow accumulation for a heightmap."""
    land = ~sea
    height = np.clip((elev - sea_level) / (1 - sea_level), 0, 1)
    precip = (0.6 + 0.9 * height + 0.3 * (1 - np.abs(lat))) * land
    filled = priority_flood(elev, sea)
    acc, down, sink = flow_routing(filled, sea, precip)
    if land.sum() == 0:
        z = np.zeros_like(sea)
        return z, z.copy(), acc
    thr = max(np.quantile(acc[land], quantile), 18.0)
    # lakes: depressions that the filling had to flood noticeably and that receive water
    depth = filled - elev
    lake = land & (depth > 0.008) & (acc >= thr * 0.3)
    river = land & (acc >= thr) & ~lake
    return river, lake, acc


def apply(world) -> None:
    """Attach rivers/lakes to a World (also used to upgrade worlds saved before hydrology existed)."""
    river, lake, acc = compute(world.elev, world.elev < world.cfg.sea_level, world.lat, world.cfg.sea_level)
    world.river = river
    world.lake = lake
    world.flow = acc.astype(np.float32)
    fresh = river | lake
    water_any = world.water | lake
    # drinkable: next to sea, a lake, or on/next to a river
    p = np.pad(fresh | water_any, 1)
    near = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    world.coast = (~world.water) & near
    world.fresh = fresh


def distance_to(mask: np.ndarray, steps: int = 14) -> np.ndarray:
    dist = np.where(mask, 0.0, 30.0)
    for _ in range(steps):
        p = np.pad(dist, 1, mode="edge")
        dist = np.minimum(dist, np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]) + 1)
    return dist.astype(np.float32)
