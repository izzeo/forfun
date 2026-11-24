"""Spatial partitioning: a uniform-grid hash for fully vectorised neighbour search."""
from __future__ import annotations

import numpy as np


def neighbor_pairs(x: np.ndarray, y: np.ndarray, radius: float, cell: float | None = None):
    """Return directed pairs (i, j, d) with i != j and dist(i, j) <= radius.

    Agents are binned into cells of size >= radius; each agent is compared only
    against agents in the 3x3 surrounding bins (O(N * local density)).
    """
    n = len(x)
    if n < 2:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0, dtype=np.float32)
    cs = float(cell or radius)
    bx = np.floor(x / cs).astype(np.int64)
    by = np.floor(y / cs).astype(np.int64)
    nby = int(by.max()) + 3
    key = (bx + 1) * nby + (by + 1)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    I, J = [], []
    idx = np.arange(n)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            k2 = key + dx * nby + dy
            lo = np.searchsorted(skey, k2, "left")
            hi = np.searchsorted(skey, k2, "right")
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            ii = np.repeat(idx, cnt)
            offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            jj = order[np.repeat(lo, cnt) + offs]
            I.append(ii); J.append(jj)
    if not I:
        e = np.zeros(0, dtype=np.int64)
        return e, e, np.zeros(0, dtype=np.float32)
    I = np.concatenate(I); J = np.concatenate(J)
    m = I != J
    I, J = I[m], J[m]
    d = np.hypot(x[I] - x[J], y[I] - y[J]).astype(np.float32)
    m = d <= radius
    return I[m], J[m], d[m]


def nearest_by(i: np.ndarray, j: np.ndarray, d: np.ndarray, n: int):
    """For each source index, the nearest target. Returns (target[-1 if none], dist[inf])."""
    tgt = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf, dtype=np.float32)
    if len(i) == 0:
        return tgt, dist
    o = np.lexsort((d, i))
    i2, j2, d2 = i[o], j[o], d[o]
    first = np.ones(len(i2), dtype=bool)
    first[1:] = i2[1:] != i2[:-1]
    tgt[i2[first]] = j2[first]
    dist[i2[first]] = d2[first]
    return tgt, dist
