"""Memory System.

* Short-term memory: a ring buffer of the last few events per organism (vectorised).
* Long-term memory: salient episodes are consolidated into a per-individual list
  whose capacity grows with brain size; the least salient (after time decay)
  are forgotten first.
* Semantic memory: learned food values, places (food / danger / water),
  skills and habits live in Population arrays and are updated by intelligence.
"""
from __future__ import annotations

import math

import numpy as np

from ..biology import genome as GN
from ..biology.population import STM_K, Population

MEM_TYPES = ["none", "ate", "poisoned", "attacked", "hunted", "gave_birth", "mated", "discovered",
             "learned", "shared", "received", "lost_kin", "escaped", "storm", "ill", "traded",
             "built", "heard_word", "ritual", "born", "starving", "eclipse", "fire", "taught",
             "friend", "first_taste"]
M = {n: i for i, n in enumerate(MEM_TYPES)}

CONSOLIDATE_THRESHOLD = 0.35


def record(pop: Population, idx: np.ndarray, etype: int, mag, other, tick: int) -> None:
    """Write events into short-term memory (vectorised)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return
    mag = np.broadcast_to(np.asarray(mag, dtype=np.float32), idx.shape)
    other = np.broadcast_to(np.asarray(other, dtype=np.float32), idx.shape)
    slot = pop.stm_ptr[idx] % STM_K
    pop.stm[idx, slot, 0] = tick
    pop.stm[idx, slot, 1] = etype
    pop.stm[idx, slot, 2] = mag
    pop.stm[idx, slot, 3] = other
    np.add.at(pop.stm_ptr, idx, 1)


def consolidate(pop: Population, idx: np.ndarray, etype: int, mag, other, tick: int,
                detail: str | None = None, force: bool = False) -> int:
    """Move salient events into long-term memory. Returns number consolidated."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return 0
    mag = np.broadcast_to(np.asarray(mag, dtype=np.float32), idx.shape)
    other = np.broadcast_to(np.asarray(other, dtype=np.float32), idx.shape)
    arousal = np.abs(pop.emotion[idx]).max(axis=1)
    sal = np.abs(mag) * (0.6 + arousal)
    sel = np.nonzero(force | (sal > CONSOLIDATE_THRESHOLD))[0]
    for k in sel:
        i = int(idx[k])
        aid = int(pop.id[i])
        brain = float(pop.genome[i, GN.T["brain"]])
        cap = int(6 + 44 * brain)
        lst = pop.ltm.setdefault(aid, [])
        lst.append(dict(tick=int(tick), type=MEM_TYPES[etype], mag=round(float(mag[k]), 3),
                        other=int(other[k]), x=round(float(pop.x[i]), 1), y=round(float(pop.y[i]), 1),
                        salience=round(float(max(sal[k], 0.4 if force else 0.0)), 3), detail=detail))
        if len(lst) > cap:
            tau = 1500.0 * (0.5 + brain)
            scores = [m["salience"] * math.exp(-(tick - m["tick"]) / tau) for m in lst]
            drop = int(np.argmin(scores))
            lst.pop(drop)
    return len(sel)


def short_term(pop: Population, i: int) -> list[dict]:
    out = []
    ptr = int(pop.stm_ptr[i])
    for k in range(min(ptr, STM_K)):
        slot = (ptr - 1 - k) % STM_K
        t, e, m, o = pop.stm[i, slot]
        out.append(dict(tick=int(t), type=MEM_TYPES[int(e)], mag=round(float(m), 3), other=int(o)))
    return out


def long_term(pop: Population, agent_id: int, tick: int) -> list[dict]:
    lst = pop.ltm.get(agent_id, [])
    i = pop.index_of(agent_id)
    brain = float(pop.genome[i, GN.T["brain"]]) if i is not None else 0.3
    tau = 1500.0 * (0.5 + brain)
    out = []
    for m in lst:
        mm = dict(m)
        mm["vividness"] = round(m["salience"] * math.exp(-(tick - m["tick"]) / tau), 3)
        out.append(mm)
    return out


def recent_count(pop: Population, idx: np.ndarray, etype: int, since_tick: int) -> np.ndarray:
    """How many events of a type each organism remembers in short-term memory since a tick."""
    s = pop.stm[idx]
    return ((s[:, :, 1] == etype) & (s[:, :, 0] >= since_tick)).sum(axis=1)
