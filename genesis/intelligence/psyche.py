"""Affective & personality model (appraisal-based).

Emotions here are *functional state variables* (fear, joy, anger, sadness,
affection) produced by appraisal of events and used to modulate behaviour.
The model makes no claim that these organisms have subjective experience;
it is a computational analogue whose behavioural effects can be studied.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..memory import memory as MEM

DECAY = np.array([0.90, 0.95, 0.93, 0.995, 0.99], np.float32)  # per-hour persistence


class Psyche:
    def update(self, sim, pc, reward: np.ndarray) -> None:
        pop = sim.pop
        if len(pop) == 0:
            return
        e = pop.emotion
        e *= DECAY
        sens = 0.5 + pop.personality[:, 4]          # neuroticism amplifies fear
        e[:, 0] = np.maximum(e[:, 0], pc.threat * sens)
        e[:, 1] += np.clip(reward, 0, None) * 0.8 * (1.5 - pop.personality[:, 4])
        e[:, 4] += 0.02 * pc.inputs[:, 9] * pop.genome[:, GN.T["sociality"]]
        np.clip(e, 0, 1, out=e)
        # slow drift of personality through experience (plasticity declines with age)
        plast = 0.0005 * np.clip(1.2 - pop.age / np.maximum(pop.max_age, 1), 0.1, 1.2)
        pop.personality[:, 4] += plast * (e[:, 0] - 0.2)          # chronic fear -> neuroticism
        pop.personality[:, 2] += plast * (e[:, 4] - 0.2)          # warm company -> extraversion
        pop.personality[:, 3] += plast * (0.1 - e[:, 2])          # anger -> less agreeable
        np.clip(pop.personality, 0, 1, out=pop.personality)

    def on_deaths(self, sim, idx: np.ndarray) -> None:
        """Grief: survivors bonded with the dead become sad and remember the loss."""
        pop, tick = sim.pop, sim.clock.tick
        if len(idx) == 0:
            return
        id2i = None
        for i in idx:
            did = int(pop.id[i])
            rel = pop.relations.get(did)
            if not rel:
                continue
            dead_set = set(int(d) for d in idx)
            for other, aff in rel.items():
                if aff < 0.3:
                    continue
                if id2i is None:
                    id2i = {int(a): k for k, a in enumerate(pop.id)}
                k = id2i.get(other)
                if k is None or k in dead_set or pop.genome[k, GN.T["brain"]] < 0.15:
                    continue
                pop.emotion[k, 3] = min(1.0, pop.emotion[k, 3] + aff)
                pop.emotion[k, 4] = max(0.0, pop.emotion[k, 4] - 0.2)
                MEM.record(pop, np.array([k]), MEM.M["lost_kin"], -aff, did, tick)
                MEM.consolidate(pop, np.array([k]), MEM.M["lost_kin"], -aff * 1.5, did, tick,
                                detail=f"lost {pop.names.get(did, 'a companion')}", force=aff > 0.5)
                if pop.partner[k] == did:
                    pop.partner[k] = -1
                sim.analytics.first(sim, "grief", "An organism showed a grief response to the death of a bonded companion")
