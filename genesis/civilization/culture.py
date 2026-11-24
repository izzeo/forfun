"""Beliefs and cultural transmission.

Belief model (cognitive science of religion, e.g. agency detection & costly
signalling theories, heavily simplified):

    beliefs[0] spirits     — attributing unexplained harm to unseen agents
    beliefs[1] naturalism  — attributing events to observed natural causes
    beliefs[2] ritual      — propensity to perform shared rituals

When a cognitively capable organism suffers harm it cannot explain with its
own (or its group's) knowledge, belief in unseen agency grows. Once the
cause becomes observable (e.g. microbes via microscope, the sky via
telescope), the same events strengthen naturalistic explanation instead.
Beliefs spread by conformist and prestige-biased transmission. Nothing here
asserts the truth or falsity of any real-world religion.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..memory import memory as MEM

CAUSE_LAYER = {"disease": "microscopic", "storm": "celestial", "eclipse": "celestial"}


class Culture:
    def unexplained_harm(self, sim, idx, salience: float, kind: str) -> None:
        pop = sim.pop
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return
        capable = pop.genome[idx, GN.T["brain"]] > 0.25
        idx = idx[capable]
        if idx.size == 0:
            return
        layer = CAUSE_LAYER.get(kind)
        explained = np.array([sim.civ.knows_layer(sim, int(i), layer) for i in idx]) if layer else np.zeros(len(idx), bool)
        un = idx[~explained]
        ex = idx[explained]
        b = pop.beliefs
        b[un, 0] = np.minimum(1, b[un, 0] + 0.08 * salience * (1 - 0.7 * b[un, 1]))
        b[un, 2] = np.minimum(1, b[un, 2] + 0.03 * salience * b[un, 0])
        b[ex, 1] = np.minimum(1, b[ex, 1] + 0.06 * salience)
        b[ex, 0] = np.maximum(0, b[ex, 0] - 0.02 * salience)
        if len(un) and (b[un, 0] > 0.5).any():
            sim.analytics.first(sim, "belief:spirits", f"Belief in unseen agents emerged (after {kind})")
        if len(ex):
            sim.analytics.first(sim, "belief:naturalism", f"A natural explanation of {kind} took hold")

    def transmit(self, sim, I, J) -> None:
        """Prestige-biased cultural transmission of beliefs and norms from J to I."""
        pop = sim.pop
        if len(I) == 0:
            return
        pres = np.clip(pop.prestige[J] - pop.prestige[I], 0, 2)
        rate = (0.01 + 0.03 * pres) * (0.3 + pop.genome[I, GN.T["sociality"]])
        pop.beliefs[I] += rate[:, None] * (pop.beliefs[J] - pop.beliefs[I])
        pop.norms[I] += 0.5 * rate[:, None] * (pop.norms[J] - pop.norms[I])

    def ritual(self, sim, i: int, j: int) -> bool:
        pop = sim.pop
        if pop.beliefs[i, 2] < 0.4 or pop.beliefs[j, 2] < 0.4:
            return False
        sim.social.adjust(int(pop.id[i]), int(pop.id[j]), 0.12)
        sim.social.adjust(int(pop.id[j]), int(pop.id[i]), 0.12)
        for k in (i, j):
            pop.emotion[k, 0] *= 0.5
            pop.emotion[k, 3] *= 0.7
            pop.energy[k] -= 0.003  # costly signal
        sim.world.ledger.heat_out += 0.006
        MEM.record(pop, np.array([i, j]), MEM.M["ritual"], 0.3, -1, sim.clock.tick)
        sim.analytics.first(sim, "ritual", "First shared ritual performed")
        return True
