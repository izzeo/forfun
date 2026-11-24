"""Modular moral-orientation system.

Each organism carries a vector of seven learned *norm weights* in [-1, 1].
They are cultural, not genetic: children receive them from parents and their
group, and they drift with personal experience (being helped, being harmed).
Norms bias concrete behaviours:

    justice        fair sharing; sanctioning aggressors within the group
    mercy          helping the weak, sick or starving, including non-kin
    patience       sustainable harvesting (leave a seed stock), less impulsive
    responsibility caring for offspring, teaching, maintaining shared work
    truthfulness   not deceiving others about food
    forgiveness    recovery of damaged relationships, ending feuds
    gratitude      reciprocating help

A *framework* is an educational ideal that elders try to pass on to children
with strength ``cfg.moral_education``. Frameworks are offered for
comparative experiments; none is imposed by default ("none" lets norms
emerge purely from experience). The Qur'an-inspired profile is a simplified,
respectful research abstraction of virtues the Qur'an emphasises (ʿadl,
raḥma, ṣabr, amāna, ṣidq, ʿafw, shukr). It is not a theological claim, and
reducing ethics to seven numbers is a strong simplification.
"""
from __future__ import annotations

import numpy as np

VIRTUES = ["justice", "mercy", "patience", "responsibility", "truthfulness", "forgiveness", "gratitude"]

FRAMEWORKS: dict[str, dict] = {
    "none": dict(
        description="No moral education. Norms emerge only from individual experience and imitation.",
        ideal=None),
    "quran_inspired": dict(
        description=("Virtues emphasised in the Qur'an, as a research abstraction: justice (ʿadl, 4:135, 16:90), "
                     "mercy (raḥma, 21:107), patience (ṣabr, 2:153), responsibility/trust (amāna, 33:72), "
                     "truthfulness (ṣidq, 9:119), forgiveness (ʿafw, 24:22, 42:40) and gratitude (shukr, 14:7)."),
        ideal=[0.9, 0.9, 0.8, 0.85, 0.9, 0.8, 0.85]),
    "utilitarian": dict(
        description="Maximise aggregate welfare: high mercy/sharing, moderate everything else.",
        ideal=[0.6, 0.8, 0.5, 0.5, 0.6, 0.5, 0.4]),
    "kin_loyalty": dict(
        description="Strong duties toward one's own family, little regard for outsiders.",
        ideal=[0.3, -0.2, 0.4, 0.9, 0.3, -0.3, 0.6]),
    "egoist": dict(
        description="Self-interest: no obligation to help, deceive when useful, never forgive.",
        ideal=[-0.3, -0.6, 0.0, -0.2, -0.6, -0.6, -0.3]),
}


class MoralSystem:
    def __init__(self, framework: str = "none", strength: float = 0.3):
        if framework not in FRAMEWORKS:
            raise ValueError(f"unknown moral framework {framework!r}; options: {list(FRAMEWORKS)}")
        self.framework = framework
        self.strength = float(strength)

    @property
    def ideal(self):
        v = FRAMEWORKS[self.framework]["ideal"]
        return None if v is None else np.array(v, np.float32)

    def educate_newborn(self, sim, ids, parent_idx) -> None:
        """Children of parents embedded in a group receive moral education."""
        ideal = self.ideal
        if ideal is None or self.strength <= 0:
            return
        pop = sim.pop
        m = len(ids)
        n0 = len(pop) - m
        child = np.arange(n0, n0 + m)
        in_culture = (pop.group[parent_idx] >= 0) | (pop.genome[parent_idx, 13] > 0.3)
        k = self.strength * np.where(in_culture, 1.0, 0.3)[:, None]
        pop.norms[child] = pop.norms[child] + k * (ideal[None, :] - pop.norms[child])

    def ongoing_education(self, sim, I, J) -> None:
        """Elders/mentors (J) reinforce the ideal in juveniles (I) during interactions."""
        ideal = self.ideal
        if ideal is None or len(I) == 0:
            return
        pop = sim.pop
        rate = 0.02 * self.strength * (0.5 + np.clip(pop.norms[J, 3], 0, 1))[:, None]
        pop.norms[I] += rate * (ideal[None, :] - pop.norms[I])

    @staticmethod
    def experience(pop, idx, virtue: str, delta: float) -> None:
        """Norm drift through personal experience (applies under every framework)."""
        k = VIRTUES.index(virtue)
        pop.norms[idx, k] = np.clip(pop.norms[idx, k] + delta, -1, 1)
