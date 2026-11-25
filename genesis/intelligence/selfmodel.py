"""Self-model: what an organism believes about its own abilities.

Two learned quantities (Bandura's self-efficacy and Seligman's learned
helplessness, heavily simplified):

* ``self_eff[i, d]`` — belief "my efforts succeed" in five domains
  (fight, forage, explore, craft, social), learned from one's own outcomes,
  from watching similar others succeed (vicarious learning), and slightly
  from upbringing.
* ``agency`` — a general sense of control. Repeated failure and harm the
  organism cannot avoid (disease, storms, being beaten) lower it; success,
  help from others and being taught raise it.

Both bias decisions. A low belief is a *soft limiter*: an organism that is
now perfectly able to win a fight or invent something may still not try,
because it learned earlier that "this is not for me". The belief only changes
if it tries again, sees someone like itself succeed, or is supported by others.
Beliefs can therefore be wrong: we measure their calibration against reality.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..memory import memory as MEM
from ..physics import laws as L

DOMAINS = ["fight", "forage", "explore", "craft", "social"]
D = {n: i for i, n in enumerate(DOMAINS)}
A = L.A
AGENCY_BASE = 0.6
HELPLESS = 0.25


def update(sim, act, attacks, gain, r_exp, r_work, harm, support) -> None:
    """Learn from this hour's own outcomes (vectorised)."""
    pop = sim.pop
    n = len(pop)
    if n == 0:
        return
    young = (pop.age < 0.25 * pop.max_age)
    lr = np.where(young, 0.10, 0.05).astype(np.float32)
    succ = np.zeros((n, len(DOMAINS)), np.float32)
    fail = np.zeros((n, len(DOMAINS)), np.float32)
    # fighting: the attacker learns from hitting or missing, the victim from being overpowered
    for a, t, hit, _ in attacks:
        (succ if hit else fail)[a, D["fight"]] += 1
        if hit:
            fail[t, D["fight"]] += 1
    eating = (act == A["eat"]) | (act == A["seek_food"])
    succ[eating & (gain > 0.03), D["forage"]] += 1
    fail[(act == A["eat"]) & (gain < 0.005), D["forage"]] += 1
    exp = act == A["experiment"]
    succ[exp & (r_exp >= 0.3), D["explore"]] += 3          # a discovery is a strong lesson
    fail[exp & (r_exp < 0.3), D["explore"]] += 0.15         # most trials fail; each failure teaches little
    wk = act == A["work"]
    succ[wk & (r_work >= 0.05), D["craft"]] += 1
    fail[wk & (r_work < 0.05), D["craft"]] += 1
    soc = act == A["social"]
    succ[soc & (support > 0), D["social"]] += 1
    fail[soc & (support <= 0), D["social"]] += 0.1
    tried = (succ + fail) > 0
    target = succ / np.maximum(succ + fail, 1e-6)
    w = np.minimum(1.0, succ + fail)
    pop.self_eff += np.where(tried, lr[:, None] * w * (target - pop.self_eff), 0).astype(np.float32)
    np.clip(pop.self_eff, 0.0, 1.0, out=pop.self_eff)

    # general sense of control
    # routine successes (finding food) say little about control; real challenges say more
    weight = np.array([1.0, 0.05, 1.0, 0.5, 0.1], np.float32)
    s = np.minimum((succ * weight).sum(axis=1), 2.0)
    f = np.minimum((fail * weight).sum(axis=1), 2.0)
    before = pop.agency.copy()
    gain_ = 0.02 * s + 0.04 * np.minimum(support, 2)
    loss_ = 0.02 * f + 0.6 * np.minimum(harm, 0.4)
    # bounded learning: confidence is hard to raise when high and hard to lower when low
    delta = gain_ * (1 - pop.agency) - loss_ * pop.agency * (0.6 + 0.8 * pop.personality[:, 4])
    baseline = AGENCY_BASE + 0.2 - 0.4 * pop.personality[:, 4]   # anxious temperaments sit lower
    delta += 0.002 * (baseline - pop.agency)              # return towards a temperament baseline
    pop.agency = np.clip(pop.agency + delta, 0, 1).astype(np.float32)

    # notable moments become memories and "firsts"
    tick = sim.clock.tick
    fell = np.nonzero((before >= HELPLESS) & (pop.agency < HELPLESS))[0]
    if len(fell):
        pop.was_helpless[fell] = True
        MEM.consolidate(pop, fell, MEM.M["ill"], -0.8, -1, tick, detail="stopped believing anything it did would help",
                        force=True)
        sim.analytics.first(sim, "learned_helplessness",
                            "An organism became passive after repeated failures it could not control (learned helplessness)")
    rose = np.nonzero(pop.was_helpless & (before < 0.5) & (pop.agency >= 0.5))[0]
    if len(rose):
        pop.was_helpless[rose] = False
        MEM.consolidate(pop, rose, MEM.M["friend"], 0.8, -1, tick, detail="regained the belief that its efforts matter",
                        force=True)
        sim.analytics.first(sim, "overcame_helplessness", "An organism recovered from learned helplessness")


def vicarious(sim, I: np.ndarray, J: np.ndarray) -> None:
    """'If someone like me can, so can I': watching similar others raises self-belief."""
    pop = sim.pop
    if len(I) == 0:
        return
    similar = (pop.group[I] == pop.group[J]) & (pop.group[I] >= 0) | (pop.parent_a[I] == pop.id[J]) | (pop.parent_b[I] == pop.id[J])
    I, J = I[similar], J[similar]
    if len(I) == 0:
        return
    gap = np.clip(pop.self_eff[J] - pop.self_eff[I], 0, None)
    pop.self_eff[I] += (0.02 * gap).astype(np.float32)
    pop.agency[I] += (0.01 * np.clip(pop.agency[J] - pop.agency[I], 0, None)).astype(np.float32)


def bias(pop, logits: np.ndarray) -> None:
    """How beliefs about oneself shape what one attempts."""
    e = pop.self_eff
    logits[:, A["attack"]] += 2.0 * (e[:, D["fight"]] - 0.5)
    logits[:, A["flee"]] -= 1.0 * (e[:, D["fight"]] - 0.5)
    logits[:, A["experiment"]] += 1.5 * (e[:, D["explore"]] - 0.4)
    logits[:, A["work"]] += 1.5 * (e[:, D["craft"]] - 0.5)
    logits[:, A["social"]] += 1.0 * (e[:, D["social"]] - 0.5)
    # learned helplessness: effortful behaviour collapses, passivity rises
    low = np.clip(0.4 - pop.agency, 0, None) * 5
    for a in ("experiment", "work", "attack", "social", "mate", "wander"):
        logits[:, A[a]] -= low
    logits[:, A["rest"]] += low


def calibration(pop, sim) -> dict:
    """How well beliefs match reality: correlation of fight self-belief with actual strength."""
    if len(pop) < 10:
        return {}
    from ..intelligence.skills import tool_power
    strength = pop.body_mass() * (0.5 + 0.5 * pop.health) * (1 + 0.4 * tool_power(pop))
    rank = strength.argsort().argsort() / max(1, len(pop) - 1)
    b = pop.self_eff[:, D["fight"]]
    if b.std() < 1e-6 or rank.std() < 1e-6:
        return dict(fight_belief_vs_strength=0.0)
    c = np.corrcoef(b, rank)[0, 1]
    return dict(fight_belief_vs_strength=round(float(c), 3))
