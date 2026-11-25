"""Social behaviour: relationships, care, sharing, teaching, talk, sanction, trade.

Relationships are sparse, individual and asymmetric (A's affinity to B need
not equal B's to A). They emerge from interactions; nothing assigns friends.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..intelligence import language, skills
from ..intelligence.brain import life_stage
from ..memory import memory as MEM
from ..physics import laws as L
from .morality import MoralSystem

A = L.A


class Social:
    def __init__(self, sim):
        self.sim = sim

    # ---- relationship store -------------------------------------------------
    def affinity(self, a: int, b: int) -> float:
        return self.sim.pop.relations.get(a, {}).get(b, 0.0)

    def adjust(self, a: int, b: int, delta: float) -> float:
        pop = self.sim.pop
        rel = pop.relations.setdefault(a, {})
        v = rel.get(b, 0.0) + float(delta)
        v = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
        rel[b] = v
        if len(rel) > 48:   # forget the weakest acquaintances (amortised: trim back to 40 at once)
            for k in sorted(rel, key=lambda k: (abs(rel[k]), k))[: len(rel) - 40]:
                del rel[k]
        return v

    def bond(self, a: int, b: int, v: float) -> None:
        self.adjust(a, b, v)
        self.adjust(b, a, v)

    # ---- per-tick interactions ----------------------------------------------
    def interact(self, pc, act, attacks) -> dict:
        sim = self.sim
        pop, rng, tick = sim.pop, sim.rng, sim.clock.tick
        n = len(pop)
        stats = dict(shared=0, talks=0, talk_success=0, taught=0, trades=0, rituals=0, care=0)
        if n < 2:
            return stats
        g = pop.genome
        cap = P.capacity(g)
        st = life_stage(pop)

        # -- parental care (vectorised): feed own young nearby ---------------
        young = np.nonzero((st <= 1) & (pc.parent_idx >= 0))[0]
        if len(young):
            par = pc.parent_idx[young]
            close = np.hypot(pop.x[par] - pop.x[young], pop.y[par] - pop.y[young]) < 1.6
            surplus = pop.energy[par] - 0.35 * cap[par]
            care = (0.25 + 0.5 * g[par, GN.T["parental"]] + 0.25 * np.clip(pop.norms[par, 3], 0, 1))
            need = (pop.energy[young] < 0.6 * cap[young])
            amt = np.where(close & need & (surplus > 0), np.minimum(0.02 * care * cap[par], surplus), 0).astype(np.float32)
            np.subtract.at(pop.energy, par, amt)
            pop.energy[young] += amt * 0.95
            nursed = close & (st[young] == 0)
            pop.hydration[young[nursed]] = np.minimum(1.0, pop.hydration[young[nursed]] + 0.02)
            sim.world.ledger.heat_out += float(amt.sum() * 0.05)
            stats["care"] = int((amt > 0).sum())

        # -- consequences of violence -------------------------------------------
        for a, t, hit, cause in attacks:
            ida, idt = int(pop.id[a]), int(pop.id[t])
            self.adjust(idt, ida, -0.4 if hit else -0.15)
            ga, gt = int(pop.group[a]), int(pop.group[t])
            if ga >= 0 and gt >= 0 and ga != gt:
                sim.civ.add_feud(ga, gt, 1.0 if hit else 0.3)
            MoralSystem.experience(pop, [t], "forgiveness", -0.02)
            MoralSystem.experience(pop, [t], "justice", 0.01)
            if ga >= 0 and ga == gt:
                # in-group aggression is sanctioned by members who value justice
                pop.prestige[a] -= 0.1
                sim.analytics.first(sim, "sanction", "Group members sanctioned an aggressor")

        # -- pairwise interactions among close neighbours -------------------------
        I, J, D = pc.pairs
        m = (D < 2.0) & ((act[I] == A["social"]) | (st[I] <= 1))
        I, J = I[m], J[m]
        if len(I) > sim.cfg.max_social_pairs:
            sel = rng.choice(len(I), sim.cfg.max_social_pairs, replace=False)
            sel.sort()
            I, J = I[sel], J[sel]
        if len(I) == 0:
            return stats
        # vectorised: imitation, cultural transmission, moral education by elders
        shared_words = np.ones(len(I), np.float32)
        stats["taught"] = skills.imitate(sim, I, J, shared_words)
        from ..intelligence import selfmodel
        selfmodel.vicarious(sim, I, J)
        sim.culture.transmit(sim, I, J)
        juvenile = st[I] <= 2
        sim.moral.ongoing_education(sim, I[juvenile], J[juvenile])

        soc = g[:, GN.T["sociality"]].tolist()
        forgive = pop.norms[:, 5].tolist()
        ids = pop.id.tolist()
        pa_l, pb_l = pop.parent_a.tolist(), pop.parent_b.tolist()
        talker = ((g[:, GN.T["communication"]] > 0.3) & (g[:, GN.T["brain"]] > 0.2)).tolist()
        ritualist = (pop.beliefs[:, 2] >= 0.4).tolist()
        grp = pop.group.tolist()
        sim._heard = []
        for i, j in zip(I.tolist(), J.tolist()):
            si, sj = ids[i], ids[j]
            aff_ij = self.affinity(si, sj)
            # grooming / companionship
            self.adjust(si, sj, 0.03 * (0.3 + soc[i]))
            self.adjust(sj, si, 0.02 * (0.3 + soc[j]))
            if aff_ij < 0:
                self.adjust(si, sj, 0.03 * max(0.0, forgive[i]))  # forgiveness
            kin = pa_l[j] == si or pb_l[j] == si or pa_l[i] == sj or pb_l[i] == sj
            # sharing food: kin selection, reciprocity, mercy
            if pop.energy[i] > 0.55 * cap[i] and pop.energy[j] < 0.3 * cap[j]:
                will = (0.6 * kin * g[i, GN.T["parental"]] + 0.5 * max(aff_ij, 0) + 0.5 * pop.norms[i, 1]
                        + 0.2 * pop.norms[i, 0] + 0.3 * pop.norms[i, 6] * (self.affinity(sj, si) > 0.3)
                        - 0.3 * g[i, GN.T["aggression"]])
                if rng.random() < will:
                    amt = float(min(0.15 * cap[i], pop.energy[i] - 0.45 * cap[i]))
                    if amt > 0:
                        pop.energy[i] -= amt
                        pop.energy[j] += amt * 0.9
                        sim.world.ledger.heat_out += amt * 0.1
                        grat = 0.5 + max(0.0, float(pop.norms[j, 6]))
                        self.adjust(sj, si, 0.2 * grat)
                        MoralSystem.experience(pop, [j], "gratitude", 0.02)
                        MoralSystem.experience(pop, [j], "mercy", 0.01)
                        MEM.record(pop, np.array([i]), MEM.M["shared"], amt, sj, tick)
                        MEM.record(pop, np.array([j]), MEM.M["received"], amt, si, tick)
                        MEM.consolidate(pop, np.array([j]), MEM.M["received"], 0.6, si, tick,
                                        detail=f"was given food by {pop.names.get(si, 'someone')}")
                        stats["shared"] += 1
                        sim.support[j] += 1.0
                        sim.support[i] += 0.5
                        pop.prestige[i] += 0.02
                        sim.analytics.first(sim, "sharing" if kin else "altruism_nonkin",
                                            "Food shared with kin" if kin else "Food shared with a non-relative")
            # talk
            if talker[i] and talker[j]:
                stats["talks"] += 1
                if language.converse(sim, pc, i, j):
                    stats["talk_success"] += 1
                    self.adjust(sj, si, 0.02)
            # ritual
            if ritualist[i] and ritualist[j] and sim.culture.ritual(sim, i, j):
                stats["rituals"] += 1
                sim.support[i] += 0.5
                sim.support[j] += 0.5
            # trade between members of different groups
            gi, gj = grp[i], grp[j]
            if gi >= 0 and gj >= 0 and gi != gj and aff_ij > -0.1:
                if sim.civ.try_trade(sim, i, j):
                    stats["trades"] += 1
            # pair bonds
            if self.affinity(si, sj) > 0.7 and self.affinity(sj, si) > 0.7 and pop.partner[i] == -1 and pop.partner[j] == -1 \
                    and st[i] >= 3 and st[j] >= 3 and not kin:
                pop.partner[i], pop.partner[j] = sj, si
                MEM.consolidate(pop, np.array([i, j]), MEM.M["friend"], 0.9, np.array([sj, si]), tick, detail="formed a lasting bond")
                sim.analytics.first(sim, "pair_bond", "First lasting pair bond formed")
        if sim._heard:
            hj = np.array([h[0] for h in sim._heard]); ht = np.array([h[1] for h in sim._heard])
            MEM.record(pop, hj, MEM.M["heard_word"], 0.2, ht, tick)
        sim._heard = None
        pop.prestige = (pop.prestige * 0.999 + 0.001 * (pop.skills.sum(axis=1) + 0.5 * (st == 4))).astype(np.float32)
        return stats
