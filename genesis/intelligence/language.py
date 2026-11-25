"""Emergent communication: a grounded naming game (Steels, 1995-2015).

Organisms with enough brain and communicative drive emit sound-forms
("words") about things both partners can perceive. No vocabulary is given:
words are invented at random, and a shared lexicon emerges (or does not)
through repeated local interactions with lateral inhibition. Communication is
useful only because it changes what the hearer knows (food sites, danger,
how to do something), so it is subject to natural selection.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..memory import memory as MEM
from ..physics import laws as L

CONCEPTS = [f"plant:{p['name']}" for p in L.PLANTS] + ["danger", "water"] + [f"skill:{s}" for s in L.SKILLS]
N_CONCEPTS = len(CONCEPTS)
C_DANGER = L.N_PLANTS
C_WATER = L.N_PLANTS + 1
C_SKILL0 = L.N_PLANTS + 2
_CONS = list("ptkmnslrvbdgh")
_VOW = list("aeiou")


def can_talk(pop, i: int) -> bool:
    g = pop.genome[i]
    return g[GN.T["communication"]] > 0.3 and g[GN.T["brain"]] > 0.2


def invent(rng) -> str:
    k = int(rng.integers(1, 4))
    return "".join(_CONS[int(rng.integers(len(_CONS)))] + _VOW[int(rng.integers(len(_VOW)))] for _ in range(k))


def top_word(lex: dict, concept: int) -> str | None:
    d = lex.get(concept)
    if not d:
        return None
    return max(d.items(), key=lambda kv: (kv[1], kv[0]))[0]


def choose_topic(sim, pc, i: int) -> int:
    pop = sim.pop
    if pc.threat[i] > 0.3 and pc.threat_idx[i] >= 0:
        return C_DANGER
    if pc.food_here[i] > 0.15:
        return int(sim.world.ptype[pc.cy[i], pc.cx[i]])
    known = np.nonzero(pop.skills[i] > 0.15)[0]
    if len(known) and sim.rng.random() < 0.5:
        return C_SKILL0 + int(sim.rng.choice(known))
    if pc.water_near[i]:
        return C_WATER
    return int(sim.world.ptype[pc.cy[i], pc.cx[i]])


def converse(sim, pc, i: int, j: int) -> bool:
    """One naming-game round, speaker i, hearer j. Returns communicative success."""
    pop, rng, tick = sim.pop, sim.rng, sim.clock.tick
    si, sj = int(pop.id[i]), int(pop.id[j])
    lex_i = pop.lexicon.setdefault(si, {})
    lex_j = pop.lexicon.setdefault(sj, {})
    topic = choose_topic(sim, pc, i)
    cap_i = 3 + int(17 * pop.genome[i, GN.T["brain"]])
    word = top_word(lex_i, topic)
    if word is None:
        if len(lex_i) >= cap_i:
            return False
        word = invent(rng)
        lex_i[topic] = {word: 0.5}
    # honesty: a speaker may withhold/mislead about food if it is low on truthfulness
    lying = topic < L.N_PLANTS and pop.norms[i, 4] < -0.3 and rng.random() < 0.3 * pop.genome[i, GN.T["aggression"]]
    known = lex_j.get(topic, {})
    success = word in known
    if success:
        for lex in (lex_i, lex_j):
            d = lex[topic]
            d[word] = min(1.0, d.get(word, 0) + 0.1)
            for w2 in [w for w in d if w != word]:
                d[w2] -= 0.2
                if d[w2] <= 0:
                    del d[w2]
    else:
        cap_j = 3 + int(17 * pop.genome[j, GN.T["brain"]])
        if topic in lex_j or len(lex_j) < cap_j:
            lex_j.setdefault(topic, {})[word] = 0.5
        if word in lex_i.get(topic, {}):
            lex_i[topic][word] = max(0.05, lex_i[topic][word] - 0.1)
    # informational effect of the utterance on the hearer
    gain = 1.0 if success else 0.3
    if rng.random() < gain:
        if topic == C_DANGER:
            pop.danger_mem[j] = (pop.x[i], pop.y[i], 0.5)
            pop.emotion[j, 0] = min(1.0, pop.emotion[j, 0] + 0.15)
        elif topic < L.N_PLANTS:
            v = float(pop.food_value[i, topic])
            if lying:
                pop.food_mem[j] = (pop.danger_mem[i, 0], pop.danger_mem[i, 1], 0.6)
                if rng.random() < 0.5:  # deception discovered
                    sim.social.adjust(sj, si, -0.25)
            elif v > pop.food_mem[j, 2]:
                pop.food_mem[j] = (pop.x[i], pop.y[i], v)
            # hearing about a food's quality shapes the hearer's expectation (social learning of taste)
            pop.food_value[j, topic] += np.float32(0.15 * (v - pop.food_value[j, topic]))
        elif topic == C_WATER:
            pop.water_mem[j] = (pop.x[i], pop.y[i], 1.0)
    if success:
        heard = getattr(sim, "_heard", None)
        if heard is not None:
            heard.append((j, topic))           # written to memory in one batch by the caller
        else:
            MEM.record(pop, np.array([j]), MEM.M["heard_word"], 0.2, topic, tick)
        if "communication" not in sim.analytics.firsts:
            sim.analytics.first(sim, "communication", f"First successful exchange of a word ('{word}' for {CONCEPTS[topic]})")
    return success


def coherence(pop, ids: list[int]) -> tuple[float, dict]:
    """Share of individuals using the modal word, averaged over concepts in use."""
    by_c: dict[int, dict[str, int]] = {}
    for a in ids:
        lex = pop.lexicon.get(a)
        if not lex:
            continue
        for c in lex:
            w = top_word(lex, c)
            if w:
                by_c.setdefault(c, {}).setdefault(w, 0)
                by_c[c][w] += 1
    if not by_c:
        return 0.0, {}
    scores, vocab = [], {}
    for c, counts in by_c.items():
        tot = sum(counts.values())
        if tot < 2:
            continue
        w, k = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
        scores.append(k / tot)
        vocab[CONCEPTS[c]] = dict(word=w, speakers=k, share=round(k / tot, 3))
    return (float(np.mean(scores)) if scores else 0.0), vocab
