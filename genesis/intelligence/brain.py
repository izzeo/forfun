"""Agent Intelligence: decision making and learning.

Three layers, cheapest first:

1. A tiny genetically-encoded neural network (16 -> 8 -> 12) computes action
   urges from the current percept. It is shaped only by evolution.
2. Individually *learned* modulators: habits (operant conditioning on the
   reward each action produced), learned food values, learned fear of
   appearances, place memories. Two genetically identical organisms raised in
   different places therefore behave differently.
3. A deliberative planner that runs only every few ticks and only for the
   most cognitively capable individuals (model-based evaluation of remembered
   goals). This is the "heavier model used selectively" — no LLM is involved.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..biology.genome import Phenotype as P
from ..physics import laws as L

A = L.A


def instinct_weights(rng: np.random.Generator, n: int, noise: float = 0.35) -> np.ndarray:
    """Primitive reflexes (chemotaxis-like) the first organisms start with, plus noise."""
    W1 = np.zeros((GN.N_IN, GN.N_HID), np.float32)
    W2 = np.zeros((GN.N_HID, GN.N_OUT), np.float32)
    b = np.zeros(GN.N_OUT, np.float32)
    # hidden units:  (input index, weight) ... ; input 0 is bias
    spec = {
        0: {1: 3.0, 0: -1.0},                 # hungry
        1: {4: 2.5, 1: 1.5, 0: -1.5},         # food here & hungry
        2: {3: 3.0, 0: -1.2},                 # thirsty
        3: {7: 4.0, 0: -0.5},                 # threatened
        4: {8: 3.0, 0: -1.0},                 # mate available
        5: {9: 2.0, 12: 1.0, 0: -1.0},        # company
        6: {2: 2.0, 12: 1.5, 0: -1.5},        # tired / hurt
        7: {10: 2.0, 1: 1.5, 0: -2.0},        # prey & hungry
    }
    for h, ins in spec.items():
        for i, wv in ins.items():
            W1[i, h] = wv
    W2[0, A["seek_food"]] = 1.5; W2[1, A["seek_food"]] = -1.0
    W2[1, A["eat"]] = 2.5
    W2[3, A["flee"]] = 3.0
    W2[6, A["rest"]] = 2.0
    W2[2, A["drink"]] = 2.5
    W2[4, A["mate"]] = 2.5
    W2[7, A["attack"]] = 2.0
    W2[5, A["social"]] = 1.0
    W2[0, A["recall"]] = 0.8
    b[:] = [0.2, 0.0, -0.5, -1.0, -0.5, -0.8, -1.0, -1.5, -0.8, -1.5, -1.2, -1.5]
    base = np.concatenate([W1.ravel(), W2.ravel(), b])
    return (base[None, :] + rng.normal(0, noise, (n, GN.N_W))).astype(np.float32)


def forward(weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    n = len(inputs)
    k1 = GN.N_IN * GN.N_HID
    k2 = k1 + GN.N_HID * GN.N_OUT
    W1 = weights[:, :k1].reshape(n, GN.N_IN, GN.N_HID)
    W2 = weights[:, k1:k2].reshape(n, GN.N_HID, GN.N_OUT)
    b = weights[:, k2:]
    h = np.tanh(np.einsum("ni,nih->nh", inputs, W1))
    return np.einsum("nh,nho->no", h, W2) + b


def life_stage(pop) -> np.ndarray:
    """0 infant, 1 child, 2 adolescent, 3 adult, 4 elder."""
    f = pop.age / np.maximum(pop.max_age, 1)
    mat = P.maturity(pop.genome)
    st = np.full(len(pop), 3, np.int8)
    st[f < mat] = 2
    st[f < mat * 0.66] = 1
    st[f < mat * 0.3] = 0
    st[f > 0.78] = 4
    return st


def choose(sim, pc) -> np.ndarray:
    pop, rng, tick = sim.pop, sim.rng, sim.clock.tick
    n = len(pop)
    if n == 0:
        return np.zeros(0, np.int64)
    g = pop.genome
    logits = forward(pop.weights, pc.inputs)
    logits += pop.habit
    O, C, E, Ag, N = (pop.personality[:, k] - 0.5 for k in range(5))
    logits[:, A["wander"]] += 0.8 * O
    logits[:, A["experiment"]] += 1.2 * O + 2.0 * g[:, GN.T["curiosity"]] * g[:, GN.T["brain"]] - 0.5
    logits[:, A["work"]] += 0.8 * C + 2.5 * (pop.skills.max(axis=1) > 0.05) * (0.5 + C)
    logits[:, A["social"]] += 0.8 * E + 1.2 * g[:, GN.T["sociality"]] * (pc.friend_idx >= 0)
    logits[:, A["attack"]] += -1.0 * Ag + 2.0 * (g[:, GN.T["diet"]] - 0.5) + 1.0 * (g[:, GN.T["aggression"]] - 0.5)
    logits[:, A["flee"]] += 1.0 * N
    emo = pop.emotion
    logits[:, A["flee"]] += 2.0 * emo[:, 0]
    logits[:, A["attack"]] += 1.5 * emo[:, 2]
    logits[:, A["rest"]] += 1.0 * emo[:, 3]
    logits[:, A["social"]] += 0.5 * emo[:, 3] + 0.8 * emo[:, 4] + 0.3 * emo[:, 1]
    logits[:, A["work"]] += 0.6 * pop.norms[:, 3]  # responsibility
    # beliefs about oneself: what one dares to attempt
    from . import selfmodel
    selfmodel.bias(pop, logits)
    # behavioural persistence: keep doing what one was doing until the goal is met
    last = pop.last_action.astype(np.int64)
    done = ((last == A["drink"]) & (pop.hydration > 0.9)) | ((last == A["eat"]) & (pc.inputs[:, 1] < 0.1)) \
        | ((last == A["flee"]) & (pc.threat < 0.05)) | ((last == A["rest"]) & (pc.inputs[:, 2] < 0.05) & (pc.inputs[:, 12] < 0.5))
    logits[np.arange(n), last] += np.where(done, 0.0, 1.2)
    # homeostatic priority: urgent bodily needs crowd out optional behaviour
    urgent = np.clip(np.maximum.reduce([pc.inputs[:, 1], pc.inputs[:, 3], pc.inputs[:, 2]]) - 0.5, 0, None) * 4
    for a in ("social", "experiment", "work", "mate", "wander"):
        logits[:, A[a]] -= urgent
    logits[:, A["drink"]] += 2 * np.clip(pc.inputs[:, 3] - 0.6, 0, None)
    # deliberate intentions
    act_int = pop.intent[:, 3] > tick
    if act_int.any():
        ii = np.nonzero(act_int)[0]
        logits[ii, pop.intent[ii, 2].astype(np.int64)] += 1.5
    # life stage constraints (physical capability, not choice)
    st = life_stage(pop)
    young = st <= 1
    logits[st <= 2, A["mate"]] = -50
    logits[young, A["attack"]] -= 3
    logits[st == 0, A["work"]] = -50
    logits[st == 0, A["experiment"]] -= 1  # infants still play/explore a bit
    logits[young & (pc.parent_idx >= 0), A["social"]] += 1.5  # stay close to a parent
    # sample (Gumbel-max == softmax sampling)
    act = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)
    return act


def learn_habits(sim, actions: np.ndarray, reward: np.ndarray) -> None:
    pop = sim.pop
    n = len(pop)
    if n == 0:
        return
    g = pop.genome
    st = life_stage(pop)
    lr = 0.15 * g[:, GN.T["learn_rate"]] * (0.2 + g[:, GN.T["brain"]]) * np.where(st <= 2, 1.8, 1.0)
    adv = reward - pop.avg_reward
    pop.habit[np.arange(n), actions] += (lr * adv).astype(np.float32)
    pop.habit *= np.float32(0.9995)
    np.clip(pop.habit, -2, 2, out=pop.habit)
    pop.avg_reward += (0.05 * (reward - pop.avg_reward)).astype(np.float32)


def deliberate(sim, pc) -> int:
    """Model-based planning for the few organisms whose brains can afford it."""
    pop, tick = sim.pop, sim.clock.tick
    g = pop.genome
    brain = g[:, GN.T["brain"]]
    cand = np.nonzero(brain > 0.5)[0]
    if len(cand) == 0:
        return 0
    cand = cand[np.argsort(-brain[cand])][:96]
    cap = P.capacity(g)
    hunger = 1 - pop.energy / cap
    thirst = 1 - pop.hydration
    cold = np.clip(np.abs(sim.world.temperature[pc.cy, pc.cx] - P.temp_opt(g)) / L.TEMP_TOLERANCE - 0.5, 0, 2)
    homes = sim.civ.group_home
    for i in cand:
        opts = []
        fm = pop.food_mem[i]
        if fm[2] > 0:
            d = float(np.hypot(fm[0] - pop.x[i], fm[1] - pop.y[i]))
            dm = pop.danger_mem[i]
            risk = float(dm[2]) if np.hypot(dm[0] - fm[0], dm[1] - fm[1]) < 3 else 0.0
            opts.append((fm[2] * hunger[i] * 1.5 - 0.02 * d - risk, fm[0], fm[1], A["recall"]))
        wm = pop.water_mem[i]
        if wm[2] > 0 and thirst[i] > 0.3:
            opts.append((thirst[i] * 1.5, wm[0], wm[1], A["drink"]))
        h = homes.get(int(pop.group[i]))
        if h is not None:
            opts.append((cold[i] * 0.6 + 0.3 * sim.clock.is_night(), h[0], h[1], A["recall"]))
        opts.append((float(pc.food_here[i]) * hunger[i], pop.x[i], pop.y[i], A["eat"]))
        opts.append((0.25 * float(g[i, GN.T["curiosity"]]), pop.x[i], pop.y[i], A["wander"]))
        best = max(opts, key=lambda o: o[0])
        pop.intent[i] = (best[1], best[2], best[3], tick + sim.cfg.deliberation_interval)
    return len(cand)
