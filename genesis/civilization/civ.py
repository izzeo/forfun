"""Civilization Engine: detection and bookkeeping of emergent social structure.

Groups, settlements, polities, trade networks, feuds and the collective
knowledge of reality are *detected* from what organisms actually do
(relationships, structures they built, skills they hold). Nothing here
creates a society by decree. Polity types follow Service's band / tribe /
chiefdom / state typology, applied as measurement criteria.
"""
from __future__ import annotations

import numpy as np

from ..biology import genome as GN
from ..intelligence import language
from ..memory import memory as MEM
from ..physics import laws as L

LAYER_SKILL = {"microscopic": "microscope", "celestial": "telescope", "subterranean": "dig"}


class Civilization:
    def __init__(self):
        self.groups: dict[int, dict] = {}
        self.next_group = 1
        self.feuds: dict[tuple[int, int], float] = {}
        self.wars: list[dict] = []
        self.trade: dict[tuple[int, int], int] = {}
        self.settlements: dict[int, dict] = {}
        self.next_settlement = 1
        self.layers: dict[str, dict] = {"macro": dict(tick=0, by=None, note="perceived directly")}
        self.skills_known: dict[str, dict] = {}
        self.group_home: dict[int, tuple[float, float]] = {}
        self.group_layers: dict[str, set] = {k: set() for k in LAYER_SKILL}
        self._hostile = np.zeros(0, np.int64)

    # ------------------------------------------------------------------ queries
    def enemy_mask(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        if len(self._hostile) == 0 or len(g1) == 0:
            return np.zeros(len(g1), bool)
        a = np.minimum(g1, g2).astype(np.int64)
        b = np.maximum(g1, g2).astype(np.int64)
        key = a * 1_000_003 + b
        return (g1 >= 0) & (g2 >= 0) & (g1 != g2) & np.isin(key, self._hostile)

    def knows_layer(self, sim, i: int, layer: str | None) -> bool:
        if layer is None:
            return False
        if layer == "macro":
            return True
        if layer not in self.layers:
            return False
        pop = sim.pop
        sk = LAYER_SKILL[layer]
        if pop.skills[i, L.SK[sk]] > 0.1:
            return True
        return int(pop.group[i]) in self.group_layers.get(layer, ())

    def hygiene(self, sim) -> np.ndarray:
        pop = sim.pop
        n = len(pop)
        h = np.zeros(n, np.float32)
        if "microscopic" in self.layers:
            knows = (pop.skills[:, L.SK["microscope"]] > 0.1) | np.isin(pop.group, list(self.group_layers["microscopic"]))
            h += 0.5 * knows
        boil = (pop.skills[:, L.SK["pottery"]] > 0.1) & (pop.skills[:, L.SK["fire"]] > 0.1)
        h += 0.15 * boil
        return h

    # ------------------------------------------------------------------ events
    def discover_skill(self, sim, i: int, sk: int, transmitted: bool = False) -> bool:
        name = L.SKILLS[sk]
        if name in self.skills_known:
            return False
        if transmitted:
            return False
        self.skills_known[name] = dict(tick=sim.clock.tick, by=int(sim.pop.id[i]), group=int(sim.pop.group[i]))
        return True

    def discover_layer(self, sim, layer: str, i: int) -> None:
        if layer in self.layers:
            return
        pop = sim.pop
        self.layers[layer] = dict(tick=sim.clock.tick, by=int(pop.id[i]), group=int(pop.group[i]),
                                  note=L.REALITY_LAYERS[layer])
        sim.history.event(sim, "layer_discovered", int(pop.id[i]), -1, pop.x[i], pop.y[i], dict(layer=layer))
        sim.analytics.first(sim, "layer:" + layer, f"Inhabitants discovered the {layer} layer of reality")
        MEM.consolidate(pop, np.array([i]), MEM.M["discovered"], 1.0, -1, sim.clock.tick,
                        detail=f"perceived the {layer} world for the first time", force=True)

    def add_feud(self, g1: int, g2: int, amt: float) -> None:
        k = (min(g1, g2), max(g1, g2))
        self.feuds[k] = self.feuds.get(k, 0.0) + amt

    def on_deaths(self, sim, idx) -> None:
        pass

    def try_trade(self, sim, i: int, j: int) -> bool:
        pop = sim.pop
        inv = pop.inventory
        give_i = [r for r in range(1, L.N_RES) if inv[i, r] >= 1 and inv[j, r] < 1]
        give_j = [r for r in range(1, L.N_RES) if inv[j, r] >= 1 and inv[i, r] < 1]
        if not give_i or not give_j:
            return False
        trust = sim.social.affinity(int(pop.id[i]), int(pop.id[j])) + 0.3 * pop.norms[i, 4] + 0.3 * pop.norms[j, 4]
        if sim.rng.random() > 0.3 + trust:
            return False
        ri, rj = give_i[0], give_j[0]
        inv[i, ri] -= 1; inv[j, ri] += 1
        inv[j, rj] -= 1; inv[i, rj] += 1
        a, b = int(pop.id[i]), int(pop.id[j])
        sim.social.adjust(a, b, 0.1); sim.social.adjust(b, a, 0.1)
        ga, gb = int(pop.group[i]), int(pop.group[j])
        k = (min(ga, gb), max(ga, gb))
        self.trade[k] = self.trade.get(k, 0) + 1
        MEM.record(pop, np.array([i, j]), MEM.M["traded"], 0.4, np.array([b, a]), sim.clock.tick)
        if self.trade[k] == 1:
            sim.history.event(sim, "trade_route", a, b, pop.x[i], pop.y[i],
                              dict(groups=list(k), goods=[L.RESOURCES[ri], L.RESOURCES[rj]]))
        sim.analytics.first(sim, "trade", f"First exchange of goods ({L.RESOURCES[ri]} for {L.RESOURCES[rj]}) between groups")
        return True

    # ------------------------------------------------------------------ periodic
    def update(self, sim) -> None:
        self._update_groups(sim)
        self._update_settlements(sim)
        self._update_polities(sim)
        self._update_conflicts(sim)

    def _update_groups(self, sim) -> None:
        pop, tick = sim.pop, sim.clock.tick
        n = len(pop)
        id2i = {int(a): k for k, a in enumerate(pop.id)}
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, rel in pop.relations.items():
            ia = id2i.get(a)
            if ia is None:
                continue
            for b, v in rel.items():
                if v <= 0.3:
                    continue
                ib = id2i.get(b)
                if ib is None:
                    continue
                if pop.relations.get(b, {}).get(a, 0.0) > 0.2:
                    ra, rb = find(ia), find(ib)
                    if ra != rb:
                        parent[ra] = rb
        comps: dict[int, list[int]] = {}
        for k in range(n):
            comps.setdefault(find(k), []).append(k)
        new_group = np.full(n, -1, np.int32)
        taken = set()
        for members in sorted(comps.values(), key=lambda m: -len(m)):
            if len(members) < 3:
                continue
            old = pop.group[members]
            old = old[old >= 0]
            gid = None
            if len(old):
                vals, cnts = np.unique(old, return_counts=True)
                for o in np.argsort(-cnts):
                    cand = int(vals[o])
                    if cand not in taken and cand in self.groups and self.groups[cand]["dissolved"] is None:
                        gid = cand
                        break
            if gid is None:
                gid = self.next_group; self.next_group += 1
                founder = members[int(np.argmax(pop.prestige[members]))]
                self.groups[gid] = dict(id=gid, name=language.invent(sim.rng).capitalize() + " folk", founded=tick,
                                        dissolved=None, founder=int(pop.id[founder]), polity="band", history=[],
                                        species=int(pop.species[founder]))
                sim.history.event(sim, "group_formed", int(pop.id[founder]), -1, pop.x[founder], pop.y[founder],
                                  dict(group=gid, size=len(members), name=self.groups[gid]["name"]))
                sim.analytics.first(sim, "group", "First stable social group formed")
            taken.add(gid)
            new_group[members] = gid
            gr = self.groups[gid]
            mem = np.array(members)
            gr["size"] = len(members)
            gr["centroid"] = [float(pop.x[mem].mean()), float(pop.y[mem].mean())]
            leader = mem[int(np.argmax(pop.prestige[mem]))]
            gr["leader"] = int(pop.id[leader])
            gr["skills"] = {L.SKILLS[s]: round(float((pop.skills[mem, s] > 0.1).mean()), 3) for s in range(L.N_SKILLS)}
            gr["norms"] = [round(float(v), 3) for v in pop.norms[mem].mean(axis=0)]
            gr["beliefs"] = [round(float(v), 3) for v in pop.beliefs[mem].mean(axis=0)]
            coh, vocab = language.coherence(pop, [int(pop.id[k]) for k in mem])
            gr["language_coherence"] = round(coh, 3)
            gr["vocabulary"] = vocab
            gr["mean_brain"] = round(float(pop.genome[mem, GN.T["brain"]].mean()), 3)
            gr["hierarchy"] = round(float((pop.prestige[mem].max() - pop.prestige[mem].mean()) / (abs(pop.prestige[mem].mean()) + 0.05)), 3)
            gr["history"].append((tick, len(members)))
            gr["history"] = gr["history"][-200:]
            if gid not in self.group_home:
                self.group_home[gid] = tuple(gr["centroid"])
        for gid, gr in self.groups.items():
            if gr["dissolved"] is None and gid not in taken:
                gr["dissolved"] = tick
                gr["size"] = 0
                sim.history.event(sim, "group_dissolved", -1, -1, 0, 0, dict(group=gid, name=gr["name"]))
        pop.group = new_group
        # collective knowledge of hidden layers
        for layer, sk in LAYER_SKILL.items():
            if layer not in self.layers:
                continue
            has = pop.skills[:, L.SK[sk]] > 0.1
            self.group_layers[layer] = set(int(g) for g in np.unique(pop.group[has]) if g >= 0)

    def _update_settlements(self, sim) -> None:
        w, pop, tick = sim.world, sim.pop, sim.clock.tick
        ys, xs = np.nonzero(w.struct > 0)
        cells = list(zip(ys.tolist(), xs.tolist()))
        cellset = set(cells)
        seen = set()
        clusters = []
        for c in cells:
            if c in seen:
                continue
            stack, comp = [c], []
            seen.add(c)
            while stack:
                cy, cx = stack.pop()
                comp.append((cy, cx))
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nb = (cy + dy, cx + dx)
                        if nb in cellset and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            kinds = [int(w.struct[y, x]) for y, x in comp]
            if len(comp) >= 3 and L.STRUCT_SHELTER in kinds:
                clusters.append((comp, kinds))
        alive = set()
        for comp, kinds in clusters:
            cs = set(comp)
            sid = None
            for k, s in self.settlements.items():
                if s["abandoned"] is None and k not in alive and len(cs & s["_cells"]) > 0:
                    sid = k
                    break
            cy = float(np.mean([c[0] for c in comp])); cx = float(np.mean([c[1] for c in comp]))
            near = np.hypot(pop.x - cx, pop.y - cy) < 6
            groups = pop.group[near]
            groups = groups[groups >= 0]
            owner = int(np.bincount(groups).argmax()) if len(groups) else -1
            if sid is None:
                sid = self.next_settlement; self.next_settlement += 1
                self.settlements[sid] = dict(id=sid, founded=tick, abandoned=None, name=language.invent(sim.rng).capitalize(),
                                             library=[0.0] * L.N_SKILLS, peak=0)
                sim.history.event(sim, "settlement_founded", -1, -1, cx, cy, dict(settlement=sid, group=owner))
                sim.analytics.first(sim, "settlement", "First settlement (cluster of dwellings) appeared")
            s = self.settlements[sid]
            s["_cells"] = cs
            s.update(center=[cx, cy], structures={L.STRUCTURES[k]: kinds.count(k) for k in set(kinds)},
                     population=int(near.sum()), group=owner)
            s["peak"] = max(s["peak"], s["population"])
            alive.add(sid)
            if owner >= 0:
                self.group_home[owner] = (cx, cy)
            # writing: a settlement's records outlive individuals
            res = np.nonzero(near)[0]
            if len(res) and (pop.skills[res, L.SK["writing"]] > 0.1).any():
                lib = np.maximum(np.array(s["library"]), pop.skills[res].max(axis=0))
                s["library"] = [round(float(v), 3) for v in lib]
                pop.skills[res] = np.maximum(pop.skills[res], 0.95 * pop.skills[res] + 0.05 * lib * 0.6)
                sim.analytics.first(sim, "library", "Knowledge recorded in writing in a settlement")
        for k, s in self.settlements.items():
            if s["abandoned"] is None and k not in alive:
                s["abandoned"] = tick
                sim.history.event(sim, "settlement_abandoned", -1, -1, *s.get("center", [0, 0]), dict(settlement=k))

    def _update_polities(self, sim) -> None:
        pop, tick = sim.pop, sim.clock.tick
        owned: dict[int, int] = {}
        for s in self.settlements.values():
            if s["abandoned"] is None and s.get("group", -1) >= 0:
                owned[s["group"]] = owned.get(s["group"], 0) + 1
        for gid, gr in self.groups.items():
            if gr["dissolved"] is not None:
                continue
            size = gr.get("size", 0)
            ns = owned.get(gid, 0)
            sk = gr.get("skills", {})
            if size >= 60 and ns >= 2 and sk.get("writing", 0) > 0.05:
                pol = "state"
            elif size >= 35 and ns >= 1 and gr.get("hierarchy", 0) > 1.5:
                pol = "chiefdom"
            elif size >= 15 and (ns >= 1 or sk.get("cultivate", 0) > 0.15):
                pol = "tribe"
            else:
                pol = "band"
            gr["settlements"] = ns
            if pol != gr["polity"]:
                sim.history.event(sim, "polity_change", gr.get("leader", -1), -1, *gr["centroid"],
                                  dict(group=gid, from_=gr["polity"], to=pol))
                gr["polity"] = pol
                sim.analytics.first(sim, "polity:" + pol, f"First {pol} emerged ({gr['name']})")

    def _update_conflicts(self, sim) -> None:
        pop = sim.pop
        forgiving = float(np.clip(pop.norms[:, 5].mean(), -1, 1)) if len(pop) else 0.0
        decay = 0.9 - 0.08 * forgiving
        hostile = []
        for k in list(self.feuds):
            v = self.feuds[k] * decay
            active = [w for w in self.wars if w["groups"] == list(k) and w["ended"] is None]
            if v > 4 and not active:
                minds = [self.groups.get(gid, {}).get("mean_brain", 0) for gid in k]
                kind = "war" if min(minds) > 0.3 else "territorial_hostility"
                self.wars.append(dict(groups=list(k), started=sim.clock.tick, ended=None, kind=kind))
                sim.history.event(sim, "conflict_started", -1, -1, 0, 0, dict(groups=list(k), kind=kind))
                sim.analytics.first(sim, kind, "First sustained conflict between groups" if kind == "war"
                                    else "First sustained hostility between animal groups")
            if v < 1 and active:
                for w in active:
                    w["ended"] = sim.clock.tick
                sim.history.event(sim, "peace", -1, -1, 0, 0, dict(groups=list(k)))
            if v < 0.05:
                del self.feuds[k]
                continue
            self.feuds[k] = v
            if v > 2:
                hostile.append(k[0] * 1_000_003 + k[1])
        self._hostile = np.array(hostile, np.int64)

    def summary(self) -> dict:
        live = [g for g in self.groups.values() if g["dissolved"] is None]
        pol = {}
        for g in live:
            pol[g["polity"]] = pol.get(g["polity"], 0) + 1
        return dict(groups=len(live), settlements=sum(1 for s in self.settlements.values() if s["abandoned"] is None),
                    polities=pol, wars=sum(1 for w in self.wars if w["ended"] is None and w.get("kind") == "war"),
                    trade_links=len(self.trade), layers=sorted(self.layers), skills_known=sorted(self.skills_known))

    def state_dict(self):
        d = dict(self.__dict__)
        d["_hostile"] = self._hostile.copy()
        return d

    def load_state(self, d):
        self.__dict__.update(d)
