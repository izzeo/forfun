"""Struct-of-arrays storage for all living organisms.

Vectorised state lives in NumPy arrays (one row per organism). Sparse,
individual structures (long-term memories, relationships, lexicons) live in
dicts keyed by the organism's permanent id, so they survive array compaction.
"""
from __future__ import annotations

import numpy as np

from ..physics import laws as L
from . import genome as GN

STM_K = 8  # short-term memory slots

FIELDS = {
    # name: (dtype, trailing shape, default)
    "id": (np.int64, (), -1),
    "x": (np.float32, (), 0.0),
    "y": (np.float32, (), 0.0),
    "heading": (np.float32, (), 0.0),
    "energy": (np.float32, (), 1.0),
    "health": (np.float32, (), 1.0),
    "hydration": (np.float32, (), 1.0),
    "age": (np.int32, (), 0),
    "max_age": (np.float32, (), 500.0),
    "generation": (np.int32, (), 0),
    "parent_a": (np.int64, (), -1),
    "parent_b": (np.int64, (), -1),
    "birth_tick": (np.int64, (), 0),
    "last_repro": (np.int64, (), -10_000),
    "species": (np.int32, (), 0),
    "group": (np.int32, (), -1),
    "partner": (np.int64, (), -1),
    "genome": (np.float32, (GN.G,), 0.5),
    "weights": (np.float32, (GN.N_W,), 0.0),
    # learning & memory
    "food_value": (np.float32, (L.N_PLANTS,), 0.5),
    "habit": (np.float32, (L.N_ACTIONS,), 0.0),
    "avg_reward": (np.float32, (), 0.0),
    "skills": (np.float32, (L.N_SKILLS,), 0.0),
    "inventory": (np.float32, (L.N_RES,), 0.0),
    "stm": (np.float32, (STM_K, 4), 0.0),      # tick, type, magnitude, other
    "stm_ptr": (np.int32, (), 0),
    "food_mem": (np.float32, (3,), 0.0),       # x, y, value
    "danger_mem": (np.float32, (3,), 0.0),
    "water_mem": (np.float32, (3,), 0.0),
    "intent": (np.float32, (4,), 0.0),         # x, y, action, expires_tick
    "explore_bias": (np.float32, (len(L.VERBS),), 0.0),
    "hue_fear": (np.float32, (8,), 0.0),       # learned fear of creatures by appearance
    # psyche
    "emotion": (np.float32, (5,), 0.0),        # fear, joy, anger, sadness, affection
    "personality": (np.float32, (5,), 0.5),    # O C E A N
    "norms": (np.float32, (7,), 0.0),          # see civilization.morality.VIRTUES
    "beliefs": (np.float32, (3,), 0.0),        # spirits, naturalism, ritual
    # health
    "infection": (np.float32, (), 0.0),
    "immunity": (np.float32, (), 0.0),
    "last_action": (np.int8, (), 0),
    "kills": (np.int32, (), 0),
    "children": (np.int32, (), 0),
    "prestige": (np.float32, (), 0.0),
    "dug_here": (np.int32, (), -1),            # cell index the organism last dug at
    "growth": (np.float32, (), 1.0),           # fraction of adult body built so far
    # material qualities of what is carried (properties follow from chemistry)
    "wood_moist": (np.float32, (), 0.45),      # moisture of carried wood (dries in air)
    "tool_q": (np.float32, (), 0.0),           # effectiveness of the stone tool (rock hardness)
    "tool_dur": (np.float32, (), 1.0),         # remaining durability of the stone tool
    "metal_q": (np.float32, (), 0.0),          # effectiveness of the metal tool (copper / iron)
    "metal_dur": (np.float32, (), 1.0),
    "glass_q": (np.float32, (), 0.0),          # clarity of the glass (sand purity)
    "sand_q": (np.float32, (), 0.85),          # purity of carried sand
    "stone_rock": (np.int8, (), 0),            # rock type of carried stone
    # self-model (see intelligence.selfmodel)
    "self_eff": (np.float32, (5,), 0.5),       # belief in own success: fight, forage, explore, craft, social
    "agency": (np.float32, (), 0.6),           # general sense of control (low = learned helplessness)
    "was_helpless": (np.bool_, (), False),
    # sexes, pregnancy, kinship (see evolution.evolution)
    "sex": (np.int8, (), -1),                  # 0 female, 1 male (-1: not yet assigned, old saves)
    "pregnant_until": (np.int64, (), -1),      # tick of birth, -1 if not pregnant
    "grand": (np.int64, (4,), -1),             # ids of the four grandparents
    "inbreeding": (np.float32, (), 0.0),       # inbreeding coefficient F of this organism
    "kin_aversion": (np.float32, (), 0.3),     # heritable reluctance to mate with relatives
    # disease strains (see biology.lifecycle)
    "strain": (np.float32, (), 0.5),           # virulence of the strain currently infecting it
    "immune_strain": (np.float32, (), 0.5),    # strain its immune memory is tuned to
}

EMO = {"fear": 0, "joy": 1, "anger": 2, "sadness": 3, "affection": 4}


class Population:
    def __init__(self):
        for name, (dt, shp, dflt) in FIELDS.items():
            setattr(self, name, np.full((0,) + shp, dflt, dtype=dt))
        self.next_id = 1
        # sparse per-individual structures
        self.ltm: dict[int, list] = {}
        self.relations: dict[int, dict[int, float]] = {}
        self.lexicon: dict[int, dict[int, dict[str, float]]] = {}
        self.names: dict[int, str] = {}
        self.pending: dict[int, dict] = {}     # pregnancies: mother id -> father's contribution

    def __len__(self) -> int:
        return len(self.id)

    def add(self, n: int, **values) -> np.ndarray:
        """Append n organisms; unspecified fields take defaults. Returns their ids."""
        ids = np.arange(self.next_id, self.next_id + n, dtype=np.int64)
        self.next_id += n
        for name, (dt, shp, dflt) in FIELDS.items():
            if name == "id":
                new = ids
            elif name in values:
                new = np.asarray(values[name], dtype=dt)
                if new.shape != (n,) + shp:
                    new = np.broadcast_to(new, (n,) + shp).astype(dt)
            else:
                new = np.full((n,) + shp, dflt, dtype=dt)
            setattr(self, name, np.concatenate([getattr(self, name), new]))
        return ids

    def remove(self, mask: np.ndarray) -> None:
        keep = ~mask
        for name in FIELDS:
            setattr(self, name, getattr(self, name)[keep])

    def index_of(self, agent_id: int) -> int | None:
        w = np.nonzero(self.id == agent_id)[0]
        return int(w[0]) if len(w) else None

    def forget(self, agent_id: int) -> None:
        """Drop sparse structures of a dead organism (its record survives in history)."""
        self.pending.pop(agent_id, None)      # an unborn child dies with its mother
        self.ltm.pop(agent_id, None)
        self.relations.pop(agent_id, None)
        self.lexicon.pop(agent_id, None)

    # convenient trait views
    def trait(self, name: str) -> np.ndarray:
        return self.genome[:, GN.T[name]]

    def body_mass(self) -> np.ndarray:
        """Energy currently locked in body tissue (grows from birth to maturity)."""
        return GN.Phenotype.body(self.genome) * self.growth

    def state_dict(self) -> dict:
        d = {name: getattr(self, name) for name in FIELDS}
        d.update(next_id=self.next_id, ltm=self.ltm, relations=self.relations,
                 lexicon=self.lexicon, names=self.names, pending=self.pending)
        return d

    def load_state(self, d: dict) -> None:
        """Load saved state, upgrading saves from older versions (new fields / new skills)."""
        n = len(d["id"])
        for name, (dt, shp, dflt) in FIELDS.items():
            arr = d.get(name)
            if arr is None:
                arr = np.full((n,) + shp, dflt, dtype=dt)
            elif arr.shape[1:] != shp:
                new = np.full((n,) + shp, dflt, dtype=dt)
                k = min(arr.shape[1], shp[0]) if shp else 0
                new[:, :k] = arr[:, :k]
                arr = new
            setattr(self, name, arr)
        self.next_id = d["next_id"]
        self.ltm, self.relations, self.lexicon, self.names = d["ltm"], d["relations"], d["lexicon"], d["names"]
        self.pending = d.get("pending", {})


_SYL = ["ka", "ri", "mo", "tu", "le", "sa", "no", "vi", "da", "ze", "lu", "ha", "po", "mi", "ra", "ten", "ob", "is", "ul", "an"]


def make_name(agent_id: int) -> str:
    """Deterministic name from the id (no Python hash randomisation)."""
    x = (agent_id * 2654435761) & 0xFFFFFFFF
    parts = []
    for _ in range(2 + (x % 2)):
        parts.append(_SYL[x % len(_SYL)])
        x //= len(_SYL)
        x = (x * 1103515245 + 12345) & 0xFFFFFFFF
    return "".join(parts).capitalize()
