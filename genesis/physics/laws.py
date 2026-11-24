"""Laws of the GENESIS universe.

Everything here is *the physics of the world*, not knowledge given to its
inhabitants. Organisms never read these tables directly: they only see the
visible consequences (colours, biomass they can see, the pain of a poisoned
stomach) and must learn the rest through their own experience.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Flora. Biomass is measured in energy units; "nutrition" is the fraction a
# generalist herbivore can assimilate (<= 1, so energy is conserved).
# "visible_color" is what a creature can perceive. "nutrition",
# "toxicity", "locked" and "hidden_fraction" are hidden physical properties.
# locked: fraction of the energy that can only be released with a skill.
# --------------------------------------------------------------------------
PLANTS = [
    # name        colour (rgb)            nutr  tox   K     locked  unlock_skill  hidden_frac  aquatic
    dict(name="grass",      color=(0.45, 0.72, 0.30), nutrition=0.50, toxicity=0.0, K=1.0, locked=0.0, unlock=None,         hidden=0.0, aquatic=False),
    dict(name="berry",      color=(0.86, 0.22, 0.26), nutrition=0.85, toxicity=0.0, K=0.6, locked=0.0, unlock=None,         hidden=0.0, aquatic=False),
    dict(name="nightshade", color=(0.62, 0.16, 0.48), nutrition=0.70, toxicity=0.9, K=0.6, locked=0.0, unlock=None,         hidden=0.0, aquatic=False),
    dict(name="nut_tree",   color=(0.50, 0.36, 0.18), nutrition=0.95, toxicity=0.0, K=1.2, locked=0.85, unlock="crack_nuts", hidden=0.0, aquatic=False),
    dict(name="algae",      color=(0.18, 0.58, 0.55), nutrition=0.45, toxicity=0.0, K=0.8, locked=0.0, unlock=None,         hidden=0.0, aquatic=True),
    dict(name="tuber",      color=(0.58, 0.64, 0.34), nutrition=0.90, toxicity=0.0, K=1.0, locked=0.9, unlock="dig",        hidden=0.8, aquatic=False),
]
N_PLANTS = len(PLANTS)
PLANT_NUTRITION = np.array([p["nutrition"] for p in PLANTS], dtype=np.float32)
PLANT_TOXICITY = np.array([p["toxicity"] for p in PLANTS], dtype=np.float32)
PLANT_K = np.array([p["K"] for p in PLANTS], dtype=np.float32)
PLANT_LOCKED = np.array([p["locked"] for p in PLANTS], dtype=np.float32)
PLANT_HIDDEN = np.array([p["hidden"] for p in PLANTS], dtype=np.float32)
PLANT_COLORS = np.array([p["color"] for p in PLANTS], dtype=np.float32)
GRASS, BERRY, NIGHTSHADE, NUT_TREE, ALGAE, TUBER = range(N_PLANTS)

# --------------------------------------------------------------------------
# Materials / inventory
# --------------------------------------------------------------------------
RESOURCES = ["food", "stone", "wood", "clay", "ore", "tool", "pottery", "metal", "glass", "sand"]
R = {n: i for i, n in enumerate(RESOURCES)}
N_RES = len(RESOURCES)
CARRY_LIMIT = np.array([3, 3, 4, 2, 2, 1, 1, 1, 1, 2], dtype=np.float32)

# --------------------------------------------------------------------------
# Skills — each is a *physical affordance* that exists in the world whether
# or not anybody knows about it. Creatures acquire them only by experiment
# (trying a verb on objects present) or by learning from someone who knows.
# --------------------------------------------------------------------------
SKILLS = ["crack_nuts", "knap_tool", "dig", "fire", "shelter", "cultivate",
          "pottery", "smelting", "glassmaking", "microscope", "telescope", "writing"]
SK = {n: i for i, n in enumerate(SKILLS)}
N_SKILLS = len(SKILLS)

VERBS = ["strike", "dig", "combine", "heat", "stack", "plant", "shape", "look_through"]

# (verb, required objects, base difficulty, extra condition key)
RECIPES = {
    "crack_nuts":  ("strike", ("stone", "nut"), 0.30, None),
    "knap_tool":   ("strike", ("stone", "stone2"), 0.20, None),
    "dig":         ("dig", ("ground", "tool"), 0.30, None),
    "fire":        ("strike", ("stone", "wood", "tool"), 0.10, "dry"),
    "fire_capture": ("combine", ("fire", "wood"), 0.35, None),   # learn fire from a wildfire
    "shelter":     ("stack", ("wood", "wood2", "tool"), 0.25, None),
    "cultivate":   ("plant", ("seed", "ground", "dug"), 0.20, None),
    "pottery":     ("heat", ("clay", "fire"), 0.20, None),
    "smelting":    ("heat", ("ore", "fire", "pottery"), 0.12, None),
    "glassmaking": ("heat", ("sand", "fire", "pottery"), 0.12, None),
    "microscope":  ("look_through", ("glass", "metal", "small_thing"), 0.08, None),
    "telescope":   ("look_through", ("glass", "metal", "sky"), 0.08, "night"),
    "writing":     ("shape", ("clay", "words", "pottery"), 0.06, None),
}
RECIPE_SKILL = {k: ("fire" if k == "fire_capture" else k) for k in RECIPES}

# --------------------------------------------------------------------------
# Actions available to an organism's decision system
# --------------------------------------------------------------------------
ACTIONS = ["wander", "seek_food", "eat", "flee", "rest", "drink", "mate",
           "attack", "social", "experiment", "recall", "work"]
A = {n: i for i, n in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

# --------------------------------------------------------------------------
# Causes of death — every disappearance must have one of these.
# --------------------------------------------------------------------------
DEATH_CAUSES = ["starvation", "old_age", "predation", "disease", "poisoning",
                "exposure", "drowning", "dehydration", "violence", "storm", "fire"]

# --------------------------------------------------------------------------
# Structures built on the grid
# --------------------------------------------------------------------------
STRUCT_NONE, STRUCT_SHELTER, STRUCT_FARM, STRUCT_SHRINE, STRUCT_HEARTH, STRUCT_WORKSHOP = range(6)
STRUCTURES = ["none", "shelter", "farm", "shrine", "hearth", "workshop"]

# --------------------------------------------------------------------------
# Layers of reality. Inhabitants start knowing only "macro".
# --------------------------------------------------------------------------
REALITY_LAYERS = {
    "macro": "Visible surface world: terrain, plants, animals, weather.",
    "subterranean": "What lies under the ground: tubers, clay beds, ore veins.",
    "microscopic": "Microbes and pathogens that cause disease and decay.",
    "celestial": "The planet's sun, moon, axial tilt and other wandering lights.",
}

# --------------------------------------------------------------------------
# Physical constants (units: energy units "eu", hours, cells)
# --------------------------------------------------------------------------
BASAL_COST = 0.0035          # eu/hour for unit body
MOVE_COST = 0.0025           # eu per (body * cell^2)
BRAIN_COST = 0.6             # multiplier on basal for brain=1
EAT_BITE = 0.35              # max biomass per hour for unit size
PREDATION_EFF = 0.6          # fraction of prey energy assimilated
DIGEST_EFF_MISMATCH = 0.35   # carnivores digest plants poorly and vice versa
DETRITUS_DECAY = 0.01        # per hour
TEMP_TOLERANCE = 11.0         # degC each side of optimum
