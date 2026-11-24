"""Chemistry and heat: substances, their properties, fire temperature and
threshold reactions.

Every material is a mixture of a few elements. Its properties (hardness,
whether it burns, whether it sparks, how clear a glass made from it is)
are *computed from the composition*, not listed per item. Fire temperature
follows from the fuel (carbon content, moisture) and the setup (open fire,
enclosed kiln, kiln with forced draught). A transformation happens only if
the temperature it needs is actually reached:

    clay  --600 C-->  ceramic                 (pottery)
    wood  --300 C, little air-->  charcoal    (pyrolysis in a closed kiln)
    copper ore + charcoal  --1085 C-->  copper  (smelting)
    iron ore + charcoal    --1250 C-->  iron    (bloomery; needs forced air)
    sand + salt (flux)     --1000 C-->  glass   (pure sand alone needs ~1700 C)

So technology has an order that nobody scripted: pottery is reachable with
dry wood on an open fire, copper only once charcoal and kilns exist, iron
only with a draught furnace. Numbers are simplified but in realistic ranges.
"""
from __future__ import annotations

import math

import numpy as np

ELEMENTS = ["C", "H2O", "Si", "Al", "Ca", "Fe", "Cu", "Na", "N"]
E = {n: i for i, n in enumerate(ELEMENTS)}


def comp(**kw) -> np.ndarray:
    v = np.zeros(len(ELEMENTS), np.float32)
    for k, x in kw.items():
        v[E[k]] = x
    return v / max(v.sum(), 1e-9)


# --------------------------------------------------------------------------
# Rocks of the crust (each cell has one). Mohs-like hardness, conchoidal
# fracture (can be knapped to an edge), spark when struck.
# --------------------------------------------------------------------------
ROCKS = [
    dict(name="granite",   comp=comp(Si=0.72, Al=0.15, Fe=0.04, Ca=0.03, Na=0.06), hardness=6.0, knappable=0.5),
    dict(name="flint",     comp=comp(Si=0.97, Fe=0.02, Ca=0.01),                     hardness=7.0, knappable=1.0),
    dict(name="limestone", comp=comp(Ca=0.92, Si=0.05, Fe=0.03),                     hardness=3.0, knappable=0.2),
    dict(name="basalt",    comp=comp(Si=0.50, Fe=0.15, Ca=0.15, Al=0.15, Na=0.05),   hardness=6.0, knappable=0.6),
]
N_ROCKS = len(ROCKS)
GRANITE, FLINT, LIMESTONE, BASALT = range(N_ROCKS)
ROCK_HARDNESS = np.array([r["hardness"] for r in ROCKS], np.float32)
ROCK_KNAP = np.array([r["knappable"] for r in ROCKS], np.float32)


def spark_chance(rock: int) -> float:
    """Striking stone on stone throws sparks only for hard siliceous or iron-bearing rock."""
    c = ROCKS[rock]["comp"]
    return float(np.clip((ROCK_HARDNESS[rock] - 5.5) * 0.4, 0, 1) * (0.5 + c[E["Si"]] * 0.3 + c[E["Fe"]] * 3))


# --------------------------------------------------------------------------
# Fuel and fire
# --------------------------------------------------------------------------
WOOD = comp(C=0.50, H2O=0.45, Ca=0.03, N=0.02)        # fresh wood, ~45 % water
CHARCOAL = comp(C=0.95, H2O=0.02, Ca=0.03)
AIR_DRY = 0.12            # wood cannot dry below ~12 % moisture in open air
WOOD_ENERGY = 0.10        # energy units of biomass locked in one unit of wood
CHARCOAL_ENERGY = 0.08    # charcoal keeps ~80 % of the wood's energy in half the mass

OPEN, KILN, FURNACE = 0, 1, 2
SETUP_NAMES = ["open fire", "kiln", "furnace"]
_ENCLOSURE = [1.0, 1.15, 1.15]
_DRAUGHT = [1.0, 1.0, 1.35]


def fire_temperature(fuel: str, wood_moisture: float, setup: int) -> float:
    """Peak temperature (deg C) of a fire. fuel: 'wood' | 'charcoal' | 'none'."""
    if fuel == "none":
        return 0.0
    if fuel == "charcoal":
        c, h2o = CHARCOAL[E["C"]], CHARCOAL[E["H2O"]]
    else:
        h2o = float(np.clip(wood_moisture, AIR_DRY, 0.6))
        c = WOOD[E["C"]] * (1 - h2o) / (1 - WOOD[E["H2O"]])  # drying concentrates carbon
        c = min(c, 0.85)
    # wet fuel wastes heat boiling water (latent heat of vaporisation)
    return 250.0 + 900.0 * c * (1 - 1.5 * h2o) * _ENCLOSURE[setup] * _DRAUGHT[setup]


# --------------------------------------------------------------------------
# Threshold transformations (the laws; inhabitants only see the outcomes)
# --------------------------------------------------------------------------
T_COOK = 70.0
T_CHARCOAL = 300.0
T_CERAMIC = 600.0
T_COPPER = 1085.0
T_IRON = 1250.0
T_GLASS_FLUXED = 1000.0
T_GLASS_PURE = 1700.0
T_TINDER = 250.0          # dry tinder catches a spark


def glass_clarity(sand_purity: float) -> float:
    """Iron impurities tint glass green/brown; lenses need nearly pure silica."""
    return float(np.clip((sand_purity - 0.80) / 0.18, 0, 1))


# material qualities when worked into tools: (effectiveness, wear per use)
TOOL_STONE_WEAR = 0.004
METALS = {"copper": dict(effect=1.3, wear=0.0012, hardness=3.0),
          "iron": dict(effect=1.8, wear=0.0005, hardness=4.5)}


def stone_tool_effect(rock: int) -> float:
    return float(ROCK_HARDNESS[rock] / 7.0 * (0.5 + 0.5 * ROCK_KNAP[rock]))


# --------------------------------------------------------------------------
# Atmosphere: carbon is bookkept through energy. Photosynthesis fixes CO2
# (solar_in), respiration / decay / burning return it (heat_out). Greenhouse
# warming follows the logarithmic forcing law (~3 deg C per doubling).
# --------------------------------------------------------------------------
ATM_CARBON = 20000.0      # atmospheric carbon, in energy-equivalent units
CLIMATE_SENSITIVITY = 3.0


def co2_ratio(solar_in: float, heat_out: float) -> float:
    return max(0.3, 1.0 + (heat_out - solar_in) / ATM_CARBON)


def greenhouse_offset(ratio: float) -> float:
    return CLIMATE_SENSITIVITY * math.log2(ratio)
