"""Chemistry, heat, materials and conservation of matter."""
import numpy as np

from genesis import Simulation
from genesis.biology import genome as GN
from genesis.intelligence import skills
from genesis.intelligence.perception import perceive
from genesis.physics import chemistry as CH, laws as L
from tests.conftest import small_cfg

A, R, S = L.A, L.R, L.SK


def test_fire_temperature_follows_fuel_and_setup():
    wet_open = CH.fire_temperature("wood", 0.45, CH.OPEN)
    dry_open = CH.fire_temperature("wood", 0.15, CH.OPEN)
    char_kiln = CH.fire_temperature("charcoal", 0, CH.KILN)
    char_furnace = CH.fire_temperature("charcoal", 0, CH.FURNACE)
    wood_furnace = CH.fire_temperature("wood", 0.0, CH.FURNACE)   # cannot dry below air-dry
    assert wet_open < CH.T_CERAMIC <= dry_open, "pottery needs dry fuel"
    assert dry_open < CH.T_COPPER <= char_kiln, "copper needs charcoal in a kiln"
    assert char_kiln < CH.T_IRON <= char_furnace, "iron needs forced draught"
    assert wood_furnace < CH.T_IRON, "wood alone never smelts iron"
    assert char_furnace < CH.T_GLASS_PURE, "pure silica cannot be melted: glass needs a flux"


def test_material_properties_come_from_composition():
    assert CH.stone_tool_effect(CH.FLINT) > CH.stone_tool_effect(CH.GRANITE) > CH.stone_tool_effect(CH.LIMESTONE)
    assert CH.spark_chance(CH.FLINT) > 0.3 and CH.spark_chance(CH.LIMESTONE) == 0
    assert CH.glass_clarity(0.98) > 0.9 and CH.glass_clarity(0.80) == 0


def _crafter(seed=6):
    s = Simulation(small_cfg(seed=seed, scenario="advanced", initial_population=4))
    s.step()
    p, w = s.pop, s.world
    y, x = 20, 20
    w.water[y - 2:y + 3, x - 2:x + 3] = False
    w.struct[y - 2:y + 3, x - 2:x + 3] = 0
    p.x[0], p.y[0] = x + 0.5, y + 0.5
    p.skills[0, :] = 1.0
    p.energy[0] = 1.0
    return s, p, w, y, x


def _work(s, i=0):
    pc = perceive(s)
    act = np.full(len(s.pop), A["rest"]); act[i] = A["work"]
    return skills.work(s, pc, act, np.zeros(len(s.pop), np.float32))


def test_pottery_needs_a_hot_enough_fire():
    s, p, w, y, x = _crafter()
    w.struct[y, x + 1] = L.STRUCT_HEARTH; w.struct_hp[y, x + 1] = 1
    p.inventory[0] = 0
    p.inventory[0, R["clay"]] = 1
    p.inventory[0, R["wood"]] = 1
    p.wood_moist[0] = 0.45                      # green wood: fire too cool
    p.skills[0, :] = 0; p.skills[0, S["pottery"]] = 1
    _work(s)
    assert p.inventory[0, R["pottery"]] == 0
    p.wood_moist[0] = 0.15                      # seasoned wood
    _work(s)
    assert p.inventory[0, R["pottery"]] == 1 and p.inventory[0, R["clay"]] == 0


def test_iron_needs_charcoal_and_a_furnace_copper_does_not():
    s, p, w, y, x = _crafter()
    p.skills[0, :] = 0; p.skills[0, S["smelting"]] = 1
    w.struct[y, x + 1] = L.STRUCT_KILN; w.struct_hp[y, x + 1] = 3
    p.inventory[0] = 0
    p.inventory[0, R["ore"]] = 1; p.inventory[0, R["charcoal"]] = 2
    _work(s)
    assert p.inventory[0, R["metal"]] == 0, "kiln + charcoal is too cool for iron"
    p.inventory[0, R["copper_ore"]] = 1
    _work(s)
    assert p.inventory[0, R["metal"]] == 1 and p.metal_q[0] == CH.METALS["copper"]["effect"]
    p.inventory[0, R["metal"]] = 0
    w.struct[y, x + 1] = L.STRUCT_FURNACE
    _work(s)
    assert p.inventory[0, R["metal"]] == 1 and p.metal_q[0] == CH.METALS["iron"]["effect"]


def test_tools_wear_out_and_leave_debris():
    s, p, w, y, x = _crafter()
    p.inventory[0] = 0
    p.inventory[0, R["tool"]] = 1
    p.tool_q[0] = 1.0
    before = s.total_minerals()
    for _ in range(int(1 / CH.TOOL_STONE_WEAR) + 2):
        skills.wear_tools(s, [0])
    assert p.inventory[0, R["tool"]] == 0
    assert abs(s.total_minerals() - before) < 1e-4, "the broken tool's stone is still somewhere"


def test_matter_and_energy_are_conserved_in_a_crafting_world():
    s = Simulation(small_cfg(seed=2, scenario="advanced", initial_population=120))
    # give everybody every skill so that all crafts actually happen
    s.pop.skills[:] = 0.5
    m0 = s.total_minerals()
    e0 = s.total_energy()
    in0, out0 = s.world.ledger.solar_in, s.world.ledger.heat_out
    s.run(600)
    crafted = s.pop.inventory[:, [R["tool"], R["pottery"], R["charcoal"], R["metal"]]].sum() + (s.world.struct >= L.STRUCT_KILN).sum()
    assert crafted > 0, "crafting happened"
    assert abs(s.total_minerals() - m0) < 1e-3, (s.total_minerals(), m0)
    expected = e0 + (s.world.ledger.solar_in - in0) - (s.world.ledger.heat_out - out0)
    assert abs(s.total_energy() - expected) / e0 < 1e-3


def test_burning_raises_co2_and_warms_the_climate():
    s = Simulation(small_cfg(seed=2, initial_population=0))
    s.step()
    t0 = s.world.greenhouse
    s.world.ledger.heat_out += CH.ATM_CARBON  # as if a huge amount of biomass had burnt
    s.step()
    assert s.world.co2 > 1.9
    assert s.world.greenhouse - t0 > 2.5, "doubling CO2 warms by ~3 C"
