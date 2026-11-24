"""Command-line entry point.

    genesis serve   [--port 8000] [--seed 1] [--scenario primordial|advanced] [--framework none]
    genesis run     --ticks 5000 [--seed 1] [--scenario ...] [--save world.pkl]
    genesis compare --ticks 3000 --seeds 1 2 [--frameworks none quran_inspired egoist]
"""
from __future__ import annotations

import argparse
import json
import time

from .config import WorldConfig


def main(argv=None):
    ap = argparse.ArgumentParser(prog="genesis", description="GENESIS artificial evolving universe")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def world_args(p):
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--scenario", default="primordial", choices=["primordial", "advanced"])
        p.add_argument("--framework", default="none")
        p.add_argument("--population", type=int, default=300)
        p.add_argument("--size", type=int, default=96)
        p.add_argument("--db", default=None, help="SQLite file for history (default: in memory)")

    s = sub.add_parser("serve", help="run the simulation with the web observer")
    world_args(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--scale", default="day")

    r = sub.add_parser("run", help="run headless and print a report")
    world_args(r)
    r.add_argument("--ticks", type=int, default=2000)
    r.add_argument("--save", default=None)
    r.add_argument("--every", type=int, default=500)

    c = sub.add_parser("compare", help="compare moral frameworks on identical worlds")
    c.add_argument("--ticks", type=int, default=3000)
    c.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    c.add_argument("--frameworks", nargs="+", default=["none", "quran_inspired", "utilitarian", "egoist"])
    c.add_argument("--scenario", default="advanced")
    c.add_argument("--population", type=int, default=250)

    a = ap.parse_args(argv)

    def cfg_of(a):
        return WorldConfig(seed=a.seed, scenario=a.scenario, moral_framework=a.framework,
                           initial_population=a.population, width=a.size, height=a.size, db_path=a.db)

    if a.cmd == "serve":
        import uvicorn
        from .api.server import create_app
        from .engine.simulation import Simulation
        sim = Simulation(cfg_of(a))
        app = create_app(sim)
        app.state.initial_scale = a.scale
        print(f"GENESIS observer on http://{a.host}:{a.port}")
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    elif a.cmd == "run":
        from .engine.simulation import Simulation
        sim = Simulation(cfg_of(a))
        t0 = time.time()
        while sim.clock.tick < a.ticks:
            sim.run(min(a.every, a.ticks - sim.clock.tick))
            row = sim.analytics.series[-1]
            print(f"[{sim.clock.info()['label']}] pop={row['population']} species={row['species']} "
                  f"gen={row.get('max_generation')} groups={row['groups']} settlements={row['settlements']} "
                  f"brain={row.get('traits', {}).get('brain', 0):.3f} ({time.time() - t0:.0f}s)")
            if not len(sim.pop):
                print("Life went extinct.")
                break
        print("\nEmergent firsts:")
        for f in sorted(sim.analytics.firsts.values(), key=lambda f: f["tick"]):
            print(f"  t={f['tick']:>6}  {f['text']}")
        print("\nCauses of death:", json.dumps(sim.history.death_causes()))
        print("Layers of reality known to inhabitants:", sorted(sim.civ.layers))
        if a.save:
            sim.save(a.save)
            print("saved to", a.save)
    elif a.cmd == "compare":
        from .analytics.experiments import compare_frameworks
        res = compare_frameworks(a.frameworks, a.seeds, a.ticks, a.scenario, a.population)
        keys = ["final_population", "mean_population", "violent_death_share", "starvation_share",
                "sharing_per_1000h", "trades", "rituals"]
        print("framework".ljust(16) + "".join(k[:18].rjust(20) for k in keys))
        for r in res:
            print(r["framework"].ljust(16) + "".join(str(r[k]).rjust(20) for k in keys))


if __name__ == "__main__":
    main()
