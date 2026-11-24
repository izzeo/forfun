"""Persistent history in SQLite: every organism ever born, significant events,
statistics and snapshots. The world's past is kept independent of whether
anyone was watching."""
from __future__ import annotations

import json
import sqlite3

import numpy as np

from ..biology import genome as GN
from ..physics import laws as L

SCHEMA = """
CREATE TABLE IF NOT EXISTS organisms (
    id INTEGER PRIMARY KEY, name TEXT, birth_tick INTEGER, death_tick INTEGER,
    parent_a INTEGER, parent_b INTEGER, generation INTEGER, species INTEGER,
    cause TEXT, killer INTEGER, x REAL, y REAL, traits TEXT, final TEXT);
CREATE INDEX IF NOT EXISTS ix_org_pa ON organisms(parent_a);
CREATE INDEX IF NOT EXISTS ix_org_pb ON organisms(parent_b);
CREATE TABLE IF NOT EXISTS events (
    tick INTEGER, kind TEXT, agent INTEGER, other INTEGER, x REAL, y REAL, data TEXT);
CREATE INDEX IF NOT EXISTS ix_ev_agent ON events(agent);
CREATE INDEX IF NOT EXISTS ix_ev_tick ON events(tick);
CREATE TABLE IF NOT EXISTS stats (tick INTEGER PRIMARY KEY, data TEXT);
"""


class History:
    def __init__(self, path: str | None):
        self.path = path or ":memory:"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._births: list[tuple] = []
        self._deaths: list[tuple] = []
        self._events: list[tuple] = []
        self.enabled = True

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, d):
        self.__init__(d["path"])

    # ---------------------------------------------------------------- writes
    def births(self, sim, idx) -> None:
        if not self.enabled:
            return
        pop = sim.pop
        for i in np.asarray(idx):
            aid = int(pop.id[i])
            traits = {t: round(float(pop.genome[i, k]), 3) for k, t in enumerate(GN.TRAITS)}
            self._births.append((aid, pop.names.get(aid, ""), int(pop.birth_tick[i]), None, int(pop.parent_a[i]),
                                 int(pop.parent_b[i]), int(pop.generation[i]), int(pop.species[i]), None, None,
                                 float(pop.x[i]), float(pop.y[i]), json.dumps(traits), None))
            if int(pop.parent_a[i]) > 0:
                self._events.append((int(pop.birth_tick[i]), "birth", aid, int(pop.parent_a[i]), float(pop.x[i]),
                                     float(pop.y[i]), json.dumps(dict(other_parent=int(pop.parent_b[i])))))

    def deaths(self, sim, idx, cause, killers: dict) -> None:
        if not self.enabled:
            return
        pop, tick = sim.pop, sim.clock.tick
        for i in idx:
            aid = int(pop.id[i])
            final = dict(age=int(pop.age[i]), children=int(pop.children[i]), kills=int(pop.kills[i]),
                         skills=[L.SKILLS[s] for s in np.nonzero(pop.skills[i] > 0.1)[0]],
                         group=int(pop.group[i]), memories=pop.ltm.get(aid, [])[-30:])
            self._deaths.append((tick, L.DEATH_CAUSES[int(cause[i])], killers.get(int(i), None), int(pop.species[i]),
                                 json.dumps(final), aid))
            self._events.append((tick, "death", aid, killers.get(int(i), -1) or -1, float(pop.x[i]), float(pop.y[i]),
                                 json.dumps(dict(cause=L.DEATH_CAUSES[int(cause[i])], age=int(pop.age[i])))))

    def event(self, sim, kind: str, agent: int, other: int, x, y, data: dict | None = None) -> None:
        if not self.enabled:
            return
        self._events.append((sim.clock.tick, kind, int(agent), int(other), float(x), float(y), json.dumps(data or {})))

    def stats(self, tick: int, data: dict) -> None:
        if not self.enabled:
            return
        self.db.execute("INSERT OR REPLACE INTO stats VALUES (?, ?)", (tick, json.dumps(data)))

    def flush(self) -> None:
        if self._births:
            self.db.executemany("INSERT OR REPLACE INTO organisms VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", self._births)
            self._births.clear()
        if self._deaths:
            self.db.executemany("UPDATE organisms SET death_tick=?, cause=?, killer=?, species=?, final=? WHERE id=?",
                                self._deaths)
            self._deaths.clear()
        if self._events:
            self.db.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?)", self._events)
            self._events.clear()
        self.db.commit()

    def truncate_after(self, tick: int) -> None:
        """Rewinding creates a new branch of history: forget the abandoned future."""
        self.flush()
        self.db.execute("DELETE FROM events WHERE tick > ?", (tick,))
        self.db.execute("DELETE FROM stats WHERE tick > ?", (tick,))
        self.db.execute("DELETE FROM organisms WHERE birth_tick > ?", (tick,))
        self.db.execute("UPDATE organisms SET death_tick=NULL, cause=NULL, killer=NULL, final=NULL WHERE death_tick > ?", (tick,))
        self.db.commit()

    # ---------------------------------------------------------------- reads
    def organism(self, aid: int) -> dict | None:
        self.flush()
        r = self.db.execute("SELECT * FROM organisms WHERE id=?", (aid,)).fetchone()
        if not r:
            return None
        cols = [c[0] for c in self.db.execute("SELECT * FROM organisms LIMIT 0").description]
        d = dict(zip(cols, r))
        d["traits"] = json.loads(d["traits"]) if d["traits"] else {}
        d["final"] = json.loads(d["final"]) if d["final"] else None
        return d

    def biography(self, aid: int, limit: int = 300) -> list[dict]:
        self.flush()
        rows = self.db.execute("SELECT tick, kind, agent, other, x, y, data FROM events WHERE agent=? OR other=? "
                               "ORDER BY tick LIMIT ?", (aid, aid, limit)).fetchall()
        return [dict(tick=t, kind=k, agent=a, other=o, x=x, y=y, data=json.loads(d)) for t, k, a, o, x, y, d in rows]

    def children_of(self, aid: int) -> list[int]:
        self.flush()
        return [r[0] for r in self.db.execute("SELECT id FROM organisms WHERE parent_a=? OR parent_b=?", (aid, aid))]

    def lineage(self, aid: int, up: int = 4, down: int = 2) -> dict:
        def node(a):
            o = self.organism(a)
            if not o:
                return dict(id=a, unknown=True)
            return dict(id=a, name=o["name"], born=o["birth_tick"], died=o["death_tick"], cause=o["cause"],
                        generation=o["generation"], species=o["species"])

        def ancestors(a, depth):
            n = node(a)
            if depth > 0 and not n.get("unknown"):
                o = self.organism(a)
                n["parents"] = [ancestors(p, depth - 1) for p in (o["parent_a"], o["parent_b"]) if p and p > 0]
            return n

        def descendants(a, depth):
            n = node(a)
            if depth > 0:
                kids = self.children_of(a)
                n["children"] = [descendants(c, depth - 1) for c in kids[:20]]
                n["n_children"] = len(kids)
            return n

        return dict(ancestors=ancestors(aid, up), descendants=descendants(aid, down))

    def events(self, kinds: list[str] | None = None, since: int = 0, limit: int = 200) -> list[dict]:
        self.flush()
        if kinds:
            q = f"SELECT tick, kind, agent, other, x, y, data FROM events WHERE tick>=? AND kind IN ({','.join('?' * len(kinds))}) ORDER BY tick DESC LIMIT ?"
            rows = self.db.execute(q, (since, *kinds, limit)).fetchall()
        else:
            rows = self.db.execute("SELECT tick, kind, agent, other, x, y, data FROM events WHERE tick>=? AND kind!='death' "
                                   "ORDER BY tick DESC LIMIT ?", (since, limit)).fetchall()
        return [dict(tick=t, kind=k, agent=a, other=o, x=x, y=y, data=json.loads(d)) for t, k, a, o, x, y, d in rows]

    def death_causes(self) -> dict:
        self.flush()
        return {c: n for c, n in self.db.execute("SELECT cause, COUNT(*) FROM organisms WHERE cause IS NOT NULL GROUP BY cause")}

    def counts(self) -> dict:
        self.flush()
        born = self.db.execute("SELECT COUNT(*) FROM organisms").fetchone()[0]
        dead = self.db.execute("SELECT COUNT(*) FROM organisms WHERE death_tick IS NOT NULL").fetchone()[0]
        no_cause = self.db.execute("SELECT COUNT(*) FROM organisms WHERE death_tick IS NOT NULL AND cause IS NULL").fetchone()[0]
        return dict(ever_lived=born, dead=dead, dead_without_cause=no_cause)
