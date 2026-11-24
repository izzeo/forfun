// GENESIS pixel view: a top-down pixel-art renderer of the living world.
// Every tile and every creature sprite is generated procedurally (no image files):
// creatures look the way their genes make them (size, colour, fins, big heads).

const T = 16; // pixels per world cell at zoom 1

// ---------------------------------------------------------------- helpers
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const canvas = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
const rgb = (c, k = 1) => `rgb(${Math.max(0, Math.min(255, c[0] * k)) | 0},${Math.max(0, Math.min(255, c[1] * k)) | 0},${Math.max(0, Math.min(255, c[2] * k)) | 0})`;
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => { const k = (n + h * 12) % 12; return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [f(0) * 255, f(8) * 255, f(4) * 255];
}
function speckle(ctx, r, base, n, spread) {
  for (let k = 0; k < n; k++) {
    const x = (r() * T) | 0, y = (r() * T) | 0;
    ctx.fillStyle = rgb(base, 1 + (r() - 0.5) * spread);
    ctx.fillRect(x, y, 1, 1);
  }
}

// ---------------------------------------------------------------- tiles (cached)
const SEASON_GRASS = { spring: [92, 164, 70], summer: [100, 150, 58], autumn: [156, 148, 66], winter: [128, 122, 92] };
const DIRT = [122, 96, 64];
const cache = new Map();
function tile(key, draw) {
  let c = cache.get(key);
  if (!c) { c = canvas(T, T); draw(c.getContext('2d')); cache.set(key, c); }
  return c;
}

function groundTile(season, density, v) {
  return tile(`g:${season}:${density}:${v}`, (ctx) => {
    const r = rng(1000 + density * 31 + v * 7);
    const base = mix(DIRT, SEASON_GRASS[season], Math.min(1, density / 3));
    ctx.fillStyle = rgb(base); ctx.fillRect(0, 0, T, T);
    speckle(ctx, r, base, 26, 0.35);
    if (density >= 2) {
      for (let k = 0; k < 6 + density * 3; k++) {
        const x = (r() * T) | 0, y = 2 + ((r() * (T - 3)) | 0);
        ctx.fillStyle = rgb(base, 1.25); ctx.fillRect(x, y, 1, 2);
      }
    }
  });
}
function flatTile(name, v) {
  return tile(`f:${name}:${v}`, (ctx) => {
    const r = rng(5000 + v * 13 + name.length * 101);
    const P = { sand: [214, 196, 142], rock: [126, 122, 116], peak: [236, 240, 246], deep: [24, 56, 108],
      shallow: [44, 98, 158], river: [58, 122, 192], farm: [112, 82, 50] }[name];
    ctx.fillStyle = rgb(P); ctx.fillRect(0, 0, T, T);
    if (name === 'sand') speckle(ctx, r, P, 30, 0.2);
    if (name === 'rock' || name === 'peak') {
      speckle(ctx, r, P, 30, 0.3);
      ctx.fillStyle = rgb(P, 0.7);
      for (let k = 0; k < 3; k++) { const x = (r() * 12) | 0, y = (r() * 12) | 0; ctx.fillRect(x, y, 3, 1); ctx.fillRect(x + 2, y + 1, 1, 2); }
    }
    if (name === 'deep' || name === 'shallow' || name === 'river') {
      speckle(ctx, r, P, 14, 0.25);
      ctx.fillStyle = rgb(P, 1.35);
      for (let k = 0; k < 2; k++) { const x = (r() * 11) | 0, y = (r() * 14) | 0; ctx.fillRect(x, y, 4, 1); }
    }
    if (name === 'farm') {
      for (let y = 1; y < T; y += 4) { ctx.fillStyle = rgb(P, 0.7); ctx.fillRect(0, y, T, 1); }
      speckle(ctx, r, P, 20, 0.3);
    }
  });
}
// plant overlays, drawn over ground
function plantTile(kind, season, amount) {
  return tile(`p:${kind}:${season}:${amount}`, (ctx) => {
    const r = rng(9000 + kind.length * 17 + amount);
    if (kind === 'tree') {
      const canopy = season === 'autumn' ? [196, 110, 40] : season === 'winter' ? null : [34, 96, 40];
      ctx.fillStyle = '#5b3a1e'; ctx.fillRect(7, 9, 2, 6);
      if (!canopy) { ctx.fillStyle = '#6b4a2e'; ctx.fillRect(5, 6, 1, 4); ctx.fillRect(10, 5, 1, 5); ctx.fillRect(6, 4, 4, 1); return; }
      const rad = 4 + amount;
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
        const d = ((x - 8) ** 2) / (rad * rad) + ((y - 6) ** 2) / ((rad - 1) ** 2);
        if (d <= 1) { ctx.fillStyle = rgb(canopy, d < 0.35 && x < 8 && y < 6 ? 1.35 : (r() < 0.2 ? 0.8 : 1)); ctx.fillRect(x, y, 1, 1); }
      }
      ctx.fillStyle = rgb(canopy, 0.6); ctx.fillRect(8 - rad + 1, 6 + rad - 2, rad * 2 - 2, 1);
      return;
    }
    if (kind === 'berry' || kind === 'nightshade') {
      const leaf = season === 'winter' ? [84, 96, 70] : [44, 118, 52];
      const fruit = kind === 'berry' ? [220, 40, 50] : [150, 40, 150];
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
        if (((x - 8) ** 2) / 30 + ((y - 9) ** 2) / 18 <= 1) { ctx.fillStyle = rgb(leaf, r() < 0.3 ? 0.8 : 1); ctx.fillRect(x, y, 1, 1); }
      }
      if (season !== 'winter') for (let k = 0; k < 3 + amount * 2; k++) { ctx.fillStyle = rgb(fruit); ctx.fillRect(4 + ((r() * 8) | 0), 6 + ((r() * 6) | 0), 1, 1); }
      return;
    }
    if (kind === 'tuber') {
      for (let k = 0; k < 4 + amount * 2; k++) {
        const x = 2 + ((r() * 12) | 0), y = 4 + ((r() * 10) | 0);
        ctx.fillStyle = '#7da24a'; ctx.fillRect(x, y, 1, 2); ctx.fillRect(x - 1, y, 1, 1); ctx.fillRect(x + 1, y, 1, 1);
      }
      return;
    }
    if (kind === 'algae') {
      for (let k = 0; k < 4 + amount * 3; k++) { ctx.fillStyle = 'rgba(90,170,110,0.9)'; ctx.fillRect((r() * T) | 0, (r() * T) | 0, 2, 1); }
      return;
    }
    if (kind === 'crop') {
      for (let y = 2; y < T; y += 4) for (let x = 1; x < T; x += 3) {
        if (r() < 0.25 + amount * 0.2) { ctx.fillStyle = season === 'autumn' ? '#d6b840' : '#5fb04a'; ctx.fillRect(x, y - 1, 1, 2); }
      }
    }
  });
}
function snowTile(heavy, v) {
  return tile(`s:${heavy}:${v}`, (ctx) => {
    const r = rng(7000 + v + heavy * 50);
    for (let k = 0; k < (heavy ? 140 : 45); k++) { ctx.fillStyle = `rgba(245,248,255,${0.6 + r() * 0.4})`; ctx.fillRect((r() * T) | 0, (r() * T) | 0, 1, 1); }
  });
}
function structTile(kind) {
  return tile(`st:${kind}`, (ctx) => {
    const px = (x, y, c, w = 1, h = 1) => { ctx.fillStyle = c; ctx.fillRect(x, y, w, h); };
    if (kind === 1) { // shelter: hut with thatched roof
      px(3, 8, '#7a5230', 10, 6); px(7, 10, '#2a1a0e', 2, 4);
      for (let k = 0; k < 6; k++) px(2 + k, 7 - k, '#c9a35a', 12 - 2 * k, 1);
      px(2, 7, '#9c7a3a', 12, 1);
    } else if (kind === 4) { // hearth: ring of stones (flame drawn animated)
      for (const [x, y] of [[5, 10], [7, 11], [9, 11], [11, 10], [4, 8], [12, 8], [6, 7], [10, 7]]) px(x, y, '#8d8a85', 2, 1);
      px(6, 9, '#3b2a1a', 5, 2);
    } else if (kind === 3) { // shrine
      px(6, 4, '#9b98a8', 4, 10); px(5, 13, '#6e6b78', 6, 2); px(6, 3, '#c19cff', 4, 1); px(7, 2, '#e3ccff', 2, 1);
    } else if (kind === 6) { // storage pit with wooden cover
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if (((x - 8) ** 2) / 36 + ((y - 9) ** 2) / 16 <= 1) px(x, y, '#3a2816');
      px(3, 8, '#8a5e30', 10, 1); px(3, 10, '#8a5e30', 10, 1);
    } else if (kind === 7 || kind === 8) { // kiln / furnace: clay dome
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if (((x - 8) ** 2) / 36 + ((y - 11) ** 2) / 49 <= 1 && y <= 14) px(x, y, (x + y) % 5 ? '#a8542e' : '#8a4424');
      px(6, 11, '#1e0f08', 4, 3);
      if (kind === 8) { px(11, 2, '#6b6158', 2, 6); px(3, 12, '#c9b49a', 2, 1); }
    } else if (kind === 5) {
      px(3, 6, '#6c7a88', 10, 8);
    }
  });
}

// ---------------------------------------------------------------- creature sprites
export function creatureSprite(h, size, aquatic, brain, carnivore, child, frame) {
  const key = `c:${(h * 24) | 0}:${(size * 6) | 0}:${aquatic}:${(brain * 3) | 0}:${carnivore}:${child}:${frame}`;
  return tile(key, (ctx) => {
    const s = (0.7 + size * 0.5) * (child ? 0.65 : 1);
    const body = hsl(((h * 24) | 0) / 24, 0.55, 0.5);
    const dark = mix(body, [0, 0, 0], 0.55);
    const m = Array.from({ length: T }, () => new Array(T).fill(null));
    const put = (x, y, c) => { if (x >= 0 && y >= 0 && x < T && y < T) m[y][x] = c; };
    if (aquatic) { // fish: body, tail, fin
      const rx = 5 * s + 1, ry = 2.6 * s + 0.5, cx = 9, cy = 9;
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1) put(x, y, body);
      const tx = Math.round(cx - rx);
      for (let k = 0; k < 3; k++) { put(tx - k, cy - 1 - k + frame, body); put(tx - k, cy + 1 + k - frame, body); put(tx - k, cy, body); }
      put(cx, Math.round(cy - ry), mix(body, [255, 255, 255], 0.3));
      put(Math.round(cx + rx - 2), cy - 1, [255, 255, 255]); put(Math.round(cx + rx - 1), cy - 1, [10, 10, 10]);
    } else if (brain < 0.5) { // quadruped, side view facing right
      const rx = 4.6 * s + 0.5, ry = 2.4 * s + 0.4, cx = 7, cy = 9.5 - frame * 0.5;
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if (((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1) put(x, y, body);
      for (let x = Math.round(cx - rx) + 1; x < cx + rx - 1; x++) put(x, Math.round(cy - ry * 0.4), mix(body, [255, 255, 255], 0.15));
      const hr = 1.5 * s + 0.4, hx = Math.round(cx + rx), hy = Math.round(cy - ry);
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if ((x - hx) ** 2 + (y - hy) ** 2 <= hr * hr + 0.5) put(x, y, body);
      put(hx - 1, hy - Math.round(hr) - 1, dark);                       // ear
      put(hx + 1, hy - 1, [10, 10, 10]);                                   // eye
      if (carnivore) { put(hx + Math.round(hr), hy + 1, [240, 240, 240]); put(hx + Math.round(hr) - 1, hy + 1, [200, 30, 30]); }
      const ly = Math.round(cy + ry) + 1;
      const legs = [cx - rx + 1.5, cx - rx + 3, cx + rx - 3, cx + rx - 1.5].map(Math.round);
      legs.forEach((lx, k) => { const off = frame && k % 2 ? 1 : 0; put(lx + off, ly - 1, dark); put(lx + off, ly, dark); });
      put(Math.round(cx - rx) - 1, Math.round(cy - 1), dark); put(Math.round(cx - rx) - 2, Math.round(cy - 2), dark); // tail
    } else { // upright, big-headed: the look of a thinking species
      const tw = 1.7 * s + 0.4, th = 3.0 * s + 0.6, cx = 8, cy = 10 - frame * 0.5;
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if (((x - cx) / tw) ** 2 + ((y - cy) / th) ** 2 <= 1) put(x, y, body);
      const hr = (1.4 + brain * 1.3) * Math.max(0.8, s), hx = cx, hy = Math.round(cy - th - hr + 0.5);
      for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if ((x - hx) ** 2 + (y - hy) ** 2 <= hr * hr) put(x, y, mix(body, [255, 220, 190], 0.25));
      put(hx + 1, hy, [10, 10, 10]); put(hx - 1, hy, [10, 10, 10]);      // eyes
      if (carnivore) put(hx, hy + Math.round(hr) - 1, [200, 30, 30]);
      const ay = Math.round(cy - th * 0.3);
      put(Math.round(cx - tw) - 1, ay + frame, dark); put(Math.round(cx - tw) - 1, ay + 1 + frame, dark);   // arms
      put(Math.round(cx + tw) + 1, ay + 1 - frame, dark); put(Math.round(cx + tw) + 1, ay + 2 - frame, dark);
      const ly = Math.round(cy + th);
      put(cx - 1 - (frame ? 1 : 0), ly, dark); put(cx - 1 - (frame ? 1 : 0), ly + 1, dark);              // legs
      put(cx + 1 + (frame ? 1 : 0), ly, dark); put(cx + 1 + (frame ? 1 : 0), ly + 1, dark);
    }
    // outline pass
    for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
      if (m[y][x]) continue;
      const n = [[1, 0], [-1, 0], [0, 1], [0, -1]].some(([dx, dy]) => m[y + dy] && m[y + dy][x + dx] && m[y + dy][x + dx] !== dark);
      if (n) { ctx.fillStyle = 'rgba(15,12,20,0.85)'; ctx.fillRect(x, y, 1, 1); }
    }
    for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) if (m[y][x]) { ctx.fillStyle = rgb(m[y][x]); ctx.fillRect(x, y, 1, 1); }
  });
}

// action → small icon above the head
const ICON = {
  eat: ['#6fdc6f', 'dot'], seek_food: null, rest: ['#cfd8ff', 'z'], flee: ['#ffffff', '!'], attack: ['#ff4d4d', 'x'],
  social: ['#ff8fc8', 'heart'], mate: ['#ff3b6b', 'heart'], experiment: ['#ffe066', '?'], work: ['#c9c9c9', 'hammer'],
  drink: ['#7cc4ff', 'drop'], recall: null, wander: null,
};
function drawIcon(ctx, kind, x, y, z) {
  const spec = ICON[kind];
  if (!spec) return;
  const [c, shape] = spec;
  const p = (dx, dy) => ctx.fillRect(x + dx * z, y + dy * z, z, z);
  ctx.fillStyle = c;
  if (shape === 'dot') { p(0, 0); p(1, 0); p(0, 1); p(1, 1); }
  else if (shape === 'z') { p(0, 0); p(1, 0); p(2, 0); p(1, 1); p(0, 2); p(1, 2); p(2, 2); }
  else if (shape === '!') { p(1, -1); p(1, 0); p(1, 2); }
  else if (shape === 'x') { p(0, 0); p(2, 0); p(1, 1); p(0, 2); p(2, 2); }
  else if (shape === 'heart') { p(0, 0); p(2, 0); p(0, 1); p(1, 1); p(2, 1); p(1, 2); }
  else if (shape === '?') { p(0, -1); p(1, -1); p(2, 0); p(1, 1); p(1, 3); }
  else if (shape === 'hammer') { p(0, 0); p(1, 0); p(2, 0); p(1, 1); p(1, 2); }
  else if (shape === 'drop') { p(1, 0); p(0, 1); p(1, 1); p(2, 1); p(1, 2); }
}

// ---------------------------------------------------------------- the view
export class PixelView {
  constructor(el, opts) {
    this.el = el;
    this.ctx = el.getContext('2d');
    this.opts = opts;
    this.world = null;
    this.fields = {};
    this.agents = new Map();
    this.zoom = 1;
    this.camX = 0; this.camY = 0;
    this.selected = null;
    this.follow = false;
    this.sun = 1;
    this.season = 'spring';
    this.overlay = null;
    this.night = new OffscreenCanvasLike();
    this._bindInput();
  }

  setWorld(world) {
    this.world = world;
    this.W = world.width; this.H = world.height;
    this.terrain = canvas(this.W * T, this.H * T);
    this.fields = { ptype: world.ptype };
    this.agents.clear();
    this.fit();
    this.rebuild();
  }

  fit() {
    const r = this.el.getBoundingClientRect();
    this.zoom = Math.max(0.25, Math.min(r.width / (this.W * T), r.height / (this.H * T)));
    this.camX = (this.W * T) / 2; this.camY = (this.H * T) / 2;
  }

  setFields(f, clock) {
    Object.assign(this.fields, f);
    if (clock) this.season = clock.season;
    this.rebuild();
  }

  setOverlay(fn) { this.overlay = fn; this.rebuild(); }

  setFrame(frame) {
    const now = performance.now();
    const seen = new Set();
    for (const a of frame.agents) {
      const id = a[0];
      seen.add(id);
      let o = this.agents.get(id);
      if (!o) { o = { x: a[1], y: a[2], px: a[1], py: a[2], t: now, face: 1 }; this.agents.set(id, o); }
      const cur = this._pos(o, now);
      o.px = cur[0]; o.py = cur[1];
      if (Math.abs(a[1] - cur[0]) > 0.01) o.face = a[1] > cur[0] ? 1 : -1;
      o.x = a[1]; o.y = a[2]; o.t = now; o.row = a;
    }
    for (const id of [...this.agents.keys()]) if (!seen.has(id)) this.agents.delete(id);
    if (typeof frame.sun === 'number') this.sun = frame.sun;
    else if (frame.clock) this.sun = -Math.cos(2 * Math.PI * (frame.clock.hour + (frame.minute || 0) / 60) / 24);
    if (frame.clock) this.season = frame.clock.season;
  }

  _pos(o, now) {
    const k = Math.min(1, (now - o.t) / 260);
    return [o.px + (o.x - o.px) * k, o.py + (o.y - o.py) * k];
  }

  // ---- static layer: terrain, plants, structures, snow, overlays ----
  rebuild() {
    if (!this.world) return;
    const w = this.world, f = this.fields, ctx = this.terrain.getContext('2d');
    const sea = w.sea_level;
    const plants = f.plants, ptype = f.ptype || w.ptype, struct = f.struct, temp = f.temp;
    const PL = ['grass', 'berry', 'nightshade', 'tree', 'algae', 'tuber'];
    for (let i = 0; i < this.W * this.H; i++) {
      const x = (i % this.W) * T, y = ((i / this.W) | 0) * T;
      const e = w.elevation[i], h = (e - sea) / (1 - sea);
      const v = (i * 2654435761 >>> 0) % 4;
      const bio = plants ? plants[i] / 200 : 0.5;
      const isRiver = w.river && w.river[i];
      const st = struct ? struct[i] : 0;
      let base;
      if (isRiver && e >= sea) base = flatTile(w.water[i] ? 'shallow' : 'river', v);
      else if (w.water[i]) base = flatTile(e < sea - 0.08 ? 'deep' : 'shallow', v);
      else if (h > 0.84) base = flatTile('peak', v);
      else if (h > 0.68) base = flatTile('rock', v);
      else if (st === 2) base = flatTile('farm', v);
      else if (w.sand && w.sand[i] && bio < 0.5) base = flatTile('sand', v);
      else base = groundTile(this.season, Math.max(0, Math.min(3, Math.round(bio * 3.2))), v);
      ctx.drawImage(base, x, y);
      const pt = PL[ptype[i]];
      const amount = Math.max(0, Math.min(2, Math.round(bio * 2)));
      if (st === 2) ctx.drawImage(plantTile('crop', this.season, amount), x, y);
      else if (w.water[i] || isRiver) { if (pt === 'algae' && bio > 0.2) ctx.drawImage(plantTile('algae', this.season, amount), x, y); }
      else if (h <= 0.68 && pt !== 'grass' && pt !== 'algae' && bio > 0.25) ctx.drawImage(plantTile(pt, this.season, amount), x, y);
      if (temp && !w.water[i]) {
        const tc = temp[i] / 4 - 30;
        if (tc < -1) ctx.drawImage(snowTile(tc < -6 ? 1 : 0, v), x, y);
      }
      if (st && st !== 2) ctx.drawImage(structTile(st), x, y);
      if (this.overlay) {
        const c = this.overlay(i);
        if (c) { ctx.fillStyle = `rgba(${c[0] * 255 | 0},${c[1] * 255 | 0},${c[2] * 255 | 0},0.62)`; ctx.fillRect(x, y, T, T); }
      }
    }
  }

  // ---- per-frame drawing ----
  resize() {
    const r = this.el.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.el.width = Math.max(1, r.width * dpr) | 0;
    this.el.height = Math.max(1, r.height * dpr) | 0;
    this.dpr = dpr;
  }

  toScreen(wx, wy) { // world cell coords -> canvas px
    const z = this.zoom * this.dpr;
    return [(wx * T - this.camX) * z + this.el.width / 2, (wy * T - this.camY) * z + this.el.height / 2];
  }
  toWorld(sx, sy) { // css px relative to canvas -> world cell coords
    const z = this.zoom;
    const r = this.el.getBoundingClientRect();
    return [((sx - r.width / 2) / z + this.camX) / T, ((sy - r.height / 2) / z + this.camY) / T];
  }

  draw() {
    if (!this.world || !this.terrain) return;
    const ctx = this.ctx, now = performance.now();
    const z = this.zoom * this.dpr;
    if (this.follow && this.selected != null && this.agents.has(this.selected)) {
      const p = this._pos(this.agents.get(this.selected), now);
      this.camX = p[0] * T; this.camY = p[1] * T;
    }
    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = '#0b1220'; ctx.fillRect(0, 0, this.el.width, this.el.height);
    const [ox, oy] = this.toScreen(0, 0);
    ctx.drawImage(this.terrain, ox, oy, this.W * T * z, this.H * T * z);

    // visible cell range
    const c0 = Math.max(0, Math.floor((-ox) / (T * z))), r0 = Math.max(0, Math.floor((-oy) / (T * z)));
    const c1 = Math.min(this.W - 1, Math.ceil((this.el.width - ox) / (T * z))), r1 = Math.min(this.H - 1, Math.ceil((this.el.height - oy) / (T * z)));
    const f = this.fields, tick = (now / 180) | 0;
    const lights = [];
    for (let cy = r0; cy <= r1; cy++) for (let cx = c0; cx <= c1; cx++) {
      const i = cy * this.W + cx;
      const [sx, sy] = this.toScreen(cx, cy);
      const st = f.struct ? f.struct[i] : 0;
      const fire = f.fire ? f.fire[i] : 0;
      if (st === 4 || st === 8 || fire > 25) {
        const fx = st === 8 ? 7 : 7, fy = st === 8 ? 10 : (st === 4 ? 6 : 5);
        const fl = ((tick + i) % 3);
        ctx.fillStyle = '#ffd24a'; ctx.fillRect(sx + (fx + 1) * z, sy + (fy + 1 - fl) * z, 2 * z, (3 + fl) * z);
        ctx.fillStyle = '#ff7a1a'; ctx.fillRect(sx + fx * z, sy + (fy + 2) * z, 4 * z, 2 * z);
        if (fire > 25 && !st) { ctx.fillStyle = 'rgba(255,120,30,0.35)'; ctx.fillRect(sx, sy, T * z, T * z); }
        lights.push([sx + 8 * z, sy + 8 * z, (st === 8 ? 5 : 3.5) * T * z]);
      }
      if (this.world.water[i] && ((i * 7 + tick) % 23 === 0)) {
        ctx.fillStyle = 'rgba(200,230,255,0.55)'; ctx.fillRect(sx + ((i * 5) % 12) * z, sy + ((i * 3) % 14) * z, 3 * z, z);
      }
      const cl = f.clouds ? f.clouds[i] : 0;
      if (cl > 120) {
        ctx.fillStyle = `rgba(20,25,40,${Math.min(0.35, (cl - 120) / 300)})`; ctx.fillRect(sx, sy, T * z, T * z);
        if (cl > 165) {
          ctx.fillStyle = 'rgba(170,200,255,0.7)';
          for (let k = 0; k < 3; k++) {
            const rx = ((i * 13 + k * 29 + tick * 7) % 15), ry = ((i * 7 + k * 41 + tick * 11) % 13);
            ctx.fillRect(sx + rx * z, sy + ry * z, Math.max(1, z * 0.6), 3 * z);
          }
        }
      }
    }

    // creatures, back to front
    const list = [];
    for (const [id, o] of this.agents) { const p = this._pos(o, now); list.push([p[1], p[0], id, o]); }
    list.sort((a, b) => a[0] - b[0]);
    const actions = this.opts.actions || [];
    for (const [py, px, id, o] of list) {
      const a = o.row;
      const [sx, sy] = this.toScreen(px, py);
      if (sx < -T * z || sy < -T * z || sx > this.el.width + T * z || sy > this.el.height + T * z) continue;
      const moving = Math.hypot(o.x - o.px, o.y - o.py) > 0.05;
      const frame = moving ? ((tick + id) % 2) : 0;
      const spr = creatureSprite(a[7], a[6], (a[10] ?? 0) > 0.5 ? 1 : 0, a[8], (a[11] ?? 0) > 0.5 ? 1 : 0, a[9] <= 1 ? 1 : 0, frame);
      ctx.save();
      ctx.translate(sx, sy - 8 * z);
      if (o.face < 0) ctx.scale(-1, 1);
      ctx.drawImage(spr, -8 * z, -8 * z, T * z, T * z);
      ctx.restore();
      if (this.groupColors && a[4] >= 0) {
        ctx.fillStyle = `hsl(${(a[4] * 137.5) % 360},80%,60%)`; ctx.fillRect(sx - 2 * z, sy + 1 * z, 4 * z, Math.max(1, z));
      }
      if (z >= 0.9) drawIcon(ctx, actions[a[5]], sx + 2 * z, sy - 18 * z, Math.max(1, Math.round(z * 0.75)));
      if (id === this.selected) {
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = Math.max(1, z * 0.6);
        ctx.strokeRect(sx - 8 * z, sy - 16 * z, T * z, T * z);
        ctx.fillStyle = '#ffffff'; ctx.font = `${Math.max(10, 6 * z)}px system-ui`;
        ctx.fillText(this.selectedName || `#${id}`, sx - 8 * z, sy - 18 * z);
      }
    }

    // night: darkness with pools of firelight
    const dark = Math.max(0, Math.min(0.72, -this.sun * 0.9 + 0.15));
    if (dark > 0.02) {
      const n = this.night.get(this.el.width, this.el.height);
      const nctx = n.getContext('2d');
      nctx.globalCompositeOperation = 'source-over';
      nctx.clearRect(0, 0, n.width, n.height);
      nctx.fillStyle = `rgba(8,12,40,${dark})`; nctx.fillRect(0, 0, n.width, n.height);
      nctx.globalCompositeOperation = 'destination-out';
      for (const [lx, ly, lr] of lights) {
        const g = nctx.createRadialGradient(lx, ly, 0, lx, ly, lr);
        g.addColorStop(0, 'rgba(0,0,0,1)'); g.addColorStop(1, 'rgba(0,0,0,0)');
        nctx.fillStyle = g; nctx.fillRect(lx - lr, ly - lr, lr * 2, lr * 2);
      }
      ctx.drawImage(n, 0, 0);
      ctx.globalCompositeOperation = 'lighter';
      for (const [lx, ly, lr] of lights) {
        const g = ctx.createRadialGradient(lx, ly, 0, lx, ly, lr * 0.7);
        g.addColorStop(0, `rgba(255,140,40,${0.25 * dark})`); g.addColorStop(1, 'rgba(255,140,40,0)');
        ctx.fillStyle = g; ctx.fillRect(lx - lr, ly - lr, lr * 2, lr * 2);
      }
      ctx.globalCompositeOperation = 'source-over';
    }
  }

  pick(sx, sy) {
    const [wx, wy] = this.toWorld(sx, sy);
    const now = performance.now();
    let best = null, bd = 0.9;
    for (const [id, o] of this.agents) {
      const p = this._pos(o, now);
      const d = Math.hypot(p[0] - wx, p[1] - 0.5 - wy);
      if (d < bd) { bd = d; best = id; }
    }
    return { id: best, cell: [Math.floor(wx), Math.floor(wy)] };
  }

  _bindInput() {
    let drag = null;
    this.el.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, cx: this.camX, cy: this.camY, moved: false }; this.el.setPointerCapture(e.pointerId); });
    this.el.addEventListener('pointermove', (e) => {
      const r = this.el.getBoundingClientRect();
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.hypot(dx, dy) > 4) { drag.moved = true; this.follow = false; if (this.opts.onFollowChange) this.opts.onFollowChange(false); }
        this.camX = drag.cx - dx / this.zoom; this.camY = drag.cy - dy / this.zoom;
      } else if (this.opts.onHover) {
        const p = this.pick(e.clientX - r.left, e.clientY - r.top);
        const o = p.id != null ? this.agents.get(p.id) : null;
        this.opts.onHover(o ? o.row : null, e.offsetX, e.offsetY);
      }
    });
    this.el.addEventListener('pointerup', (e) => {
      const r = this.el.getBoundingClientRect();
      if (drag && !drag.moved) {
        const p = this.pick(e.clientX - r.left, e.clientY - r.top);
        if (this.opts.isMicro && this.opts.isMicro()) this.opts.onMicro(p.cell[0], p.cell[1]);
        else if (p.id != null) this.opts.onSelect(p.id);
      }
      drag = null;
    });
    this.el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = this.el.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const before = this.toWorld(mx, my);
      this.zoom = Math.max(0.2, Math.min(6, this.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      const after = this.toWorld(mx, my);
      this.camX += (before[0] - after[0]) * T; this.camY += (before[1] - after[1]) * T;
    }, { passive: false });
  }
}

class OffscreenCanvasLike {
  get(w, h) {
    if (!this.c || this.c.width !== w || this.c.height !== h) this.c = canvas(w, h);
    return this.c;
  }
}
