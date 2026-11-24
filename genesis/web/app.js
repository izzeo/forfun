import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const api = (p, opt) => fetch(p, opt).then((r) => r.json());
const post = (body) => api('/api/control', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

let W = 0, H = 0, world = null, fields = {}, agents = [], selected = null, layer = 'surface', meta = null;
let state = null, lastStats = [], knownLayers = {};
const HSCALE = 9;

// ---------------------------------------------------------------- three.js scene
const canvas = $('#scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
scene.add(sun);
const ambient = new THREE.AmbientLight(0x8899bb, 0.45);
scene.add(ambient);

let terrain, terrainColors, water, clouds, cloudTex, agentMesh, structMesh, ring;
const dummy = new THREE.Object3D();
const tmpColor = new THREE.Color();

function heightAt(x, y) {
  if (!world) return 0;
  const cx = Math.min(W - 1, Math.max(0, Math.floor(x))), cy = Math.min(H - 1, Math.max(0, Math.floor(y)));
  const e = world.elevation[cy * W + cx];
  return Math.max(e, world.sea_level) * HSCALE;
}

function buildWorld() {
  [terrain, water, clouds, agentMesh, structMesh, ring].forEach((o) => o && scene.remove(o));
  const geo = new THREE.PlaneGeometry(W, H, W - 1, H - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const e = world.elevation[i];
    pos.setY(i, Math.max(e, world.sea_level - 0.02) * HSCALE);
  }
  geo.computeVertexNormals();
  terrainColors = new Float32Array(pos.count * 3);
  geo.setAttribute('color', new THREE.BufferAttribute(terrainColors, 3));
  terrain = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }));
  scene.add(terrain);

  const wgeo = new THREE.PlaneGeometry(W, H);
  wgeo.rotateX(-Math.PI / 2);
  water = new THREE.Mesh(wgeo, new THREE.MeshPhongMaterial({ color: 0x1d5f8f, transparent: true, opacity: 0.55, shininess: 80 }));
  water.position.y = world.sea_level * HSCALE + 0.02;
  scene.add(water);

  cloudTex = new THREE.DataTexture(new Uint8Array(W * H * 4), W, H, THREE.RGBAFormat);
  cloudTex.magFilter = THREE.LinearFilter; cloudTex.minFilter = THREE.LinearFilter;
  const cgeo = new THREE.PlaneGeometry(W, H);
  cgeo.rotateX(-Math.PI / 2);
  clouds = new THREE.Mesh(cgeo, new THREE.MeshBasicMaterial({ map: cloudTex, transparent: true, depthWrite: false }));
  clouds.position.y = HSCALE + 3;
  scene.add(clouds);

  agentMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.35, 10, 8), new THREE.MeshLambertMaterial(), 8000);
  agentMesh.count = 0;
  agentMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  scene.add(agentMesh);
  structMesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.8, 0.6, 0.8), new THREE.MeshLambertMaterial(), 6000);
  structMesh.count = 0;
  scene.add(structMesh);
  ring = new THREE.Mesh(new THREE.TorusGeometry(0.8, 0.12, 8, 24), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  ring.rotation.x = Math.PI / 2; ring.visible = false;
  scene.add(ring);

  camera.position.set(0, W * 0.75, H * 0.85);
  controls.target.set(0, 2, 0);
  controls.update();
  paintTerrain();
}

const lerp = (a, b, t) => a + (b - a) * t;
function ramp(t, stops) {
  t = Math.max(0, Math.min(1, t));
  const k = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
  const f = t * (stops.length - 1) - k;
  return [0, 1, 2].map((c) => lerp(stops[k][c], stops[k + 1][c], f));
}

function paintTerrain() {
  if (!terrain) return;
  const plants = fields.plants, ptype = fields.ptype || world.ptype;
  const f = fields[layer];
  for (let i = 0; i < W * H; i++) {
    let c;
    const e = world.elevation[i];
    const wet = world.water[i];
    if (layer === 'surface' || layer === 'plants') {
      if (wet) c = ramp(e / world.sea_level, [[0.05, 0.12, 0.25], [0.12, 0.35, 0.5]]);
      else {
        const pc = world.plants[ptype[i]].color;
        const b = plants ? plants[i] / 200 : 0.5;
        const rock = [0.45, 0.42, 0.38], snow = [0.92, 0.94, 0.96];
        const h = (e - world.sea_level) / (1 - world.sea_level);
        if (layer === 'plants') c = ramp(b / 1.2, [[0.25, 0.2, 0.15], [0.3, 0.55, 0.2], [0.1, 0.9, 0.3]]);
        else {
          c = [0, 1, 2].map((k) => lerp(rock[k], pc[k], Math.min(1, b * 1.3)));
          if (h > 0.8) c = c.map((v, k) => lerp(v, snow[k], (h - 0.8) * 5));
        }
      }
      const s = fields.struct ? fields.struct[i] : 0;
      if (s) c = [0.9, 0.75, 0.45];
      const fire = fields.fire ? fields.fire[i] / 255 : 0;
      if (fire > 0.05) c = [1, 0.45 * (1 - fire), 0.1];
    } else if (f) {
      const v = f.values[i];
      if (layer === 'temperature') c = ramp((v + 15) / 55, [[0.2, 0.3, 0.9], [0.3, 0.8, 0.9], [0.95, 0.9, 0.3], [0.9, 0.2, 0.15]]);
      else if (layer === 'moisture') c = ramp(v, [[0.6, 0.45, 0.25], [0.3, 0.6, 0.5], [0.1, 0.3, 0.8]]);
      else if (layer === 'pathogens') c = ramp(v / Math.max(0.05, f.max), [[0.08, 0.1, 0.12], [0.4, 0.15, 0.5], [1, 0.3, 0.6]]);
      else if (layer === 'ore') c = wet ? [0.05, 0.1, 0.2] : ramp(v / Math.max(0.5, f.max), [[0.15, 0.13, 0.12], [0.7, 0.45, 0.2], [1, 0.85, 0.4]]);
      else c = [v, v, v];
    } else c = [0.3, 0.3, 0.3];
    terrainColors[i * 3] = c[0]; terrainColors[i * 3 + 1] = c[1]; terrainColors[i * 3 + 2] = c[2];
  }
  terrain.geometry.attributes.color.needsUpdate = true;
}

function paintClouds() {
  if (!cloudTex || !fields.clouds) return;
  const d = cloudTex.image.data;
  const show = $('#show-clouds').checked;
  for (let i = 0; i < W * H; i++) {
    const v = fields.clouds[i];
    d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 240;
    d[i * 4 + 3] = show ? Math.max(0, v - 110) * 0.8 : 0;
  }
  cloudTex.needsUpdate = true;
}

function paintStructures() {
  if (!fields.struct) return;
  let k = 0;
  const colors = { 1: 0xd8b070, 2: 0x9bd35a, 3: 0xc08cff, 4: 0xff8a3d, 5: 0x9ab0c8 };
  for (let i = 0; i < W * H && k < 6000; i++) {
    const s = fields.struct[i];
    if (!s) continue;
    const x = i % W, y = Math.floor(i / W);
    dummy.position.set(x - W / 2 + 0.5, heightAt(x, y) + 0.3, y - H / 2 + 0.5);
    dummy.scale.set(1, s === 2 ? 0.2 : 1, 1);
    dummy.updateMatrix();
    structMesh.setMatrixAt(k, dummy.matrix);
    structMesh.setColorAt(k, tmpColor.setHex(colors[s] || 0xffffff));
    k++;
  }
  structMesh.count = k;
  structMesh.instanceMatrix.needsUpdate = true;
  if (structMesh.instanceColor) structMesh.instanceColor.needsUpdate = true;
}

function groupColor(g) {
  if (g < 0) return tmpColor.setHSL(0, 0, 0.55);
  return tmpColor.setHSL(((g * 137.508) % 360) / 360, 0.75, 0.55);
}

function paintAgents() {
  if (!agentMesh) return;
  const byGroup = $('#color-groups').checked;
  const n = Math.min(agents.length, 8000);
  for (let k = 0; k < n; k++) {
    const [id, x, y, sp, g, act, size, hue, brain, stage] = agents[k];
    const s = 0.6 + size * 1.2 * (stage <= 1 ? 0.6 : 1);
    dummy.position.set(x - W / 2, heightAt(x, y) + 0.3 * s, y - H / 2);
    dummy.scale.set(s, s, s);
    dummy.updateMatrix();
    agentMesh.setMatrixAt(k, dummy.matrix);
    agentMesh.setColorAt(k, byGroup ? groupColor(g) : tmpColor.setHSL(hue, 0.75, 0.45 + brain * 0.3));
  }
  agentMesh.count = n;
  agentMesh.instanceMatrix.needsUpdate = true;
  if (agentMesh.instanceColor) agentMesh.instanceColor.needsUpdate = true;
  if (selected) {
    const a = agents.find((r) => r[0] === selected);
    if (a) {
      ring.visible = true;
      ring.position.set(a[1] - W / 2, heightAt(a[1], a[2]) + 0.15, a[2] - H / 2);
    } else ring.visible = false;
  }
}

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------- picking
const ray = new THREE.Raycaster();
const mouse = new THREE.Vector2();
function pick(ev) {
  const r = canvas.getBoundingClientRect();
  mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(mouse, camera);
  return ray;
}
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const rc = pick(e);
  if ($('#micro-mode').checked) {
    const hit = rc.intersectObject(terrain)[0];
    if (hit) openMicro(Math.floor(hit.point.x + W / 2), Math.floor(hit.point.z + H / 2));
    return;
  }
  const hit = rc.intersectObject(agentMesh)[0];
  if (hit && agents[hit.instanceId]) selectAgent(agents[hit.instanceId][0]);
});
canvas.addEventListener('pointermove', (e) => {
  const rc = pick(e);
  const tip = $('#tooltip');
  const hit = agentMesh && rc.intersectObject(agentMesh)[0];
  if (hit && agents[hit.instanceId]) {
    const a = agents[hit.instanceId];
    tip.style.display = 'block';
    tip.style.left = e.offsetX + 14 + 'px'; tip.style.top = e.offsetY + 10 + 'px';
    tip.textContent = `#${a[0]} · ${meta.actions[a[5]]} · species ${a[3]}${a[4] >= 0 ? ' · group ' + a[4] : ''}`;
  } else tip.style.display = 'none';
});

// ---------------------------------------------------------------- live stream
function connect() {
  const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onmessage = (m) => {
    const f = JSON.parse(m.data);
    agents = f.agents;
    if (f.fields) {
      Object.assign(fields, f.fields);
      if (layer === 'surface' || layer === 'plants') paintTerrain();
      paintClouds();
      paintStructures();
    }
    if (typeof f.sun === 'number') {
      const day = Math.max(0, f.sun);
      sun.intensity = 0.25 + 1.1 * day;
      ambient.intensity = 0.25 + 0.3 * day;
      sun.position.set(Math.cos(f.clock.hour / 24 * Math.PI * 2) * 80, 20 + 80 * day, 40);
      scene.background.setRGB(0.03 + 0.05 * day, 0.05 + 0.07 * day, 0.08 + 0.1 * day);
    }
    $('#clock').textContent = `${f.clock.label.replace(':00', ':' + String(f.minute).padStart(2, '0'))} · ${f.clock.season}${f.clock.night ? ' · night' : ''} · ${f.population} alive`;
    $('#btn-play').textContent = f.paused ? '▶' : '⏸';
    $('#rate').textContent = f.paused ? 'paused' : `${f.effective_rate} h/s`;
    paintAgents();
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

// ---------------------------------------------------------------- panels
const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : v ?? '—');
function bars(el, obj, opts = {}) {
  const entries = Object.entries(obj || {});
  const max = opts.max ?? Math.max(1e-9, ...entries.map(([, v]) => Math.abs(v)));
  el.innerHTML = entries.map(([k, v]) => `<div class="bar"><span>${k}</span><div class="track"><div class="fill ${v < 0 ? 'neg' : ''}" style="width:${Math.min(100, (Math.abs(v) / max) * 100)}%"></div></div><span class="val">${fmt(v, opts.d ?? 2)}</span></div>`).join('');
}
function lineChart(cv, series, colors, opts = {}) {
  const ctx = cv.getContext('2d');
  const cssH = cv.clientHeight || 110;  // height comes from CSS, never from the canvas buffer
  const w = (cv.width = cv.clientWidth * 2), h = (cv.height = cssH * 2);
  ctx.clearRect(0, 0, w, h);
  const all = series.flat().filter((v) => Number.isFinite(v));
  if (!all.length) return;
  const lo = opts.min ?? Math.min(...all), hi = opts.max ?? Math.max(...all);
  series.forEach((s, k) => {
    ctx.strokeStyle = colors[k]; ctx.lineWidth = 3; ctx.beginPath();
    s.forEach((v, i) => {
      const x = (i / Math.max(1, s.length - 1)) * (w - 10) + 5;
      const y = h - 8 - ((v - lo) / Math.max(1e-9, hi - lo)) * (h - 16);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = '#8796a8'; ctx.font = '20px system-ui';
  ctx.fillText(fmt(hi, opts.d ?? 0), 8, 22); ctx.fillText(fmt(lo, opts.d ?? 0), 8, h - 10);
}
const TRAIT_COLORS = { brain: '#ff9f6b', sociality: '#7cc4ff', communication: '#c38cff', size: '#7ee08f', diet: '#ff7a7a', curiosity: '#ffe36b' };

async function refreshState() {
  state = await api('/api/state');
  const L = state.latest || {};
  const k = [
    ['alive', state.population], ['species', L.species ?? '—'], ['generation', L.max_generation ?? 0],
    ['groups', state.civ.groups], ['settlements', state.civ.settlements], ['wars', state.civ.wars],
    ['speakers', L.speakers ?? 0], ['language coh.', fmt(L.language_coherence)], ['ms / tick', state.ms_per_tick],
  ];
  $('#kpis').innerHTML = k.map(([a, b]) => `<div class="kpi"><b>${b}</b><span>${a}</span></div>`).join('');
  knownLayers = state.civ.layers;
  document.querySelectorAll('.hid').forEach((e) => {
    const known = knownLayers.includes(e.dataset.layer);
    e.textContent = known ? 'known' : 'hidden';
    e.classList.toggle('known', known);
  });
  const snaps = state.snapshots;
  const rw = $('#rw-range');
  rw.max = state.clock.tick; if (!rw.dataset.touched) rw.value = state.clock.tick;
  $('#rw-label').textContent = `tick ${rw.value}`;
}

async function refreshStats() {
  const { rows } = await api('/api/stats?limit=400');
  lastStats = rows;
  if (!rows.length) return;
  lineChart($('#ch-pop'), [rows.map((r) => r.population)], ['#7cc4ff']);
  const tr = Object.keys(TRAIT_COLORS);
  lineChart($('#ch-traits'), tr.map((t) => rows.map((r) => r.traits?.[t] ?? NaN)), tr.map((t) => TRAIT_COLORS[t]), { min: 0, max: 1, d: 1 });
  $('#lg-traits').innerHTML = tr.map((t) => `<span><i style="background:${TRAIT_COLORS[t]}"></i>${t}</span>`).join('');
  lineChart($('#ch-species'), [rows.map((r) => r.species)], ['#7ee08f']);
  const last = rows[rows.length - 1];
  const acts = {}; let tot = 0;
  rows.slice(-20).forEach((r) => Object.entries(r.actions || {}).forEach(([a, v]) => { acts[a] = (acts[a] || 0) + v; tot += v; }));
  Object.keys(acts).forEach((a) => (acts[a] /= Math.max(1, tot)));
  bars($('#actions'), acts, { d: 3 });
  bars($('#skills'), last.skills || {}, { max: 1, d: 3 });
  if (meta) {
    bars($('#norms'), Object.fromEntries(meta.virtues.map((v, i) => [v, last.norms?.[i] ?? 0])), { max: 1 });
    bars($('#beliefs'), { spirits: last.beliefs?.[0] ?? 0, naturalism: last.beliefs?.[1] ?? 0, ritual: last.beliefs?.[2] ?? 0 }, { max: 1 });
  }
  const d = await api('/api/deaths');
  bars($('#deaths'), d.causes, { d: 0 });
}

const who = (id, name) => `<a class="who" data-id="${id}">${name || '#' + id}</a>`;
document.addEventListener('click', (e) => {
  const a = e.target.closest('a.who');
  if (a) selectAgent(+a.dataset.id);
});

async function refreshCiv() {
  const [g, s, k, sp] = await Promise.all([api('/api/groups'), api('/api/settlements'), api('/api/knowledge'), api('/api/species')]);
  $('#layers-known').innerHTML = Object.entries(k.all_layers).map(([n, d]) => {
    const kn = k.layers[n];
    return `<div class="card"><h4>${kn ? '✓' : '○'} ${n}</h4><div class="small muted">${d}</div>${kn ? `<div class="small">discovered at tick ${kn.tick}${kn.by ? ' by ' + who(kn.by) : ''}</div>` : '<div class="small muted">not yet perceived by inhabitants (still acts on them)</div>'}</div>`;
  }).join('');
  const live = g.groups.filter((x) => x.dissolved == null).sort((a, b) => b.size - a.size).slice(0, 12);
  $('#groups').innerHTML = live.map((x) => `<div class="card"><h4>${x.name} <span class="pill">${x.polity}</span></h4>
    <div class="grid2"><span>members</span><span>${x.size}</span><span>founded</span><span>tick ${x.founded}</span><span>leader</span><span>${who(x.leader)}</span>
    <span>language coherence</span><span>${fmt(x.language_coherence)}</span><span>mean brain</span><span>${fmt(x.mean_brain)}</span>
    <span>settlements</span><span>${x.settlements ?? 0}</span></div>
    <div class="small">${Object.entries(x.vocabulary || {}).slice(0, 6).map(([c, v]) => `<span class="pill">${c.replace('plant:', '')}: “${v.word}”</span>`).join('')}</div></div>`).join('') || '<p class="muted">No stable groups yet.</p>';
  $('#settlements').innerHTML = s.settlements.filter((x) => x.abandoned == null).map((x) => `<div class="card"><h4>${x.name}</h4><div class="grid2"><span>founded</span><span>tick ${x.founded}</span><span>population nearby</span><span>${x.population}</span><span>structures</span><span>${Object.entries(x.structures || {}).map(([a, b]) => a + ' ' + b).join(', ')}</span></div></div>`).join('') || '<p class="muted">No settlements yet.</p>';
  $('#species').innerHTML = sp.species.filter((x) => x.extinct == null).sort((a, b) => b.count - a.count).slice(0, 20).map((x) =>
    `<div class="card"><h4>${x.name} <span class="pill">${x.count}</span></h4><div class="small muted">origin tick ${x.origin}${x.parent ? ', from species ' + x.parent : ''} · peak ${x.peak}</div>
    <div class="grid2 small">${['size', 'brain', 'diet', 'sociality', 'speed', 'aquatic'].map((t) => `<span>${t}</span><span>${fmt(x.traits[t])}</span>`).join('')}</div></div>`).join('');
  const f = await api('/api/firsts');
  $('#firsts').innerHTML = f.firsts.map((x) => `<li><span class="t">t=${x.tick}</span>${x.text}</li>`).join('');
  const ev = await api('/api/events?limit=60');
  $('#events').innerHTML = ev.events.filter((x) => x.kind !== 'first').map((x) => `<li><span class="t">t=${x.tick}</span>${x.kind.replace(/_/g, ' ')} ${x.agent > 0 ? who(x.agent) : ''} <span class="muted small">${JSON.stringify(x.data).slice(0, 80)}</span></li>`).join('');
}

function tree(n) {
  if (!n) return '';
  const label = `${who(n.id, n.name)} <span class="muted small">gen ${n.generation ?? '?'}${n.died != null ? ', died ' + (n.cause || '') : ''}</span>`;
  const kids = (n.parents || n.children || []);
  return `<li class="${n.died != null ? 'dead' : ''}">${label}${kids.length ? '<ul>' + kids.map(tree).join('') + '</ul>' : ''}${n.n_children > (n.children || []).length ? `<div class="small muted">+${n.n_children - n.children.length} more</div>` : ''}</li>`;
}

async function selectAgent(id) {
  selected = id;
  document.querySelector('[data-tab="organism"]').click();
  const [d, lin] = await Promise.all([api('/api/agent/' + id), api(`/api/agent/${id}/lineage`)]);
  $('#org-empty').style.display = 'none';
  if (d.detail) { $('#org').innerHTML = `<p class="muted">${d.detail}</p>`; return; }
  const bio = (d.biography || []).map((e) => `<li><span class="t">t=${e.tick}</span>${e.kind.replace(/_/g, ' ')} ${e.other > 0 && e.other !== id ? who(e.other) : ''} <span class="muted small">${JSON.stringify(e.data).slice(0, 70)}</span></li>`).join('');
  if (!d.alive) {
    const r = d.record;
    $('#org').innerHTML = `<div class="card"><h4>${r.name || '#' + id} <span class="pill">deceased</span></h4><div class="grid2"><span>born</span><span>tick ${r.birth_tick}</span><span>died</span><span>tick ${r.death_tick} (${r.cause})</span><span>generation</span><span>${r.generation}</span></div></div>
      <h3>Last memories</h3><ol class="timeline">${(r.final?.memories || []).map((m) => `<li><span class="t">t=${m.tick}</span>${m.detail || m.type}</li>`).join('')}</ol>
      <h3>Lineage</h3><div class="tree"><ul>${tree(lin.ancestors)}</ul><h3>Descendants</h3><ul>${tree(lin.descendants)}</ul></div><h3>Biography</h3><ol class="timeline">${bio}</ol>`;
    return;
  }
  const bar = (o, max = 1) => Object.entries(o).map(([k, v]) => `<div class="bar"><span>${k}</span><div class="track"><div class="fill ${v < 0 ? 'neg' : ''}" style="width:${Math.min(100, Math.abs(v) / max * 100)}%"></div></div><span class="val">${fmt(v)}</span></div>`).join('');
  $('#org').innerHTML = `
    <div class="card"><h4>${d.name} <span class="pill">${d.stage}</span> <span class="pill">${d.action}</span></h4>
      <div class="grid2"><span>species</span><span>${d.species.name}</span><span>group</span><span>${d.group ? d.group.name + ' (' + d.group.polity + ')' : '—'}</span>
      <span>age</span><span>${d.age_days} / ~${d.max_age_days} days</span><span>generation</span><span>${d.generation}</span>
      <span>parents</span><span>${d.parents.filter((p) => p > 0).map((p) => who(p)).join(', ') || 'first generation'}</span>
      <span>partner</span><span>${d.partner > 0 ? who(d.partner, d.partner_name) : '—'}</span><span>children / kills</span><span>${d.children} / ${d.kills}</span>
      <span>energy</span><span>${fmt(d.energy)} / ${fmt(d.capacity)}</span><span>health · water</span><span>${fmt(d.health)} · ${fmt(d.hydration)}</span>
      <span>infection (hidden)</span><span>${fmt(d.infection)}</span><span>prestige</span><span>${fmt(d.prestige)}</span></div></div>
    <h3>Emotions (functional model)</h3><div class="bars">${bar(d.emotions)}</div>
    <h3>Personality</h3><div class="bars">${bar(d.personality)}</div>
    <h3>Learned food values vs. innate</h3><div class="bars">${bar(Object.fromEntries(Object.entries(d.food_values).map(([k, v]) => [k + ' (' + fmt(d.innate_food[k], 1) + ')', v])), 1.5)}</div>
    <h3>Habits</h3><div class="bars">${bar(d.habits, 2)}</div>
    <h3>Skills</h3><div>${Object.entries(d.skills).map(([k, v]) => `<span class="pill">${k} ${fmt(v)}</span>`).join('') || '<span class="muted">none</span>'}</div>
    <h3>Inventory</h3><div>${Object.entries(d.inventory).map(([k, v]) => `<span class="pill">${k} ${fmt(v, 1)}</span>`).join('') || '<span class="muted">empty</span>'}</div>
    <h3>Norms</h3><div class="bars">${bar(d.norms)}</div>
    <h3>Beliefs</h3><div class="bars">${bar(d.beliefs)}</div>
    <h3>Words</h3><div>${Object.entries(d.lexicon).map(([c, w]) => `<span class="pill">${c.replace('plant:', '')}: “${w}”</span>`).join('') || '<span class="muted">no words</span>'}</div>
    <h3>Relationships</h3><div class="bars">${d.relations.map((r) => `<div class="bar"><span>${who(r.id, r.name)}</span><div class="track"><div class="fill ${r.affinity < 0 ? 'neg' : ''}" style="width:${Math.abs(r.affinity) * 100}%"></div></div><span class="val">${fmt(r.affinity)}</span></div>`).join('')}</div>
    <h3>Short-term memory</h3><ol class="timeline">${d.short_term_memory.map((m) => `<li><span class="t">t=${m.tick}</span>${m.type} (${fmt(m.mag)})</li>`).join('')}</ol>
    <h3>Long-term memory</h3><ol class="timeline">${d.long_term_memory.slice().reverse().map((m) => `<li><span class="t">t=${m.tick}</span>${m.detail || m.type} <span class="muted small">vividness ${fmt(m.vividness)}</span></li>`).join('')}</ol>
    <h3>Lineage</h3><div class="tree"><ul>${tree(lin.ancestors)}</ul><h3>Descendants</h3><ul>${tree(lin.descendants)}</ul></div>
    <h3>Recorded history</h3><ol class="timeline">${bio}</ol>`;
}

// ---------------------------------------------------------------- micro view
async function openMicro(x, y) {
  const m = await api(`/api/micro?x=${x}&y=${y}`);
  $('#micro').classList.remove('hidden');
  $('#micro-info').textContent = `cell (${x}, ${y}) · ${m.count} microbes · virulence ${fmt(m.mean_virulence)}`;
  $('#micro-note').textContent = m.known_to_inhabitants ? 'Inhabitants have discovered this layer.' : 'Inhabitants cannot perceive this layer yet — it still causes disease and decay.';
  const cv = $('#micro-canvas'), ctx = cv.getContext('2d');
  ctx.fillStyle = '#071016'; ctx.fillRect(0, 0, cv.width, cv.height);
  m.decomposers.forEach(([a, b]) => { ctx.fillStyle = '#5fa46b88'; ctx.beginPath(); ctx.arc(a * 420, b * 420, 2, 0, 6.3); ctx.fill(); });
  m.microbes.forEach(([a, b, v]) => { ctx.fillStyle = `hsl(${320 - v * 60},80%,${50 + v * 15}%)`; ctx.beginPath(); ctx.ellipse(a * 420, b * 420, 4, 2.2, a * 9, 0, 6.3); ctx.fill(); });
}
$('#micro-close').onclick = () => $('#micro').classList.add('hidden');

// ---------------------------------------------------------------- controls
$('#btn-play').onclick = () => post({ action: state && !state.paused ? 'pause' : 'resume' }).then(refreshState);
$('#btn-step').onclick = () => post({ action: 'step', value: 1 }).then(refreshState);
$('#scale').onchange = (e) => post({ action: 'scale', value: e.target.value });
document.querySelectorAll('input[name=layer]').forEach((r) => r.addEventListener('change', async (e) => {
  layer = e.target.value;
  if (!['surface', 'plants'].includes(layer)) fields[layer] = await api('/api/field/' + layer);
  paintTerrain();
}));
$('#show-clouds').onchange = paintClouds;
$('#color-groups').onchange = paintAgents;
document.querySelectorAll('#tabs button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('#tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === 'tab-' + b.dataset.tab));
  if (b.dataset.tab === 'civ' || b.dataset.tab === 'evolution' || b.dataset.tab === 'history') refreshCiv();
}));
$('#rw-range').oninput = (e) => { e.target.dataset.touched = 1; $('#rw-label').textContent = 'tick ' + e.target.value; };
$('#rw-go').onclick = async () => {
  const r = await post({ action: 'rewind', value: +$('#rw-range').value });
  delete $('#rw-range').dataset.touched;
  if (r.detail) alert(r.detail);
  refreshState(); refreshStats(); refreshCiv();
};
$('#btn-save').onclick = async () => { const r = await post({ action: 'save', value: $('#save-path').value }); alert(r.ok ? 'Saved to ' + r.path : r.detail); };
$('#btn-load').onclick = async () => { await post({ action: 'load', value: $('#save-path').value }); await init(); };
$('#nw-go').onclick = async () => {
  await post({ action: 'new', seed: +$('#nw-seed').value, scenario: $('#nw-scn').value, framework: $('#nw-fw').value });
  await init();
};

async function init() {
  meta = world = await api('/api/world');
  W = world.width; H = world.height;
  fields = { ptype: world.ptype };
  const mor = await api('/api/morality');
  $('#nw-fw').innerHTML = Object.entries(mor.frameworks).map(([k, v]) => `<option value="${k}" title="${v.description}" ${k === mor.current ? 'selected' : ''}>${k}</option>`).join('');
  buildWorld();
  resize();
  refreshState(); refreshStats();
}

await init();
animate();
connect();
setInterval(refreshState, 1000);
setInterval(refreshStats, 4000);
setInterval(() => { const t = document.querySelector('#tabs button.active').dataset.tab; if (t !== 'overview' && t !== 'organism') refreshCiv(); }, 6000);
