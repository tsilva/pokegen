const TYPE_COLORS = {
  normal: "#A8A77A", fire: "#EE8130", water: "#6390F0", electric: "#F7D02C",
  grass: "#7AC74C", ice: "#96D9D6", fighting: "#C22E28", poison: "#A33EA1",
  ground: "#E2BF65", flying: "#A98FF3", psychic: "#F95587", bug: "#A6B91A",
  rock: "#B6A136", ghost: "#735797", dragon: "#6F35FC", dark: "#705746",
  steel: "#B7B7CE", fairy: "#D685AD",
};

const TYPE_LIST = Object.keys(TYPE_COLORS);

const GENS = [
  { label: "All generations", min: 1, max: 1025 },
  { label: "Gen I · Kanto (1–151)", min: 1, max: 151 },
  { label: "Gen II · Johto (152–251)", min: 152, max: 251 },
  { label: "Gen III · Hoenn (252–386)", min: 252, max: 386 },
  { label: "Gen IV · Sinnoh (387–493)", min: 387, max: 493 },
  { label: "Gen V · Unova (494–649)", min: 494, max: 649 },
  { label: "Gen VI · Kalos (650–721)", min: 650, max: 721 },
  { label: "Gen VII · Alola (722–809)", min: 722, max: 809 },
  { label: "Gen VIII · Galar (810–905)", min: 810, max: 905 },
  { label: "Gen IX · Paldea (906–1025)", min: 906, max: 1025 },
];

const STATS = [
  ["hp", "HP"],
  ["attack", "Attack"],
  ["defense", "Defense"],
  ["special-attack", "Sp. Atk"],
  ["special-defense", "Sp. Def"],
  ["speed", "Speed"],
];

const STAT_MAX = 255;
const STAT_COLORS = {
  hp: "#ff5959", attack: "#f5ac78", defense: "#fae078",
  "special-attack": "#9db7f5", "special-defense": "#a7db8d", speed: "#fa92b2",
};

const state = {
  all: [],
  filtered: [],
  search: "",
  types: new Set(),
  gen: 0,
  sort: "id-asc",
  view: "artwork",
  index: -1,
};

const $ = (sel) => document.querySelector(sel);
const grid = $("#grid");
const modal = $("#modal");
const detail = $("#detail");
const search = $("#search");

const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const prettyName = (name) =>
  name.split("-").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");

const dexId = (id) => "#" + String(id).padStart(4, "0");

const imgPath = (p, kind = state.view) =>
  `data/${kind === "sprites" ? "sprites" : "artwork"}/${String(p.id).padStart(4, "0")}-${p.name}.png`;

async function init() {
  const res = await fetch("data/metadata.json");
  if (!res.ok) throw new Error(`metadata.json: HTTP ${res.status}`);
  state.all = await res.json();

  const genSelect = $("#gen");
  GENS.forEach((g, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = g.label;
    genSelect.append(opt);
  });

  const typesEl = $("#types");
  for (const t of TYPE_LIST) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.dataset.type = t;
    chip.textContent = t;
    chip.style.background = TYPE_COLORS[t];
    typesEl.append(chip);
  }

  typesEl.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    const t = chip.dataset.type;
    state.types.has(t) ? state.types.delete(t) : state.types.add(t);
    chip.classList.toggle("active", state.types.has(t));
    applyFilters();
  });

  search.addEventListener("input", () => {
    state.search = search.value;
    applyFilters();
  });

  genSelect.addEventListener("change", () => {
    state.gen = Number(genSelect.value);
    applyFilters();
  });

  $("#sort").addEventListener("change", (e) => {
    state.sort = e.target.value;
    applyFilters();
  });

  $("#viewtoggle").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-view]");
    if (!btn) return;
    setView(btn.dataset.view);
  });

  $("#clear").addEventListener("click", clearFilters);
  $("#random").addEventListener("click", openRandom);

  grid.addEventListener("click", (e) => {
    const card = e.target.closest(".card");
    if (card) openDetail(Number(card.dataset.index));
  });

  document.addEventListener("keydown", onKeydown);
  window.addEventListener("hashchange", applyHash);
  modal.addEventListener("click", (e) => {
    const act = e.target.closest("[data-action]");
    if (e.target.closest("[data-close]")) return closeDetail();
    if (!act) return;
    if (act.dataset.action === "prev") step(-1);
    if (act.dataset.action === "next") step(1);
    if (act.dataset.action === "view") setView(state.view === "artwork" ? "sprites" : "artwork");
  });

  applyFilters();
  applyHash();
}

function clearFilters() {
  state.search = "";
  state.types.clear();
  state.gen = 0;
  state.sort = "id-asc";
  search.value = "";
  $("#gen").value = "0";
  $("#sort").value = "id-asc";
  document.querySelectorAll(".chip.active").forEach((c) => c.classList.remove("active"));
  applyFilters();
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("#viewtoggle button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === view)
  );
  renderGrid();
  if (state.index >= 0) renderDetail(state.filtered[state.index]);
}

function matches(p, gen) {
  if (p.id < gen.min || p.id > gen.max) return false;
  if (state.types.size && !p.types.some((t) => state.types.has(t))) return false;
  const q = state.search.trim().toLowerCase();
  if (!q) return true;
  const qn = q.replace(/^#/, "");
  if (/^\d+$/.test(qn) && Number(qn) === p.id) return true;
  return p.name.includes(q) || prettyName(p.name).toLowerCase().includes(q);
}

function applyFilters() {
  const gen = GENS[state.gen];
  state.filtered = state.all.filter((p) => matches(p, gen));
  const [key, dir] = state.sort.split("-");
  const sign = dir === "asc" ? 1 : -1;
  state.filtered.sort((a, b) =>
    key === "name" ? sign * a.name.localeCompare(b.name) : sign * (a.id - b.id)
  );
  renderGrid();
}

function renderGrid() {
  const frag = document.createDocumentFragment();
  for (const [i, p] of state.filtered.entries()) {
    const card = document.createElement("button");
    card.className = "card";
    card.dataset.index = i;
    card.dataset.id = p.id;
    card.style.setProperty("--c", TYPE_COLORS[p.types[0]] || "#666");
    card.innerHTML = `
      <span class="dex">${dexId(p.id)}</span>
      <img class="${state.view === "sprites" ? "pixel" : ""}" loading="lazy" decoding="async"
           src="${imgPath(p)}" alt="${esc(prettyName(p.name))}">
      <span class="name">${esc(prettyName(p.name))}</span>
      <span class="badges">${p.types
        .map((t) => `<span class="badge" style="background:${TYPE_COLORS[t] || "#666"}">${esc(t)}</span>`)
        .join("")}</span>`;
    frag.append(card);
  }
  grid.replaceChildren(frag);
  $("#count").textContent = `${state.filtered.length} / ${state.all.length}`;
  $("#empty").hidden = state.filtered.length > 0;
}

function statRow([key, label], value) {
  const pct = Math.min(100, (value / STAT_MAX) * 100);
  return `
    <div class="stat">
      <span class="label">${label}</span>
      <span class="val">${value}</span>
      <span class="track"><span class="fill" style="width:${pct}%;background:${STAT_COLORS[key]}"></span></span>
    </div>`;
}

function renderDetail(p) {
  const total = STATS.reduce((sum, [k]) => sum + (p.base_stats?.[k] || 0), 0);
  detail.style.setProperty("--c", TYPE_COLORS[p.types[0]] || "#666");
  detail.innerHTML = `
    <button class="close" data-close aria-label="Close">×</button>
    <button class="nav prev" data-action="prev" aria-label="Previous">‹</button>
    <button class="nav next" data-action="next" aria-label="Next">›</button>
    <div class="detail-art">
      <img class="${state.view === "sprites" ? "pixel" : ""}" src="${imgPath(p)}"
           alt="${esc(prettyName(p.name))} ${state.view}">
    </div>
    <div class="detail-info">
      <div class="detail-head">
        <span class="dex">${dexId(p.id)}</span>
        <h2>${esc(prettyName(p.name))}</h2>
      </div>
      <div class="badges">${p.types
        .map((t) => `<span class="badge" style="background:${TYPE_COLORS[t] || "#666"}">${esc(t)}</span>`)
        .join("")}</div>
      <div class="stats">
        ${STATS.map(([k, label]) => statRow([k, label], p.base_stats?.[k] || 0)).join("")}
        <div class="stat total">
          <span class="label">Total</span>
          <span class="val">${total}</span>
          <span></span>
        </div>
      </div>
      <div class="facts">
        <span>Height <b>${(p.height / 10).toFixed(1)} m</b></span>
        <span>Weight <b>${(p.weight / 10).toFixed(1)} kg</b></span>
      </div>
      <div class="abilities">
        ${(p.abilities || [])
          .map((a) => `<span class="ability">${esc(prettyName(a))}</span>`)
          .join("")}
      </div>
      <div class="detail-actions">
        <button data-action="view" class="ghost">Show ${state.view === "artwork" ? "pixel sprite" : "official artwork"}</button>
        <button data-action="next" class="ghost">Next ›</button>
      </div>
      <div class="kbd-hint">← → to browse · Esc to close</div>
    </div>`;
}

function preloadNeighbors() {
  for (const d of [-1, 1]) {
    const p = state.filtered[(state.index + d + state.filtered.length) % state.filtered.length];
    if (p) new Image().src = imgPath(p, "artwork");
  }
}

function openDetail(index) {
  const p = state.filtered[index];
  if (!p) return;
  state.index = index;
  renderDetail(p);
  modal.hidden = false;
  document.body.style.overflow = "hidden";
  preloadNeighbors();
  if (location.hash !== `#/${p.id}`) history.pushState(null, "", `#/${p.id}`);
  detail.querySelector(".close").focus({ preventScroll: true });
}

function closeDetail() {
  if (state.index < 0) return;
  const idx = state.index;
  state.index = -1;
  modal.hidden = true;
  document.body.style.overflow = "";
  history.replaceState(null, "", location.pathname + location.search);
  grid.querySelector(`.card[data-index="${idx}"]`)?.focus({ preventScroll: true });
}

function step(delta) {
  if (state.index < 0 || !state.filtered.length) return;
  const next = (state.index + delta + state.filtered.length) % state.filtered.length;
  openDetail(next);
}

function openRandom() {
  const pool = state.filtered.length ? state.filtered : state.all;
  const p = pool[Math.floor(Math.random() * pool.length)];
  const idx = state.filtered.indexOf(p);
  if (idx >= 0) return openDetail(idx);
  clearFilters();
  openDetail(state.filtered.findIndex((x) => x.id === p.id));
}

function applyHash() {
  const m = location.hash.match(/^#\/(\d+)$/);
  if (!m) {
    if (state.index >= 0) closeDetail();
    return;
  }
  const id = Number(m[1]);
  let idx = state.filtered.findIndex((p) => p.id === id);
  if (idx < 0) {
    clearFilters();
    idx = state.filtered.findIndex((p) => p.id === id);
  }
  if (idx >= 0 && state.index !== idx) openDetail(idx);
}

const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

function gridColumns() {
  return getComputedStyle(grid).gridTemplateColumns.split(" ").length;
}

function onKeydown(e) {
  const inModal = !modal.hidden;
  const target = e.target instanceof Element ? e.target : document.body;
  const inField = target.matches("input, select, textarea");

  if (e.key === "/" && !inField && !inModal) {
    e.preventDefault();
    search.focus();
    return;
  }

  if (inModal) {
    if (e.key === "Escape") return closeDetail();
    if (e.key === "ArrowLeft") return step(-1);
    if (e.key === "ArrowRight") return step(1);
    return;
  }

  if (inField) {
    if (e.key === "Escape") {
      if (search.value) {
        search.value = "";
        state.search = "";
        applyFilters();
      } else {
        search.blur();
      }
    }
    return;
  }

  const active = document.activeElement;
  if (!active?.classList.contains("card")) {
    if (e.key === "ArrowDown" && state.filtered.length) {
      e.preventDefault();
      grid.querySelector(".card")?.focus();
    }
    return;
  }

  if (e.key === "Enter" || e.key === " ") return;
  const vec = ARROWS[e.key];
  if (!vec) return;
  e.preventDefault();
  const [dx, dy] = vec;
  const nextIndex = Number(active.dataset.index) + dx + dy * gridColumns();
  const card = grid.querySelector(`.card[data-index="${nextIndex}"]`);
  if (card) card.focus();
  else if (dy > 0) grid.querySelector(".card:last-child")?.focus();
}

init().catch((err) => {
  document.body.insertAdjacentHTML(
    "beforeend",
    `<p class="empty">Failed to load dataset: ${esc(err.message)}</p>`
  );
});
