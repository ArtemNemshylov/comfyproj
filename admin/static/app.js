// compfui-admin — фронтенд (vanilla JS, без збірки).
// Спілкується з бекендом через /api/* (див. admin/app/routers/*.py).

const state = {
  characters: [],
  selectedId: null,
  detail: null,       // { character, versions }
  photoAssets: [],
  videoAssets: [],
  activeTab: "overview",
  trainPollTimer: null,
};

const POSE_PRESETS = [
  "портрет анфас, дивиться в камеру",
  "сидить на траві",
  "біжить у парку",
  "лежить, відпочиває",
  "стрибає за м'ячиком",
  "крупний план морди",
];

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c === null || c === undefined) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) {
    const msg = (data && data.detail) ? data.detail : `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data;
}

function badge(status) {
  return el("span", { class: `badge ${status}` }, status);
}

// ---------- health ----------
async function refreshHealth() {
  const h = document.getElementById("health");
  try {
    const data = await api("/api/health");
    h.textContent = data.comfyui_alive ? "ComfyUI: онлайн" : "ComfyUI: офлайн (запустіть run_all.sh)";
    h.className = "health " + (data.comfyui_alive ? "ok" : "err");
  } catch (e) {
    h.textContent = "Бекенд недоступний";
    h.className = "health err";
  }
}

// ---------- sidebar ----------
async function loadCharacters() {
  state.characters = await api("/api/characters");
  renderSidebar();
}

function renderSidebar() {
  const list = document.getElementById("charList");
  list.innerHTML = "";
  for (const c of state.characters) {
    const card = el("div", {
      class: "char-card" + (c.id === state.selectedId ? " active" : ""),
      onclick: () => selectCharacter(c.id),
    }, [
      el("span", { class: "char-name" }, c.name),
      badge(c.status),
    ]);
    list.appendChild(card);
  }
}

async function selectCharacter(id) {
  state.selectedId = id;
  state.activeTab = "overview";
  renderSidebar();
  await loadDetail();
}

async function loadDetail() {
  if (!state.selectedId) return;
  state.detail = await api(`/api/characters/${state.selectedId}`);
  const [photos, videos] = await Promise.all([
    api(`/api/characters/${state.selectedId}/assets?asset_type=photo`),
    api(`/api/characters/${state.selectedId}/assets?asset_type=video`),
  ]);
  state.photoAssets = photos;
  state.videoAssets = videos;
  renderMain();
}

// ---------- main panel ----------
function renderMain() {
  const main = document.getElementById("main");
  main.innerHTML = "";
  if (!state.detail) {
    main.appendChild(el("div", { class: "empty-state" }, el("p", {}, "Оберіть персонажа зліва або створіть нового.")));
    return;
  }
  const { character, versions } = state.detail;

  const header = el("div", { class: "card" }, [
    el("div", { style: "display:flex;justify-content:space-between;align-items:center;" }, [
      el("div", {}, [
        el("h2", { style: "margin:0" }, character.name),
        el("div", { class: "muted" }, character.base_description || "(без опису)"),
      ]),
      badge(character.status),
    ]),
  ]);
  main.appendChild(header);

  const tabs = el("div", { class: "tabs" }, [
    tabBtn("overview", "Огляд і памʼять"),
    tabBtn("generate", "Генерація фото"),
    tabBtn("video", "Відео"),
    tabBtn("newtrait", "Нова риса / перетренувати"),
  ]);
  main.appendChild(tabs);

  const body = el("div", { id: "tabBody" });
  main.appendChild(body);

  if (state.activeTab === "overview") renderOverview(body, character, versions);
  else if (state.activeTab === "generate") renderGenerate(body, character, versions);
  else if (state.activeTab === "video") renderVideo(body, character, versions);
  else if (state.activeTab === "newtrait") renderNewTrait(body, character, versions);
}

function tabBtn(key, label) {
  return el("div", {
    class: "tab" + (state.activeTab === key ? " active" : ""),
    onclick: () => { state.activeTab = key; renderMain(); },
  }, label);
}

function latestVersion(versions) {
  return versions[versions.length - 1];
}

function renderOverview(body, character, versions) {
  const v = latestVersion(versions);

  const card = el("div", { class: "card" });
  card.appendChild(el("h3", { style: "margin-top:0" }, "Поточна (найновіша) версія"));
  card.appendChild(el("p", {}, [
    el("b", {}, `v${v.version_number}: `),
    v.trait_note || "—",
    " ",
    badge(v.status),
  ]));
  card.appendChild(el("p", { class: "muted" }, `Кумулятивний опис для промптів: "${v.prompt_prefix}"`));

  if (v.reference_image_urls && v.reference_image_urls.length) {
    const gal = el("div", { class: "grid" });
    for (const url of v.reference_image_urls) {
      gal.appendChild(el("div", { class: "asset-card" }, el("img", { src: url })));
    }
    card.appendChild(gal);
  }

  if (v.status === "pending" || v.status === "failed" || v.status === "rejected") {
    card.appendChild(el("button", {
      class: "btn btn-primary", style: "margin-top:10px",
      onclick: () => startTraining(character.id, v.id),
    }, v.status !== "pending" ? "Тренувати ще раз (з поточними налаштуваннями якості)" : "Тренувати модель (LoRA)"));
    if (v.status === "failed") card.appendChild(el("div", { class: "error-banner" }, "Попереднє тренування завершилось помилкою. Дивіться логи в admin/data/jobs/."));
  } else if (v.status === "training") {
    card.appendChild(el("div", { id: "trainProgress" }, "Тренування триває..."));
    card.appendChild(el("div", { class: "progress-bar" }, el("div", { id: "trainBar", style: "width:0%" })));
    pollTraining(character.id, v.id);
  } else if (v.status === "ready_for_review") {
    card.appendChild(el("p", {}, "Тренування завершено. Перегляньте прев'ю нижче і підтвердьте портрет (або відхиліть і перетренуйте)."));
    const actions = el("div", { style: "display:flex;gap:8px" }, [
      el("button", { class: "btn btn-primary", onclick: () => confirmVersion(character.id, v.id) }, "Підтвердити портрет"),
      el("button", { class: "btn", onclick: () => rejectVersion(character.id, v.id) }, "Відхилити"),
    ]);
    card.appendChild(actions);
  } else if (v.status === "confirmed") {
    card.appendChild(el("p", { class: "muted" }, `Підтверджено ${v.confirmed_at || ""}. Ця версія використовується для всіх нових фото/відео.`));
  }
  body.appendChild(card);

  // прев'ю/усі фото цієї версії
  const previewCard = el("div", { class: "card" });
  previewCard.appendChild(el("h3", { style: "margin-top:0" }, "Згенеровані фото цього персонажа"));
  const assetsForVersion = state.photoAssets.filter(a => a.character_version_id === v.id);
  previewCard.appendChild(assetGrid(assetsForVersion, character.id));
  body.appendChild(previewCard);

  if (versions.length > 1) {
    const historyCard = el("div", { class: "card" });
    historyCard.appendChild(el("h3", { style: "margin-top:0" }, "Історія версій (памʼять персонажа)"));
    for (const hv of versions.slice().reverse()) {
      historyCard.appendChild(el("div", { class: "version-row" }, [
        el("div", {}, [el("b", {}, `v${hv.version_number}`), " — ", hv.trait_note || "—"]),
        badge(hv.status),
      ]));
    }
    body.appendChild(historyCard);
  }
}

async function startTraining(charId, versionId) {
  try {
    await api(`/api/characters/${charId}/versions/${versionId}/train`, { method: "POST" });
    await loadDetail();
  } catch (e) { alert("Помилка запуску тренування: " + e.message); }
}

function pollTraining(charId, versionId) {
  clearInterval(state.trainPollTimer);
  state.trainPollTimer = setInterval(async () => {
    try {
      const res = await api(`/api/characters/${charId}/versions/${versionId}/training-status`);
      const bar = document.getElementById("trainBar");
      const label = document.getElementById("trainProgress");
      const prog = res.progress || {};
      if (bar && prog.total) {
        const pct = Math.min(100, Math.round(100 * (prog.step || 0) / prog.total));
        bar.style.width = pct + "%";
      }
      if (label) label.textContent = prog.message || res.status;
      if (res.status !== "training") {
        clearInterval(state.trainPollTimer);
        await loadDetail();
      }
    } catch (e) {
      clearInterval(state.trainPollTimer);
    }
  }, 3000);
}

async function confirmVersion(charId, versionId) {
  try {
    await api(`/api/characters/${charId}/versions/${versionId}/confirm`, { method: "POST" });
    await loadDetail();
  } catch (e) { alert("Помилка підтвердження: " + e.message); }
}

async function rejectVersion(charId, versionId) {
  try {
    await api(`/api/characters/${charId}/versions/${versionId}/reject`, { method: "POST" });
    await loadDetail();
  } catch (e) { alert("Помилка: " + e.message); }
}

function assetGrid(assets, charId) {
  const grid = el("div", { class: "grid" });
  if (!assets.length) {
    grid.appendChild(el("div", { class: "muted" }, "Поки що нічого не згенеровано."));
    return grid;
  }
  for (const a of assets) {
    const media = a.type === "video"
      ? el("video", { src: a.url, controls: "true" })
      : el("img", { src: a.url });
    const card = el("div", { class: "asset-card" }, [
      media,
      el("div", { class: "asset-meta" }, a.meta && (a.meta.pose_prompt || a.meta.motion_prompt) || ""),
      el("div", { class: "asset-actions" }, [
        el("span", { class: "ig-status" }, "IG: " + a.instagram_status),
        el("button", { class: "btn btn-small", onclick: () => deployInstagram(a.id) }, "Deploy → Instagram"),
      ]),
    ]);
    grid.appendChild(card);
  }
  return grid;
}

async function deployInstagram(assetId) {
  try {
    const res = await api(`/api/assets/${assetId}/publish/instagram`, { method: "POST" });
    alert(res.message);
    await loadDetail();
  } catch (e) { alert("Помилка: " + e.message); }
}

function renderGenerate(body, character, versions) {
  const v = character.current_version_id ? versions.find(x => x.id === character.current_version_id) : null;
  if (!v) {
    body.appendChild(el("div", { class: "card" }, "У персонажа ще немає підтвердженої версії — спершу натренуйте й підтвердіть портрет у вкладці «Огляд і памʼять»."));
    return;
  }
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", { style: "margin-top:0" }, `Генерація фото (v${v.version_number}, підтверджено)`));

  const promptInput = el("textarea", { placeholder: "Опишіть позу/сцену, напр. «сидить на пляжі на заході сонця»" });
  const chips = el("div", { class: "pose-presets" });
  for (const pose of POSE_PRESETS) {
    chips.appendChild(el("span", { class: "pose-chip", onclick: () => { promptInput.value = pose; } }, pose));
  }
  card.appendChild(chips);
  card.appendChild(el("div", { class: "form-row" }, [el("label", {}, "Промпт пози/сцени"), promptInput]));

  const genBtn = el("button", { class: "btn btn-primary" }, "Згенерувати фото");
  const statusEl = el("span", { class: "muted", style: "margin-left:10px" });
  genBtn.addEventListener("click", async () => {
    if (!promptInput.value.trim()) { alert("Введіть опис пози"); return; }
    genBtn.disabled = true;
    statusEl.textContent = "Генерація... (може тривати десятки секунд)";
    try {
      await api(`/api/characters/${character.id}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pose_prompt: promptInput.value.trim() }),
      });
      statusEl.textContent = "Готово";
      await loadDetail();
    } catch (e) {
      statusEl.textContent = "";
      alert("Помилка генерації: " + e.message);
    } finally {
      genBtn.disabled = false;
    }
  });
  card.appendChild(genBtn);
  card.appendChild(statusEl);
  body.appendChild(card);

  const galCard = el("div", { class: "card" });
  galCard.appendChild(el("h3", { style: "margin-top:0" }, "Усі фото персонажа"));
  galCard.appendChild(assetGrid(state.photoAssets, character.id));
  body.appendChild(galCard);
}

function renderVideo(body, character, versions) {
  const v = character.current_version_id ? versions.find(x => x.id === character.current_version_id) : null;
  if (!v) {
    body.appendChild(el("div", { class: "card" }, "У персонажа ще немає підтвердженої версії — спершу натренуйте й підтвердіть портрет."));
    return;
  }
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", { style: "margin-top:0" }, `Генерація відео (v${v.version_number})`));
  card.appendChild(el("p", { class: "muted" }, "POC-режим: невелика роздільність і мало кадрів, щоб MacBook не перегрівався. Перша генерація може тривати кілька хвилин."));

  const motionInput = el("textarea", { placeholder: "Опишіть рух, напр. «біжить по газону, виляє хвостом»" });
  card.appendChild(el("div", { class: "form-row" }, [el("label", {}, "Опис руху"), motionInput]));

  const genBtn = el("button", { class: "btn btn-primary" }, "Згенерувати відео");
  const statusEl = el("span", { class: "muted", style: "margin-left:10px" });
  genBtn.addEventListener("click", async () => {
    if (!motionInput.value.trim()) { alert("Введіть опис руху"); return; }
    genBtn.disabled = true;
    statusEl.textContent = "Генерація відео... це може зайняти кілька хвилин на MacBook";
    try {
      await api(`/api/characters/${character.id}/generate-video`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ motion_prompt: motionInput.value.trim() }),
      });
      statusEl.textContent = "Готово";
      await loadDetail();
    } catch (e) {
      statusEl.textContent = "";
      alert("Помилка генерації відео: " + e.message);
    } finally {
      genBtn.disabled = false;
    }
  });
  card.appendChild(genBtn);
  card.appendChild(statusEl);
  body.appendChild(card);

  const galCard = el("div", { class: "card" });
  galCard.appendChild(el("h3", { style: "margin-top:0" }, "Усі відео персонажа"));
  galCard.appendChild(assetGrid(state.videoAssets, character.id));
  body.appendChild(galCard);
}

function renderNewTrait(body, character, versions) {
  if (!character.current_version_id) {
    body.appendChild(el("div", { class: "card" }, "Спершу підтвердіть початкову версію персонажа у вкладці «Огляд і памʼять» — нову рису можна додати лише поверх підтвердженого портрета."));
    return;
  }
  const card = el("div", { class: "card" });
  card.appendChild(el("h3", { style: "margin-top:0" }, "Додати нову рису (напр. зʼявився шрам)"));
  card.appendChild(el("p", { class: "muted" }, "Нова версія успадкує всі попередні референсні фото й опис — стара памʼять не втрачається, лише доповнюється."));

  const traitInput = el("input", { type: "text", placeholder: "напр. «зʼявився шрам на лівому вусі»" });
  const fileInput = el("input", { type: "file", multiple: "true", accept: "image/*" });
  card.appendChild(el("div", { class: "form-row" }, [el("label", {}, "Що змінилось"), traitInput]));
  card.appendChild(el("div", { class: "form-row" }, [el("label", {}, "Нові фото (де видно зміну)"), fileInput]));

  const btn = el("button", { class: "btn btn-primary" }, "Створити нову версію");
  btn.addEventListener("click", async () => {
    if (!traitInput.value.trim()) { alert("Опишіть, що змінилось"); return; }
    const fd = new FormData();
    fd.append("trait_note", traitInput.value.trim());
    for (const f of fileInput.files) fd.append("files", f);
    btn.disabled = true;
    try {
      await api(`/api/characters/${character.id}/new-version`, { method: "POST", body: fd });
      state.activeTab = "overview";
      await loadDetail();
    } catch (e) {
      alert("Помилка: " + e.message);
    } finally {
      btn.disabled = false;
    }
  });
  card.appendChild(btn);
  body.appendChild(card);
}

// ---------- create character modal ----------
function openNewCharacterModal() {
  const root = document.getElementById("modalRoot");
  const nameInput = el("input", { type: "text", placeholder: "напр. Рекс" });
  const descInput = el("textarea", { placeholder: "напр. рудий корги з білими лапками" });
  const fileInput = el("input", { type: "file", multiple: "true", accept: "image/*" });

  const close = () => { root.innerHTML = ""; };

  const submitBtn = el("button", { class: "btn btn-primary" }, "Створити");
  submitBtn.addEventListener("click", async () => {
    if (!nameInput.value.trim()) { alert("Вкажіть імʼя"); return; }
    if (!fileInput.files.length) { alert("Завантажте хоча б одне референсне фото"); return; }
    const fd = new FormData();
    fd.append("name", nameInput.value.trim());
    fd.append("base_description", descInput.value.trim());
    for (const f of fileInput.files) fd.append("files", f);
    submitBtn.disabled = true;
    try {
      const character = await api("/api/characters", { method: "POST", body: fd });
      close();
      await loadCharacters();
      await selectCharacter(character.id);
    } catch (e) {
      alert("Помилка створення: " + e.message);
      submitBtn.disabled = false;
    }
  });

  const modal = el("div", { class: "modal" }, [
    el("h3", {}, "Новий персонаж (собака)"),
    el("div", { class: "form-row" }, [el("label", {}, "Імʼя"), nameInput]),
    el("div", { class: "form-row" }, [el("label", {}, "Опис (порода, забарвлення...)"), descInput]),
    el("div", { class: "form-row" }, [el("label", {}, "Референсні фото (перша фотографія і, за бажанням, ще кілька)"), fileInput]),
    el("div", { style: "display:flex;gap:8px;justify-content:flex-end" }, [
      el("button", { class: "btn", onclick: close }, "Скасувати"),
      submitBtn,
    ]),
  ]);
  root.appendChild(el("div", { class: "modal-backdrop", onclick: (e) => { if (e.target === e.currentTarget) close(); } }, modal));
}

// ---------- init ----------
document.getElementById("newCharBtn").addEventListener("click", openNewCharacterModal);
refreshHealth();
setInterval(refreshHealth, 10000);
loadCharacters();
