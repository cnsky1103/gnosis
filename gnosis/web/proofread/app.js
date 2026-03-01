const state = {
  projects: [],
  project: "",
  lines: [],
  windowFlags: [],
  characters: [],
  keyMap: [],
  keyToName: new Map(),
  index: 0,
  dirty: new Set(),
  inflight: new Map(),
  saveTimer: null,
};

const contextRadius = 3;
const speakerWindowSize = 20;
const maxSpeakersInWindow = 2;

const els = {
  projectSelect: document.getElementById("projectSelect"),
  jumpInput: document.getElementById("jumpInput"),
  jumpBtn: document.getElementById("jumpBtn"),
  nextRedBtn: document.getElementById("nextRedBtn"),
  reloadBtn: document.getElementById("reloadBtn"),
  saveStatus: document.getElementById("saveStatus"),
  progressText: document.getElementById("progressText"),
  progressBar: document.getElementById("progressBar"),
  contextList: document.getElementById("contextList"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  lineMeta: document.getElementById("lineMeta"),
  textInput: document.getElementById("textInput"),
  speakerInput: document.getElementById("speakerInput"),
  emotionInput: document.getElementById("emotionInput"),
  typeInput: document.getElementById("typeInput"),
  hotkeyGrid: document.getElementById("hotkeyGrid"),
};

async function fetchJson(url, options = {}) {
  const resp = await fetch(url, options);
  const payload = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const message = payload && payload.error ? payload.error : `HTTP ${resp.status}`;
    throw new Error(message);
  }
  return payload;
}

function normalizeLine(raw) {
  const line = raw && typeof raw === "object" ? raw : {};
  return {
    ...line,
    text: line.text == null ? "" : String(line.text),
    speaker: line.speaker == null ? "" : String(line.speaker),
    emotion: line.emotion == null ? "" : String(line.emotion),
    type: line.type == null ? "" : String(line.type),
  };
}

function getProjectFromQuery() {
  const params = new URLSearchParams(window.location.search);
  return params.get("project") || "";
}

function isEditableTarget(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tagName = (target.tagName || "").toUpperCase();
  return tagName === "INPUT" || tagName === "TEXTAREA";
}

function setSaveStatus(kind, text) {
  els.saveStatus.classList.remove("idle", "saving", "saved", "error");
  els.saveStatus.classList.add(kind);
  els.saveStatus.textContent = text;
}

function persistIndex() {
  if (!state.project) return;
  localStorage.setItem(`proofread:index:${state.project}`, String(state.index));
}

function restoreIndex(total) {
  if (!state.project) return 0;
  const raw = localStorage.getItem(`proofread:index:${state.project}`);
  const value = Number.parseInt(raw || "0", 10);
  if (Number.isNaN(value)) return 0;
  if (value < 0) return 0;
  return Math.min(total - 1, value);
}

function renderProjectSelect() {
  els.projectSelect.innerHTML = "";
  for (const project of state.projects) {
    const option = document.createElement("option");
    option.value = project;
    option.textContent = project;
    option.selected = project === state.project;
    els.projectSelect.appendChild(option);
  }
}

function renderHotkeys() {
  els.hotkeyGrid.innerHTML = "";
  state.keyToName.clear();
  for (const item of state.keyMap) {
    state.keyToName.set(item.key.toLowerCase(), item.name);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "hotkey-chip";
    btn.innerHTML = `<span class="key">${item.key.toUpperCase()}</span>${escapeHtml(item.name)}`;
    btn.addEventListener("click", () => {
      void assignSpeakerAndAdvance(item.name);
    });
    els.hotkeyGrid.appendChild(btn);
  }
}

function normalizeSpeakerKey(value) {
  const speaker = value == null ? "" : String(value).trim();
  return speaker.toLowerCase();
}

function recomputeWindowFlags() {
  const total = state.lines.length;
  const redFlags = new Array(total).fill(false);
  const greenFlags = new Array(total).fill(false);

  for (let start = 0; start < total; start += 1) {
    const end = Math.min(total, start + speakerWindowSize);
    const speakers = new Set();
    for (let idx = start; idx < end; idx += 1) {
      const key = normalizeSpeakerKey(state.lines[idx]?.speaker);
      if (key) {
        speakers.add(key);
        if (speakers.size > maxSpeakersInWindow) {
          break;
        }
      }
    }

    const isRed = speakers.size > maxSpeakersInWindow;
    for (let idx = start; idx < end; idx += 1) {
      if (isRed) {
        redFlags[idx] = true;
      } else {
        greenFlags[idx] = true;
      }
    }
  }

  state.windowFlags = new Array(total);
  for (let idx = 0; idx < total; idx += 1) {
    if (redFlags[idx]) {
      state.windowFlags[idx] = "red";
    } else if (greenFlags[idx]) {
      state.windowFlags[idx] = "green";
    } else {
      state.windowFlags[idx] = "neutral";
    }
  }
}

function renderProgress() {
  const total = state.lines.length;
  if (total === 0) {
    els.progressText.textContent = "0 / 0";
    els.progressBar.style.width = "0%";
    return;
  }
  const current = state.index + 1;
  const reviewed = state.lines.reduce((acc, line) => {
    return acc + (line.speaker && line.speaker.trim() ? 1 : 0);
  }, 0);
  const pointerPercent = (current / total) * 100;
  els.progressText.textContent = `第 ${current} / ${total} 句 · 已标注说话者 ${reviewed} 句`;
  els.progressBar.style.width = `${pointerPercent.toFixed(2)}%`;
}

function renderEditor() {
  const line = state.lines[state.index];
  if (!line) {
    els.lineMeta.textContent = "空项目";
    els.textInput.value = "";
    els.speakerInput.value = "";
    els.emotionInput.value = "";
    els.typeInput.value = "";
    return;
  }

  els.lineMeta.textContent = `第 ${state.index + 1} 句`;
  if (document.activeElement !== els.textInput) {
    els.textInput.value = line.text;
  }
  if (document.activeElement !== els.speakerInput) {
    els.speakerInput.value = line.speaker;
  }
  if (document.activeElement !== els.emotionInput) {
    els.emotionInput.value = line.emotion;
  }
  if (document.activeElement !== els.typeInput) {
    els.typeInput.value = line.type || "";
  }
}

function renderContext() {
  els.contextList.innerHTML = "";
  for (let offset = -contextRadius; offset <= contextRadius; offset += 1) {
    const idx = state.index + offset;
    const line = idx >= 0 && idx < state.lines.length ? state.lines[idx] : null;
    const zone = idx >= 0 && idx < state.windowFlags.length ? state.windowFlags[idx] : "neutral";
    const block = document.createElement("div");
    block.className = "ctx-line";
    if (zone === "red") {
      block.classList.add("window-red");
    } else if (zone === "green") {
      block.classList.add("window-green");
    }
    if (offset === 0) {
      block.classList.add("current");
    } else {
      block.classList.add("muted");
    }

    if (!line) {
      block.innerHTML = `
        <div class="ctx-meta"><span class="ctx-speaker">-</span><span class="ctx-line-index">行号 --</span></div>
        <div class="ctx-text">（无内容）</div>
      `;
      els.contextList.appendChild(block);
      continue;
    }

    block.innerHTML = `
      <div class="ctx-meta"><span class="ctx-speaker">${escapeHtml(line.speaker || "未标注")}</span><span class="ctx-line-index">第 ${idx + 1} 句</span></div>
      <div class="ctx-text">${escapeHtml(line.text || "（空）")}</div>
    `;
    block.addEventListener("click", () => {
      void gotoIndex(idx);
    });
    els.contextList.appendChild(block);
  }
}

function renderAll() {
  renderProgress();
  renderEditor();
  renderContext();
}

function markDirty(index) {
  state.dirty.add(index);
  setSaveStatus("saving", "待保存");
}

function scheduleSave(index) {
  if (state.saveTimer) {
    clearTimeout(state.saveTimer);
  }
  state.saveTimer = window.setTimeout(() => {
    void saveLine(index);
  }, 220);
}

async function saveLine(index) {
  if (!state.project || index < 0 || index >= state.lines.length) return false;
  if (!state.dirty.has(index)) return true;
  if (state.inflight.has(index)) return state.inflight.get(index);

  const line = state.lines[index];
  setSaveStatus("saving", "保存中");
  const req = fetchJson(`/api/project/${encodeURIComponent(state.project)}/line/${index}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: line.text,
      speaker: line.speaker,
      emotion: line.emotion,
      type: line.type,
    }),
  })
    .then((result) => {
      if (state.lines[index]) {
        state.lines[index] = normalizeLine(result.line);
      }
      recomputeWindowFlags();
      state.dirty.delete(index);
      setSaveStatus("saved", "已保存");
      renderProgress();
      if (state.index === index) {
        renderEditor();
        renderContext();
      }
      return true;
    })
    .catch((err) => {
      setSaveStatus("error", "保存失败");
      console.error(err);
      return false;
    })
    .finally(() => {
      state.inflight.delete(index);
    });

  state.inflight.set(index, req);
  return req;
}

async function flushCurrentLine() {
  if (state.saveTimer) {
    clearTimeout(state.saveTimer);
    state.saveTimer = null;
  }
  return saveLine(state.index);
}

async function gotoIndex(index) {
  if (index < 0 || index >= state.lines.length) return;
  const ok = await flushCurrentLine();
  if (!ok) return;
  state.index = index;
  persistIndex();
  renderAll();
}

async function gotoNext() {
  if (state.index >= state.lines.length - 1) return;
  await gotoIndex(state.index + 1);
}

async function gotoPrev() {
  if (state.index <= 0) return;
  await gotoIndex(state.index - 1);
}

async function assignSpeakerAndAdvance(name) {
  const line = state.lines[state.index];
  if (!line) return;
  line.speaker = name;
  recomputeWindowFlags();
  markDirty(state.index);
  renderEditor();
  renderContext();
  const ok = await flushCurrentLine();
  if (!ok) return;
  if (state.index < state.lines.length - 1) {
    state.index += 1;
    persistIndex();
    renderAll();
  }
}

function updateCurrentField(field, value) {
  const line = state.lines[state.index];
  if (!line) return;
  if (line[field] === value) return;
  line[field] = value;
  markDirty(state.index);
  if (field === "speaker") {
    recomputeWindowFlags();
  }
  if (field === "text" || field === "speaker") {
    renderContext();
  }
  scheduleSave(state.index);
}

async function jumpToInputLine() {
  const total = state.lines.length;
  if (!total) return;
  const raw = Number.parseInt(els.jumpInput.value || "", 10);
  if (Number.isNaN(raw)) return;
  const nextIndex = Math.min(total - 1, Math.max(0, raw - 1));
  await gotoIndex(nextIndex);
}

async function gotoNextRedRegion() {
  for (let idx = state.index + 1; idx < state.windowFlags.length; idx += 1) {
    if (state.windowFlags[idx] !== "red") {
      continue;
    }
    if (idx > 0 && state.windowFlags[idx - 1] === "red") {
      continue;
    }
    await gotoIndex(idx);
    return;
  }
  setSaveStatus("saved", "no red ahead");
}

async function loadProject(projectName) {
  const payload = await fetchJson(`/api/project/${encodeURIComponent(projectName)}`);
  state.project = payload.project;
  state.lines = (payload.lines || []).map(normalizeLine);
  recomputeWindowFlags();
  state.characters = payload.characters || [];
  state.keyMap = payload.key_map || [];
  state.index = state.lines.length > 0 ? restoreIndex(state.lines.length) : 0;
  renderProjectSelect();
  renderHotkeys();
  renderAll();
  setSaveStatus("saved", "已加载");
}

async function loadProjects() {
  setSaveStatus("saving", "加载中");
  const payload = await fetchJson("/api/projects");
  state.projects = payload.projects || [];
  if (!state.projects.length) {
    throw new Error("没有可用项目");
  }
  const queryProject = getProjectFromQuery();
  const defaultProject = payload.default_project || state.projects[0];
  const initial = state.projects.includes(queryProject) ? queryProject : defaultProject;
  await loadProject(initial);
}

function bindEvents() {
  els.projectSelect.addEventListener("change", () => {
    const targetProject = els.projectSelect.value;
    void (async () => {
      if (targetProject === state.project) return;
      const ok = await flushCurrentLine();
      if (!ok) {
        els.projectSelect.value = state.project;
        return;
      }
      await loadProject(targetProject);
    })();
  });
  els.reloadBtn.addEventListener("click", () => {
    void (async () => {
      const ok = await flushCurrentLine();
      if (!ok) return;
      await loadProject(state.project);
    })();
  });
  els.jumpBtn.addEventListener("click", () => {
    void jumpToInputLine();
  });
  els.nextRedBtn.addEventListener("click", () => {
    void gotoNextRedRegion();
  });
  els.jumpInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      void jumpToInputLine();
    }
  });

  els.prevBtn.addEventListener("click", () => {
    void gotoPrev();
  });
  els.nextBtn.addEventListener("click", () => {
    void gotoNext();
  });

  els.textInput.addEventListener("input", () => {
    updateCurrentField("text", els.textInput.value);
  });
  els.speakerInput.addEventListener("input", () => {
    updateCurrentField("speaker", els.speakerInput.value);
  });
  els.emotionInput.addEventListener("input", () => {
    updateCurrentField("emotion", els.emotionInput.value);
  });
  els.typeInput.addEventListener("input", () => {
    updateCurrentField("type", els.typeInput.value);
  });

  document.addEventListener("keydown", (event) => {
    if (event.isComposing) return;
    const target = event.target;
    if (event.key === "Enter") {
      if (target === els.jumpInput) return;
      event.preventDefault();
      if (event.shiftKey) {
        void gotoPrev();
      } else {
        void gotoNext();
      }
      return;
    }

    if (isEditableTarget(target)) return;

    if (event.key === "ArrowLeft") {
      event.preventDefault();
      void gotoPrev();
      return;
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      void gotoNext();
      return;
    }

    const lower = (event.key || "").toLowerCase();
    const name = state.keyToName.get(lower);
    if (name) {
      event.preventDefault();
      void assignSpeakerAndAdvance(name);
    }
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function main() {
  bindEvents();
  try {
    await loadProjects();
  } catch (err) {
    console.error(err);
    setSaveStatus("error", "加载失败");
    els.contextList.innerHTML = `<div class="ctx-line current"><div class="ctx-text">${escapeHtml(err.message || "加载失败")}</div></div>`;
  }
}

void main();
