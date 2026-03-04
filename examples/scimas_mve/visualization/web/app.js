const stageOrder = ["prepare", "profile", "literature", "read", "hypothesize", "experiment", "review", "write", "replicate", "other"];

const state = {
  payload: null,
  episode: 0,
  mode: "live",
  replayPercent: 100,
  search: "",
  runSearch: "",
  typeFilter: new Set(),
  errFilter: new Set(),
  agent: "",
  selected: null,
  tab: "overview",
  filePath: "",
  snapshots: [],
  inspectorCollapsed: false,
  gpuHistory: [],
  mainSplitRatio: 0.68,
  sidebarWidth: 300,
  inspectorWidth: 460,
  inspectorScrollByKey: {},
  lastInspectorKey: "",
  ioScope: { episode: "current", agent: "", action: "" },
  ioListByTab: {},
  ioDetailById: {},
  ioLoadingByTab: {},
  ioLoadingByDetail: {},
  ioLastSeenTs: {},
  ioExpandedBlocks: {},
  ioSelectionByTab: {},
};

function esc(s) {
  return String(s == null ? "" : s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function toLocal(ts) {
  if (!ts) return "-";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return String(ts);
  }
}

function fmtNum(v, digits = 3) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(digits);
}

function ioSetExpanded(key, value) {
  state.ioExpandedBlocks[key] = !!value;
}

function ioIsExpanded(key, fallback = false) {
  if (Object.prototype.hasOwnProperty.call(state.ioExpandedBlocks, key)) {
    return !!state.ioExpandedBlocks[key];
  }
  return !!fallback;
}

function ioScopeParams() {
  const params = new URLSearchParams();
  const selected = state.selected?.value || {};
  const episodeScope = String(state.ioScope.episode || "current").toLowerCase();
  if (episodeScope === "all") {
    params.set("episode", "all");
  } else if (state.episode > 0) {
    params.set("episode", String(state.episode));
  } else {
    params.set("episode", "current");
  }
  const agent = String(selected.owner || selected.agent_id || state.ioScope.agent || "").trim();
  const action = String(selected.task_type || selected.action || state.ioScope.action || "").trim();
  const taskId = String(selected.task_id || "").trim();
  const runId = String(selected.run_id || "").trim();
  if (agent) params.set("agent", agent);
  if (action) params.set("action", action);
  if (taskId) params.set("task_id", taskId);
  if (runId) params.set("run_id", runId);
  params.set("limit", "80");
  return params;
}

function ioEndpointForTab(tab) {
  if (tab === "llm_io") return "/api/audit/llm";
  if (tab === "rag_io") return "/api/audit/rag";
  if (tab === "retrieve") return "/api/retrieve/pipeline";
  return "";
}

function ioKindForTab(tab) {
  if (tab === "llm_io") return "llm_io";
  if (tab === "rag_io") return "rag_io";
  if (tab === "retrieve") return "retrieve_pipeline";
  return "";
}

function ioCacheKey(tab) {
  const params = ioScopeParams();
  return `${tab}|${params.toString()}`;
}

async function fetchIoList(tab) {
  const endpoint = ioEndpointForTab(tab);
  if (!endpoint) return;
  const key = ioCacheKey(tab);
  if (state.ioLoadingByTab[key]) return;
  state.ioLoadingByTab[key] = true;
  try {
    const params = ioScopeParams();
    const res = await fetch(`${endpoint}?${params.toString()}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`io_list_http_${res.status}`);
    const data = await res.json();
    state.ioListByTab[key] = data;
    const rows = data.rows || [];
    const prevTs = String(state.ioLastSeenTs[tab] || "");
    if (rows.length && !prevTs) state.ioLastSeenTs[tab] = String(rows[0].ts || "");
    if (!state.ioSelectionByTab[tab] && rows.length) state.ioSelectionByTab[tab] = String(rows[0].id || "");
  } catch (e) {
    state.ioListByTab[key] = { ok: false, error: String(e), rows: [] };
  } finally {
    state.ioLoadingByTab[key] = false;
    renderInspector();
  }
}

async function fetchIoDetail(kind, id) {
  if (!kind || !id) return;
  const key = `${kind}:${id}`;
  if (state.ioDetailById[key] || state.ioLoadingByDetail[key]) return;
  state.ioLoadingByDetail[key] = true;
  try {
    const res = await fetch(`/api/audit/detail?kind=${encodeURIComponent(kind)}&id=${encodeURIComponent(id)}`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`io_detail_http_${res.status}`);
    const data = await res.json();
    state.ioDetailById[key] = data;
  } catch (e) {
    state.ioDetailById[key] = { ok: false, error: String(e), kind, record: null };
  } finally {
    state.ioLoadingByDetail[key] = false;
    renderInspector();
  }
}

function renderCollapsibleText(title, text, key, limit = 300) {
  const raw = String(text || "");
  const short = raw.length > limit ? `${raw.slice(0, limit)}...` : raw;
  const expanded = ioIsExpanded(key, false);
  return `
    <div class="io-block">
      <div class="io-block-head">
        <div class="io-block-title">${esc(title)}</div>
        <button class="io-mini-btn" data-io-toggle="${esc(key)}">${expanded ? "Collapse" : "Expand"}</button>
      </div>
      <pre>${esc(expanded ? raw : short)}</pre>
    </div>
  `;
}

function renderJsonTree(obj, keyPrefix = "json", depth = 0, maxDepth = 4) {
  if (depth > maxDepth) return `<span class="muted">...</span>`;
  if (obj == null) return `<span class="json-null">null</span>`;
  if (typeof obj !== "object") return `<span class="json-leaf">${esc(String(obj))}</span>`;
  if (Array.isArray(obj)) {
    const items = obj.slice(0, 40).map((v, i) => `<li>${renderJsonTree(v, `${keyPrefix}.${i}`, depth + 1, maxDepth)}</li>`).join("");
    return `<details class="json-tree" open><summary>[${obj.length}]</summary><ul>${items}</ul></details>`;
  }
  const entries = Object.entries(obj).slice(0, 80);
  const body = entries
    .map(([k, v]) => `<li><span class="json-key">${esc(k)}</span>: ${renderJsonTree(v, `${keyPrefix}.${k}`, depth + 1, maxDepth)}</li>`)
    .join("");
  return `<details class="json-tree" open><summary>{${entries.length}}</summary><ul>${body}</ul></details>`;
}

function bindIoToggles(container) {
  container.querySelectorAll("[data-io-toggle]").forEach((btn) => {
    btn.onclick = () => {
      const key = btn.dataset.ioToggle || "";
      ioSetExpanded(key, !ioIsExpanded(key));
      renderInspector();
    };
  });
}

function setHealthDot(level) {
  const dot = document.getElementById("health-dot");
  dot.style.background = level === "red" ? "var(--err)" : level === "yellow" ? "var(--warn)" : "var(--ok)";
}

function replayCutoffTs(events) {
  if (state.mode !== "replay") return null;
  const tsList = events.map((e) => Date.parse(e.ts || "")).filter(Number.isFinite).sort((a, b) => a - b);
  if (!tsList.length) return null;
  const idx = Math.max(0, Math.min(tsList.length - 1, Math.floor((state.replayPercent / 100) * (tsList.length - 1))));
  return tsList[idx];
}

function matchFilters(obj, cutoff) {
  const ep = Number(obj.episode_id || 0);
  if (state.episode > 0 && ep !== state.episode) return false;

  if (cutoff != null) {
    const t = Date.parse(obj.ts || obj.last_update || "");
    if (Number.isFinite(t) && t > cutoff) return false;
  }

  if (state.agent && String(obj.owner || obj.agent_id || "").toLowerCase() !== state.agent.toLowerCase()) return false;

  if (state.typeFilter.size > 0) {
    const taskType = String(obj.task_type || "");
    if (!state.typeFilter.has(taskType)) return false;
  }

  const text = JSON.stringify(obj).toLowerCase();
  if (state.search && !text.includes(state.search)) return false;

  if (state.runSearch) {
    const runText = `${obj.run_id || ""} ${obj.task_id || ""} ${obj.owner || ""}`.toLowerCase();
    if (!runText.includes(state.runSearch)) return false;
  }

  if (state.errFilter.size > 0) {
    let ok = false;
    for (const tag of state.errFilter) {
      if (text.includes(tag)) {
        ok = true;
        break;
      }
    }
    if (!ok) return false;
  }
  return true;
}

function estimateConfidence(task) {
  const result = task.result || {};
  const actionStatus = String(result.action_status || "").toLowerCase();
  const actionData = result.action_data || {};
  if (typeof actionData.confidence === "number") {
    return Math.max(0, Math.min(1, actionData.confidence));
  }
  let score = actionStatus === "success" ? 0.65 : 0.35;
  const rationale = (((actionData.plan_spec || {}).rationale) || []);
  score += Math.min(0.2, rationale.length * 0.05);
  if ((task.task_type || "") === "experiment" && task.run_id) score += 0.1;
  if ((task.release_reason || "").includes("inner_action_failed")) score -= 0.25;
  return Math.max(0.05, Math.min(0.95, score));
}

function classifyTTL(task) {
  const ttl = Number(task.lease_ttl || 0);
  if (ttl <= 0) return { remainRatio: 0, cls: "ttl-red" };
  const started = Number(task.started_tick ?? -1);
  const hb = Number(task.last_heartbeat_tick ?? started);
  const elapsed = started >= 0 && hb >= 0 ? Math.max(0, hb - started) : 0;
  const remain = Math.max(0, ttl - elapsed);
  const ratio = Math.max(0, Math.min(1, remain / ttl));
  const cls = ratio > 0.66 ? "ttl-green" : ratio > 0.33 ? "ttl-yellow" : "ttl-red";
  return { remainRatio: ratio, cls };
}

function inferBridgeStatus(task, runMap) {
  const rid = String(task.run_id || "");
  if (!rid) return "-";
  const run = runMap[rid];
  if (!run) return "-";
  return run.data_bridge_used ? "bridge:on" : "bridge:off";
}

function buildFailureHints(stderrText) {
  const s = String(stderrText || "");
  const hints = [];
  if (/ModuleNotFoundError:\s*No module named '([^']+)'/i.test(s)) {
    const m = s.match(/ModuleNotFoundError:\s*No module named '([^']+)'/i);
    hints.push(`Missing dependency: ${m ? m[1] : "unknown"}. 建议在 runtime 镜像里安装该包。`);
  }
  if (/FileNotFoundError/i.test(s)) {
    hints.push("输入文件路径不存在。检查 data bridge 输出路径与任务脚本期望路径。");
  }
  if (/SyntaxError/i.test(s)) {
    hints.push("代码语法错误。优先修复触发行号附近的拼接/转义问题。");
  }
  if (/Killed/i.test(s)) {
    hints.push("进程被系统杀死（常见于内存上限触发）。建议降低 batch/模型复杂度或提升容器内存。");
  }
  if (/AttributeError: 'Column' object has no attribute 'tolist'/i.test(s)) {
    hints.push("HuggingFace datasets 的 Column 不支持 tolist()；应先转 pandas 或用 list(column)。");
  }
  if (!hints.length) hints.push("未命中规则化归因，请结合 traceback 行号做手动定位。");
  return hints;
}

async function fetchSnapshot() {
  const res = await fetch("/api/snapshot", { cache: "no-store" });
  if (!res.ok) throw new Error(`snapshot_http_${res.status}`);
  const payload = await res.json();
  applyPayload(payload);
}

function connectSSE() {
  const es = new EventSource("/api/events");
  es.addEventListener("snapshot", (evt) => {
    try {
      applyPayload(JSON.parse(evt.data || "{}"));
    } catch {
      // ignore parse noise
    }
  });
  es.onerror = () => {
    setTimeout(() => {
      try {
        es.close();
      } catch {}
      connectSSE();
    }, 2000);
  };
}

function applyPayload(payload) {
  state.payload = payload || {};
  const gpu = (((state.payload || {}).server || {}).gpu || {});
  if (gpu.available) {
    state.gpuHistory.push(Number(gpu.memory_ratio || 0));
    if (state.gpuHistory.length > 24) state.gpuHistory.shift();
  }
  renderAll();
}

function bindControls() {
  document.getElementById("search-input").oninput = (e) => {
    state.search = e.target.value.trim().toLowerCase();
    renderAll();
  };

  document.getElementById("run-search").oninput = (e) => {
    state.runSearch = e.target.value.trim().toLowerCase();
    renderAll();
  };

  document.getElementById("episode-select").onchange = (e) => {
    state.episode = Number(e.target.value || 0);
    renderAll();
  };

  document.getElementById("mode-select").onchange = (e) => {
    state.mode = e.target.value;
    document.getElementById("replay-slider").classList.toggle("hidden", state.mode !== "replay");
    renderAll();
  };

  document.getElementById("replay-slider").oninput = (e) => {
    state.replayPercent = Number(e.target.value || 100);
    renderAll();
  };

  document.getElementById("snapshot-btn").onclick = () => {
    const snap = {
      ts: new Date().toISOString(),
      mode: state.mode,
      episode: state.episode,
      selected: state.selected,
      alerts: (((state.payload || {}).meta || {}).alerts || []),
      top_release_reason: (((state.payload || {}).meta || {}).top_release_reason || "-"),
      timing: (((state.payload || {}).taskboard || {}).timing || {}),
    };
    state.snapshots.push(snap);
    const blob = new Blob([JSON.stringify(snap, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scimas_snapshot_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  document.getElementById("export-btn").onclick = () => {
    const out = {
      payload: state.payload || {},
      snapshots: state.snapshots,
    };
    const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scimas_console_export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  document.getElementById("inspector-toggle").onclick = () => {
    state.inspectorCollapsed = !state.inspectorCollapsed;
    const root = document.getElementById("layout-root");
    root.classList.toggle("inspector-collapsed", state.inspectorCollapsed);
    applyColumnWidths();
    document.getElementById("inspector-toggle").textContent = state.inspectorCollapsed ? "Show Inspector" : "Hide Inspector";
  };
}

function setupEpisodeSelect(events, runs) {
  const select = document.getElementById("episode-select");
  const eps = new Set([0]);
  events.forEach((e) => {
    const ep = Number(e.episode_id || 0);
    if (ep > 0) eps.add(ep);
  });
  runs.forEach((r) => {
    const ep = Number(r.episode_id || 0);
    if (ep > 0) eps.add(ep);
  });
  const sorted = [...eps].sort((a, b) => a - b);
  select.innerHTML = sorted.map((ep) => `<option value="${ep}">${ep === 0 ? "All Episodes" : `Episode ${ep}`}</option>`).join("");
  if (!sorted.includes(state.episode)) state.episode = 0;
  select.value = String(state.episode);
}

function renderHeader(meta, timing, teamSummary, serverInfo) {
  setHealthDot(meta.health || "green");
  document.getElementById("health-text").textContent = `health: ${meta.health || "green"}`;
  document.getElementById("alert-chip").textContent = `alerts: ${meta.alert_count || 0}`;
  document.getElementById("top-release-chip").textContent = `top release: ${meta.top_release_reason || "-"}`;
  document.getElementById("alert-count-mini").textContent = `${meta.alert_count || 0} active`;

  const avgCs = Number(timing.avg_claim_to_start_ticks || 0).toFixed(1);
  const avgSc = Number(timing.avg_start_to_complete_ticks || 0).toFixed(1);
  const episodesCount = Number(teamSummary.episodes_count || meta.episodes_count || 0);
  const headerHint = `episodes: ${episodesCount} | C→S:${avgCs} | S→C:${avgSc}`;
  document.title = `SCIMAS ResearchOps Console · ${headerHint}`;

  const now = serverInfo.ts || new Date().toISOString();
  document.getElementById("sim-clock-chip").textContent = `clock: ${toLocal(now)}`;

  const runCount = Number((((state.payload || {}).runs || {}).runs || []).length);
  const tokenCost = (runCount * 0.0025).toFixed(4);
  document.getElementById("token-chip").textContent = `token cost: $${tokenCost}`;

  const gpu = serverInfo.gpu || {};
  if (gpu.available) {
    const memPct = Math.round((Number(gpu.memory_ratio || 0)) * 100);
    document.getElementById("gpu-chip").textContent = `gpu: ${Math.round(gpu.util_percent || 0)}% | mem ${memPct}%`;
  } else {
    document.getElementById("gpu-chip").textContent = "gpu: n/a";
  }
}

function renderAlertCenter(alerts) {
  const list = document.getElementById("alert-list");
  if (!alerts.length) {
    list.innerHTML = `<div class="muted">No active alert</div>`;
    return;
  }
  list.innerHTML = alerts.map((a) => {
    const level = String(a.level || "warn");
    return `<div class="alert-item ${esc(level)}"><b>${esc(a.key)}</b> <span class="muted">${esc(a.value)}</span></div>`;
  }).join("");
}

function renderTypeFilters(tasks) {
  const typeWrap = document.getElementById("type-filters");
  const types = [...new Set(tasks.map((t) => String(t.task_type || "")).filter(Boolean))].sort();
  typeWrap.innerHTML = types.map((t) => `<button class="chip ${state.typeFilter.has(t) ? "on" : ""}" data-type="${esc(t)}">${esc(t)}</button>`).join("");
  typeWrap.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      const t = btn.dataset.type || "";
      if (state.typeFilter.has(t)) state.typeFilter.delete(t);
      else state.typeFilter.add(t);
      renderAll();
    };
  });
}

function renderErrorFilters() {
  const errWrap = document.getElementById("error-filters");
  const errs = ["lease_expired", "inner_action_failed", "timeout", "oom", "json_parse"];
  errWrap.innerHTML = errs.map((e) => `<button class="chip ${state.errFilter.has(e) ? "on" : ""}" data-err="${e}">${e}</button>`).join("");
  errWrap.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      const e = btn.dataset.err || "";
      if (state.errFilter.has(e)) state.errFilter.delete(e);
      else state.errFilter.add(e);
      renderAll();
    };
  });
}

function renderAgents(agents) {
  const box = document.getElementById("agent-list");
  document.getElementById("agent-count").textContent = `${agents.length} agents`;
  box.innerHTML = agents.map((a) => {
    const active = state.agent === a.agent_id ? "active" : "";
    const statusClass = a.status === "running" ? "status-ok" : a.status === "error" ? "status-err" : "status-warn";
    return `<div class="agent-item ${active}" data-agent="${esc(a.agent_id)}"><div><span class="status-dot ${statusClass}"></span>${esc(a.agent_id)}</div><div class="muted">${a.claimed}/${a.completed}/${a.released}</div></div>`;
  }).join("");

  box.querySelectorAll(".agent-item").forEach((el) => {
    el.onclick = () => {
      const aid = el.dataset.agent || "";
      state.agent = state.agent === aid ? "" : aid;
      renderAll();
    };
  });
}

function computeKnowledgeLineage(selected, events) {
  if (!selected || selected.type !== "event") return new Set();
  const v = selected.value || {};
  if (String(v.stage || "") !== "experiment") return new Set();
  const ep = Number(v.episode_id || 0);
  const owner = String(v.owner || "");
  const sid = new Set();
  events.forEach((e) => {
    if (Number(e.episode_id || 0) !== ep) return;
    if (owner && String(e.owner || "") !== owner) return;
    const stage = String(e.stage || "");
    if (["read", "literature", "hypothesize", "profile", "prepare", "experiment"].includes(stage)) sid.add(e.id);
  });
  return sid;
}

function drawFlowGraph() {
  const canvas = document.getElementById("flow-canvas");
  const shell = document.querySelector(".timeline-shell");
  const timeline = document.getElementById("timeline");
  if (!canvas || !timeline || !shell) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const graphWidth = Math.max(timeline.scrollWidth, shell.clientWidth);
  const graphHeight = Math.max(timeline.scrollHeight, shell.clientHeight);
  canvas.width = Math.max(1, Math.floor(graphWidth * dpr));
  canvas.height = Math.max(1, Math.floor(graphHeight * dpr));
  canvas.style.width = `${graphWidth}px`;
  canvas.style.height = `${graphHeight}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, graphWidth, graphHeight);

  const heads = [...timeline.querySelectorAll(".lane")].map((lane) => ({
    x: lane.offsetLeft + lane.offsetWidth / 2,
    y: lane.offsetTop + 38,
  }));

  ctx.lineWidth = 1.4;
  ctx.strokeStyle = "rgba(76, 201, 240, 0.22)";
  for (let i = 0; i < heads.length - 1; i++) {
    const a = heads[i];
    const b = heads[i + 1];
    ctx.beginPath();
    const mx = (a.x + b.x) / 2;
    ctx.moveTo(a.x, a.y);
    ctx.bezierCurveTo(mx, a.y - 18, mx, b.y + 18, b.x, b.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.fillStyle = "rgba(76, 201, 240, 0.45)";
    ctx.arc(a.x + (b.x - a.x) * 0.65, a.y + (b.y - a.y) * 0.05, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }
}

function refreshTimelineCanvas() {
  window.requestAnimationFrame(() => {
    drawFlowGraph();
  });
}

function applyMainSplitRatio(ratio) {
  const main = document.querySelector(".main");
  if (!main) return;

  const splitterRowPx = 10;
  const minTop = 360;
  const minBottom = 180;
  const total = Math.max(0, main.clientHeight - splitterRowPx - 24);
  if (total <= minTop + minBottom) return;

  const safeRatio = Math.max(0.2, Math.min(0.85, Number(ratio) || 0.68));
  let top = Math.round(total * safeRatio);
  top = Math.max(minTop, Math.min(total - minBottom, top));
  const finalRatio = top / total;

  state.mainSplitRatio = finalRatio;
  main.style.gridTemplateRows = `${top}px ${splitterRowPx}px minmax(${minBottom}px, 1fr)`;
  localStorage.setItem("scimas.mainSplitRatio", String(finalRatio));
  refreshTimelineCanvas();
}

function applyColumnWidths() {
  const root = document.getElementById("layout-root");
  if (!root) return;
  root.style.setProperty("--sidebar-w", `${Math.round(state.sidebarWidth)}px`);
  root.style.setProperty("--inspector-w", `${Math.round(state.inspectorWidth)}px`);
}

function bindColumnSplitters() {
  const root = document.getElementById("layout-root");
  const left = document.getElementById("splitter-left");
  const right = document.getElementById("splitter-right");
  if (!root || !left || !right) return;

  const savedLeft = Number(localStorage.getItem("scimas.sidebarWidth"));
  const savedRight = Number(localStorage.getItem("scimas.inspectorWidth"));
  if (Number.isFinite(savedLeft)) state.sidebarWidth = savedLeft;
  if (Number.isFinite(savedRight)) state.inspectorWidth = savedRight;
  applyColumnWidths();

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const minMain = () => (window.innerWidth <= 1440 ? 560 : 640);

  let mode = "";
  const onMove = (clientX) => {
    const rect = root.getBoundingClientRect();
    const sep = state.inspectorCollapsed ? 10 : 20;
    if (mode === "left") {
      const maxLeft = rect.width - sep - state.inspectorWidth - minMain();
      state.sidebarWidth = clamp(clientX - rect.left, 220, Math.max(260, maxLeft));
      localStorage.setItem("scimas.sidebarWidth", String(state.sidebarWidth));
      applyColumnWidths();
      refreshTimelineCanvas();
      return;
    }
    if (mode === "right") {
      if (state.inspectorCollapsed) return;
      const maxRight = rect.width - sep - state.sidebarWidth - minMain();
      state.inspectorWidth = clamp(rect.right - clientX, 320, Math.max(360, maxRight));
      localStorage.setItem("scimas.inspectorWidth", String(state.inspectorWidth));
      applyColumnWidths();
      refreshTimelineCanvas();
    }
  };

  const down = (side) => (evt) => {
    mode = side;
    document.body.style.cursor = "col-resize";
    evt.preventDefault();
  };
  left.addEventListener("mousedown", down("left"));
  right.addEventListener("mousedown", down("right"));

  window.addEventListener("mousemove", (evt) => {
    if (!mode) return;
    onMove(evt.clientX);
  });
  window.addEventListener("mouseup", () => {
    if (!mode) return;
    mode = "";
    document.body.style.cursor = "";
  });

  left.addEventListener("keydown", (evt) => {
    const step = evt.key === "ArrowLeft" ? -16 : evt.key === "ArrowRight" ? 16 : 0;
    if (!step) return;
    evt.preventDefault();
    state.sidebarWidth = clamp(state.sidebarWidth + step, 220, 560);
    localStorage.setItem("scimas.sidebarWidth", String(state.sidebarWidth));
    applyColumnWidths();
    refreshTimelineCanvas();
  });
  right.addEventListener("keydown", (evt) => {
    if (state.inspectorCollapsed) return;
    const step = evt.key === "ArrowLeft" ? 16 : evt.key === "ArrowRight" ? -16 : 0;
    if (!step) return;
    evt.preventDefault();
    state.inspectorWidth = clamp(state.inspectorWidth + step, 320, 760);
    localStorage.setItem("scimas.inspectorWidth", String(state.inspectorWidth));
    applyColumnWidths();
    refreshTimelineCanvas();
  });
}

function bindMainSplitter() {
  const main = document.querySelector(".main");
  const splitter = document.getElementById("main-splitter");
  if (!main || !splitter) return;

  const cached = Number(localStorage.getItem("scimas.mainSplitRatio"));
  if (Number.isFinite(cached)) {
    state.mainSplitRatio = Math.max(0.2, Math.min(0.85, cached));
  }
  applyMainSplitRatio(state.mainSplitRatio);

  let dragging = false;
  const onMove = (clientY) => {
    const rect = main.getBoundingClientRect();
    const splitterRowPx = 10;
    const total = Math.max(0, rect.height - splitterRowPx - 24);
    if (total <= 0) return;
    const y = clientY - rect.top;
    const ratio = y / total;
    applyMainSplitRatio(ratio);
  };

  splitter.addEventListener("mousedown", (evt) => {
    dragging = true;
    document.body.style.cursor = "row-resize";
    evt.preventDefault();
  });

  window.addEventListener("mousemove", (evt) => {
    if (!dragging) return;
    onMove(evt.clientY);
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = "";
  });

  splitter.addEventListener("keydown", (evt) => {
    const delta = evt.key === "ArrowUp" ? -0.03 : evt.key === "ArrowDown" ? 0.03 : 0;
    if (!delta) return;
    evt.preventDefault();
    applyMainSplitRatio(state.mainSplitRatio + delta);
  });
}

function renderTimeline(events, cutoff) {
  const lanes = Object.fromEntries(stageOrder.map((s) => [s, []]));
  events.filter((e) => matchFilters(e, cutoff)).forEach((ev) => {
    const s = stageOrder.includes(ev.stage) ? ev.stage : "other";
    lanes[s].push(ev);
  });
  for (const s of stageOrder) lanes[s] = lanes[s].slice(-24);

  const highlightIds = computeKnowledgeLineage(state.selected, events);

  const wrap = document.getElementById("timeline");
  wrap.innerHTML = stageOrder.map((stage) => {
    const items = lanes[stage];
    const cards = items.map((ev) => {
      const cls = [
        ev.event === "complete" ? "success" : "",
        ev.event === "release" ? "release fail" : "",
        ev.event === "claim" ? "running" : "",
        highlightIds.has(ev.id) ? "highlight" : "",
      ].join(" ");
      return `<article class="event-card ${cls}" data-eid="${esc(ev.id)}">
        <div><b>${esc(ev.task_type || stage)}</b></div>
        <div class="event-meta">${esc(ev.task_id)} · ${esc(ev.owner || "-")}</div>
        <div class="event-meta">${esc(ev.event)} · ${toLocal(ev.ts)}</div>
        <div class="event-meta">${esc(ev.run_id || "")} C→S:${esc(ev.claim_to_start_ticks ?? "-")} S→C:${esc(ev.start_to_complete_ticks ?? "-")}</div>
      </article>`;
    }).join("");

    return `<section class="lane"><div class="lane-head">${stage} <span class="muted">${items.length}</span></div><div class="lane-body">${cards}</div></section>`;
  }).join("");

  wrap.querySelectorAll(".event-card").forEach((card) => {
    card.onclick = () => {
      const eid = card.dataset.eid;
      const ev = events.find((x) => x.id === eid);
      state.selected = { type: "event", value: ev };
      state.tab = "overview";
      renderInspector();
      renderTimeline(events, cutoff);
    };
  });

  drawFlowGraph();
}

function artifactBadges(task) {
  const bits = [];
  const resultText = JSON.stringify(task.result || {}).toLowerCase();
  if (resultText.includes("data_card")) bits.push("data");
  if (resultText.includes("method_card")) bits.push("method");
  if (resultText.includes("evidence")) bits.push("evidence");
  if (task.task_type === "write") bits.push("paper");
  if (task.task_type === "replicate") bits.push("repl");
  if (task.run_id) bits.push("run");
  return bits.join("|") || "-";
}

function avg(arr) {
  if (!arr.length) return "-";
  return (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2);
}

function renderTaskboard(tasks, cutoff, runMap) {
  const filtered = tasks.filter((t) => matchFilters(t, cutoff));
  const counts = { open: 0, claimed: 0, running: 0, completed: 0, released: 0 };
  filtered.forEach((r) => {
    const k = String(r.state || "unknown");
    counts[k] = (counts[k] || 0) + 1;
  });

  const c2s = filtered.map((x) => Number(x.claim_to_start_ticks)).filter((x) => Number.isFinite(x) && x >= 0);
  const s2c = filtered.map((x) => Number(x.start_to_complete_ticks)).filter((x) => Number.isFinite(x) && x >= 0);

  document.getElementById("task-pills").innerHTML = [
    `open: ${counts.open || 0}`,
    `claimed: ${counts.claimed || 0}`,
    `running: ${counts.running || 0}`,
    `completed: ${counts.completed || 0}`,
    `avg C→S: ${avg(c2s)}`,
    `avg S→C: ${avg(s2c)}`,
  ].map((s) => `<span class="stat-pill">${s}</span>`).join("");

  document.getElementById("task-summary").textContent = `rows: ${filtered.length}`;

  const tbody = document.getElementById("task-tbody");
  tbody.innerHTML = filtered.map((t) => {
    const selected = state.selected && state.selected.type === "task" && state.selected.value?.task_id === t.task_id ? "selected" : "";
    const deps = (t.depends_on || []).slice(0, 2).join("→") || "-";
    const conf = estimateConfidence(t);
    const ttl = classifyTTL(t);
    const bridge = inferBridgeStatus(t, runMap);
    return `<tr class="${selected}" data-task="${esc(t.task_id)}">
      <td>${esc(t.task_id)}</td>
      <td>${esc(t.task_type || "")}</td>
      <td>${esc(t.state || "")}</td>
      <td>${esc(t.owner || "-")}</td>
      <td><div>${esc(t.lease_ttl || 0)}/${esc(t.heartbeat || 0)}</div><div class="ttl-wrap"><div class="ttl-bar ${ttl.cls}" style="width:${Math.round(ttl.remainRatio * 100)}%"></div></div></td>
      <td>${esc(deps)}</td>
      <td>${esc(t.claim_to_start_ticks ?? "-")}</td>
      <td>${esc(t.start_to_complete_ticks ?? "-")}</td>
      <td>${Math.round(conf * 100)}%</td>
      <td>${esc(toLocal(t.last_update))}</td>
      <td>${esc(t.release_reason || "-")}</td>
      <td>${esc(artifactBadges(t))}</td>
      <td>${esc(bridge)}</td>
      <td>${esc(t.run_id || "-")}</td>
    </tr>`;
  }).join("");

  tbody.querySelectorAll("tr").forEach((row) => {
    row.onclick = () => {
      const id = row.dataset.task;
      const task = tasks.find((x) => x.task_id === id);
      state.selected = { type: "task", value: task };
      state.tab = "overview";
      renderAll();
    };
  });
}

function runForSelection(runs) {
  if (!state.selected) return null;
  const v = state.selected.value || {};
  if (state.selected.type === "run") return v;
  let runId = String(v.run_id || "");
  if (!runId && v.task_id) {
    const payload = state.payload || {};
    const tasks = ((payload.taskboard || {}).task_snapshot || []);
    const t = tasks.find((x) => String(x.task_id || "") === String(v.task_id || ""));
    runId = String((t || {}).run_id || "");
  }
  if (!runId) return null;

  return runs.find((r) => String(r.run_id) === runId && (state.episode <= 0 || Number(r.episode_id) === state.episode)) ||
    runs.find((r) => String(r.run_id) === runId) ||
    null;
}

function taskForSelection(tasks) {
  if (!state.selected) return null;
  const v = state.selected.value || {};
  if (state.selected.type === "task") return v;
  if (!v.task_id) return null;
  return tasks.find((t) => String(t.task_id || "") === String(v.task_id || "")) || null;
}

function ownerEpisodeSummary(events, owner, episodeId) {
  const rows = events.filter(
    (e) =>
      String(e.owner || "") === String(owner || "") &&
      Number(e.episode_id || 0) === Number(episodeId || 0)
  );
  const c = { claim: 0, complete: 0, release: 0, running: 0 };
  rows.forEach((r) => {
    const ev = String(r.event || "");
    if (ev in c) c[ev] += 1;
  });
  return { count: rows.length, ...c };
}

function relatedRuns(runs, selected, run) {
  if (run) return [run];
  const v = selected || {};
  const ep = Number(v.episode_id || 0);
  if (ep > 0) {
    return runs.filter((r) => Number(r.episode_id || 0) === ep).slice(-10);
  }
  return runs.slice(-10);
}

function renderInspector() {
  const payload = state.payload || {};
  const tb = payload.taskboard || {};
  const tasks = tb.task_snapshot || [];
  const events = tb.events || [];
  const tbArtifacts = tb.artifacts_index || [];
  const codeLoopsAll = payload.code_loops || [];
  const gatesAll = payload.precondition_gates || [];
  const evalFailsAll = payload.eval_failures || [];
  const runs = ((payload.runs || {}).runs || []);
  const papers = payload.papers || [];
  const evidence = payload.evidence_cards || [];
  const traces = payload.action_trace || [];

  const hint = document.getElementById("inspector-hint");
  const summary = document.getElementById("inspector-summary");
  const tabs = document.getElementById("inspector-tabs");
  const content = document.getElementById("inspector-content");
  if (state.lastInspectorKey) {
    state.inspectorScrollByKey[state.lastInspectorKey] = content.scrollTop;
  }

  if (!state.selected) {
    hint.textContent = "select event/task/agent";
    summary.innerHTML = `<div class="k">status</div><div>No selection</div>`;
    tabs.innerHTML = "";
    content.innerHTML = `<div class="muted">点击时间线事件或任务行查看细节。</div>`;
    state.lastInspectorKey = "";
    return;
  }

  const sel = state.selected;
  const v = sel.value || {};
  const task = taskForSelection(tasks);
  const run = runForSelection(runs);
  const relRuns = relatedRuns(runs, v, run);
  const ep = Number(v.episode_id || task?.episode_id || run?.episode_id || 0);
  const owner = String(v.owner || task?.owner || "");
  const ownerSummary = ownerEpisodeSummary(events, owner, ep);
  const traceRows = traces
    .filter((t) => (ep <= 0 || Number(t.episode_id || 0) === ep))
    .filter((t) => {
      if (v.task_id && String(t.task_id || "") === String(v.task_id || "")) return true;
      if (run?.run_id && String(t.run_id || "") === String(run.run_id || "")) return true;
      if (owner && String(t.agent_id || "") === owner) return true;
      return false;
    })
    .slice(-20);
  const artifactRows = tbArtifacts
    .filter((a) => (ep <= 0 || Number(a.episode_id || 0) === ep))
    .filter((a) => {
      if (v.task_id && String(a.task_id || "") === String(v.task_id || "")) return true;
      if (run?.run_id && String(a.run_id || "") === String(run.run_id || "")) return true;
      return false;
    })
    .slice(-20);
  const codeLoopRows = codeLoopsAll
    .filter((x) => (ep <= 0 || Number(x.episode_id || 0) === ep))
    .filter((x) => {
      if (owner && String(x.agent_id || "") === owner) return true;
      if (v.task_name && String(x.task_name || "") === String(v.task_name || "")) return true;
      if (v.task_id && traceRows.some((t) => String(t.task_id || "") === String(v.task_id || "") && String(t.agent_id || "") === String(x.agent_id || ""))) return true;
      return !owner && !v.task_name && !v.task_id;
    })
    .slice(-12);
  const gateRows = gatesAll
    .filter((x) => (ep <= 0 || Number(x.episode_id || 0) === ep))
    .filter((x) => {
      if (owner && String(x.agent_id || "") === owner) return true;
      if (v.task_name && String(x.task_name || "") === String(v.task_name || "")) return true;
      return !owner && !v.task_name;
    })
    .slice(-20);
  const evalFailRows = evalFailsAll
    .filter((x) => (ep <= 0 || Number(x.episode_id || 0) === ep))
    .filter((x) => {
      if (v.task_name && String(x.task_name || "") === String(v.task_name || "")) return true;
      if (run?.task_name && String(x.task_name || "") === String(run.task_name || "")) return true;
      return !v.task_name && !run?.task_name;
    })
    .slice(-20);

  hint.textContent = `${sel.type} · ${v.task_id || v.run_id || v.id || "-"}`;
  summary.innerHTML = `
    <div class="k">episode</div><div>${esc(ep || "-")}</div>
    <div class="k">task</div><div>${esc(v.task_type || task?.task_type || run?.task_name || "-")}</div>
    <div class="k">owner</div><div>${esc(owner || "-")}</div>
    <div class="k">state</div><div>${esc(v.state || v.event || "-")}</div>
    <div class="k">run_id</div><div>${esc(run?.run_id || task?.run_id || v.run_id || "-")}</div>
    <div class="k">release</div><div>${esc(v.release_reason || task?.release_reason || "-")}</div>
    <div class="k">deps</div><div>${esc(((task?.depends_on || []).join(" → ")) || "-")}</div>
    <div class="k">confidence</div><div>${task ? `${Math.round(estimateConfidence(task) * 100)}%` : "-"}</div>
  `;

  const tabNames = ["overview", "code", "console", "artifacts", "reasoning", "paper", "compare", "llm_io", "rag_io", "retrieve"];
  tabs.innerHTML = tabNames.map((n) => `<button class="tab-btn ${state.tab === n ? "active" : ""}" data-tab="${n}">${n}</button>`).join("");
  tabs.querySelectorAll("button").forEach((btn) => {
    btn.onclick = () => {
      state.tab = btn.dataset.tab;
      renderInspector();
    };
  });

  const inspectorKey = `${state.tab}|${sel.type}|${String(v.id || v.task_id || v.run_id || "")}|${String(ep || "")}`;
  const restoreScroll = () => {
    state.lastInspectorKey = inspectorKey;
    const y = Number(state.inspectorScrollByKey[inspectorKey] || 0);
    window.requestAnimationFrame(() => {
      content.scrollTop = y;
    });
  };

  if (["llm_io", "rag_io", "retrieve"].includes(state.tab)) {
    const key = ioCacheKey(state.tab);
    const cache = state.ioListByTab[key];
    if (!cache && !state.ioLoadingByTab[key]) {
      content.innerHTML = `<div class="muted">Loading ${esc(state.tab)} ...</div>`;
      fetchIoList(state.tab);
      restoreScroll();
      return;
    }
    if (state.ioLoadingByTab[key] && !cache) {
      content.innerHTML = `<div class="muted">Loading ${esc(state.tab)} ...</div>`;
      restoreScroll();
      return;
    }
    if (!cache || !cache.ok) {
      content.innerHTML = `<div class="muted">Failed to load ${esc(state.tab)}.</div><pre>${esc(cache?.error || "unknown_error")}</pre>`;
      restoreScroll();
      return;
    }
    const rows = cache.rows || [];
    const tab = state.tab;
    const selectedIoId = state.ioSelectionByTab[tab] || (rows[0] ? rows[0].id : "");
    if (selectedIoId && !state.ioSelectionByTab[tab]) {
      state.ioSelectionByTab[tab] = selectedIoId;
    }
    const kind = ioKindForTab(tab);
    const detailKey = `${kind}:${selectedIoId}`;
    const detail = selectedIoId ? state.ioDetailById[detailKey] : null;
    if (selectedIoId && !detail && !state.ioLoadingByDetail[detailKey]) {
      fetchIoDetail(kind, selectedIoId);
    }

    const lastSeenTs = String(state.ioLastSeenTs[tab] || "");
    const listHtml = rows
      .map((r) => {
        const id = String(r.id || "");
        const isSel = id === selectedIoId;
        const ts = String(r.ts || "");
        const isNew = !!lastSeenTs && ts > lastSeenTs;
        const status = String(r.status || (r.ok_status ? "ok" : (r.ok_status === false ? "error" : "")));
        const sub = [
          r.agent_id ? `agent:${r.agent_id}` : "",
          r.action ? `action:${r.action}` : "",
          r.phase ? `phase:${r.phase}` : "",
          r.operation ? `op:${r.operation}` : "",
        ].filter(Boolean).join(" · ");
        return `
          <button class="io-list-item ${isSel ? "active" : ""}" data-io-row-id="${esc(id)}">
            <div class="io-list-top">
              <span class="io-list-title">${esc(toLocal(ts))}</span>
              ${isNew ? '<span class="io-badge new">NEW</span>' : ""}
              ${status ? `<span class="io-badge">${esc(status)}</span>` : ""}
            </div>
            <div class="io-list-sub">${esc(sub || "-")}</div>
          </button>
        `;
      })
      .join("");

    let detailHtml = `<div class="muted">Select a record to view detail.</div>`;
    if (selectedIoId) {
      if (state.ioLoadingByDetail[detailKey] && !detail) {
        detailHtml = `<div class="muted">Loading detail...</div>`;
      } else if (detail && detail.ok) {
        const rec = detail.record || {};
        const metaKv = `
          <div class="kv">
            <div class="k">id</div><div>${esc(selectedIoId)}</div>
            <div class="k">episode</div><div>${esc(rec.episode_id || "-")}</div>
            <div class="k">task</div><div>${esc(rec.task_name || "-")}</div>
            <div class="k">agent</div><div>${esc(rec.agent_id || "-")}</div>
            <div class="k">action</div><div>${esc(rec.action || "-")}</div>
            <div class="k">ts</div><div>${esc(toLocal(rec.ts || ""))}</div>
          </div>
        `;
        if (tab === "llm_io") {
          const promptKey = `llm_prompt_${selectedIoId}`;
          const rawKey = `llm_raw_${selectedIoId}`;
          detailHtml = `
            <div class="io-block">
              <div class="io-block-title">Meta</div>
              ${metaKv}
            </div>
            <div class="io-block">
              <div class="io-block-title">Prompt Summary</div>
              <div class="kv">
                <div class="k">prompt_chars</div><div>${esc(rec.prompt_chars)}</div>
                <div class="k">response_chars</div><div>${esc(rec.response_chars)}</div>
                <div class="k">ok</div><div>${esc(rec.ok_status)}</div>
                <div class="k">reason</div><div>${esc(rec.reason || "-")}</div>
              </div>
            </div>
            ${renderCollapsibleText("Prompt Body", rec.prompt || "", promptKey, 500)}
            <div class="io-block">
              <div class="io-block-title">Parsed Output</div>
              <div class="json-host">${renderJsonTree(rec.parsed_json || {}, `llm_parsed_${selectedIoId}`)}</div>
            </div>
            ${renderCollapsibleText("Raw Response", rec.raw_response || "", rawKey, 500)}
          `;
        } else if (tab === "rag_io") {
          const ctxKey = `rag_ctx_${selectedIoId}`;
          detailHtml = `
            <div class="io-block">
              <div class="io-block-title">Meta</div>
              ${metaKv}
            </div>
            <div class="io-block">
              <div class="io-block-title">Operation</div>
              <div class="kv">
                <div class="k">operation</div><div>${esc(rec.operation || "-")}</div>
                <div class="k">status</div><div>${esc(rec.status || "-")}</div>
                <div class="k">fallback_reason</div><div>${esc(rec.fallback_reason || "-")}</div>
                <div class="k">retrieval_mode</div><div>${esc(rec.retrieval_mode || "-")}</div>
                <div class="k">selected_count</div><div>${esc(rec.selected_count || 0)}</div>
              </div>
            </div>
            <div class="io-block">
              <div class="io-block-title">Query Params</div>
              <div class="json-host">${renderJsonTree({
                query_text: rec.query_text || "",
                collections: rec.collections || [],
                quotas: rec.quotas || {},
                refs: rec.refs || []
              }, `rag_query_${selectedIoId}`)}</div>
            </div>
            <div class="io-block">
              <div class="io-block-title">Selected Rows</div>
              <div class="table-compact-wrap">
                <table class="table-compact">
                  <thead><tr><th>source</th><th>id</th><th>score</th><th>mode</th></tr></thead>
                  <tbody>
                    ${(rec.selected_rows || []).slice(0, 30).map((x) => `
                      <tr>
                        <td>${esc(x.source_type || "-")}</td>
                        <td>${esc(x.source_id || "-")}</td>
                        <td>${esc(fmtNum(x.score, 3))}</td>
                        <td>${esc(x.retrieval_mode || "-")}</td>
                      </tr>
                    `).join("")}
                  </tbody>
                </table>
              </div>
            </div>
            ${renderCollapsibleText("Context", rec.context || "", ctxKey, 500)}
          `;
        } else {
          const payloadKey = `retrieve_payload_${selectedIoId}`;
          detailHtml = `
            <div class="io-block">
              <div class="io-block-title">Meta</div>
              ${metaKv}
            </div>
            <div class="io-block">
              <div class="io-block-title">Pipeline Phase</div>
              <div class="kv">
                <div class="k">phase</div><div>${esc(rec.phase || "-")}</div>
                <div class="k">source</div><div>${esc((rec.payload || {}).source || "-")}</div>
                <div class="k">ok</div><div>${esc((rec.payload || {}).ok ?? true)}</div>
                <div class="k">reward</div><div>${esc((rec.payload || {}).reward ?? "-")}</div>
              </div>
            </div>
            ${renderCollapsibleText("Pipeline Payload", JSON.stringify(rec.payload || {}, null, 2), payloadKey, 700)}
            <div class="io-block">
              <div class="io-block-title">Guardrail (nearest)</div>
              <div class="json-host">${renderJsonTree((rec.related_guardrail || [])[0] || {}, `retrieve_guard_${selectedIoId}`)}</div>
            </div>
            <div class="io-block">
              <div class="io-block-title">Evidence (nearest)</div>
              <div class="json-host">${renderJsonTree((rec.related_evidence || [])[0] || {}, `retrieve_evid_${selectedIoId}`)}</div>
            </div>
          `;
        }
      } else if (detail && !detail.ok) {
        detailHtml = `<div class="muted">Detail not found.</div><pre>${esc(detail.error || "detail_not_found")}</pre>`;
      }
    }
    const scopeVal = String(state.ioScope.episode || "current").toLowerCase() === "all" ? "all" : "current";
    content.innerHTML = `
      <div class="io-shell">
        <div class="io-toolbar">
          <label>
            episode scope:
            <select data-io-scope-episode>
              <option value="current" ${scopeVal === "current" ? "selected" : ""}>current</option>
              <option value="all" ${scopeVal === "all" ? "selected" : ""}>all</option>
            </select>
          </label>
          <button class="io-mini-btn" data-io-refresh="1">Refresh</button>
        </div>
        <div class="io-list">${listHtml || '<div class="muted">No records.</div>'}</div>
        <div class="io-detail">${detailHtml}</div>
      </div>
    `;
    const scopeSelect = content.querySelector("[data-io-scope-episode]");
    if (scopeSelect) {
      scopeSelect.onchange = () => {
        state.ioScope.episode = String(scopeSelect.value || "current");
        state.ioSelectionByTab[tab] = "";
        fetchIoList(tab);
      };
    }
    const refreshBtn = content.querySelector("[data-io-refresh]");
    if (refreshBtn) {
      refreshBtn.onclick = () => {
        const cacheKey = ioCacheKey(tab);
        delete state.ioListByTab[cacheKey];
        state.ioSelectionByTab[tab] = "";
        fetchIoList(tab);
      };
    }
    content.querySelectorAll("[data-io-row-id]").forEach((btn) => {
      btn.onclick = () => {
        const id = btn.dataset.ioRowId || "";
        state.ioSelectionByTab[tab] = id;
        if (id) fetchIoDetail(kind, id);
        renderInspector();
      };
    });
    bindIoToggles(content);
    restoreScroll();
    return;
  }

  if (state.tab === "overview") {
    const relatedTaskRows = tasks
      .filter((t) => (ep <= 0 || Number(t.episode_id || 0) === ep))
      .filter((t) => {
        if (v.task_id && String(t.task_id || "") === String(v.task_id || "")) return true;
        if (owner && String(t.owner || "") === owner) return true;
        if (v.task_type && String(t.task_type || "") === String(v.task_type || "")) return true;
        return false;
      })
      .slice(-20);

    content.innerHTML = `
      <div class="muted">code loop attempts</div>
      <pre>${esc(JSON.stringify(codeLoopRows, null, 2))}</pre>
      <div class="muted">precondition gates</div>
      <pre>${esc(JSON.stringify(gateRows, null, 2))}</pre>
      <div class="muted">evaluation failures</div>
      <pre>${esc(JSON.stringify(evalFailRows, null, 2))}</pre>
      <div class="muted">owner summary in episode</div>
      <pre>${esc(JSON.stringify(ownerSummary, null, 2))}</pre>
      <div class="muted">related tasks</div>
      <pre>${esc(JSON.stringify(relatedTaskRows.map((x) => ({
        task_id: x.task_id,
        task_type: x.task_type,
        state: x.state,
        owner: x.owner,
        run_id: x.run_id || "",
        release_reason: x.release_reason || "",
        claim_to_start_ticks: x.claim_to_start_ticks,
        start_to_complete_ticks: x.start_to_complete_ticks
      })), null, 2))}</pre>
      <div class="muted">candidate runs in episode</div>
      <pre>${esc(JSON.stringify(relRuns.map((r) => ({ run_id: r.run_id, exit_code: r.exit_code, dev_score: r.dev_score, duration_s: r.duration_s, error: r.error_signature })), null, 2))}</pre>
    `;
    restoreScroll();
    return;
  }

  if (state.tab === "code") {
    if (!run) {
      const refs = traceRows.map((x) => ({ ts: x.ts, action: x.action, status: x.status, task_id: x.task_id, reason: x.reason }));
      content.innerHTML = `<div class="muted">No linked run for this selection. Below are nearby action records.</div><pre>${esc(JSON.stringify(refs, null, 2))}</pre>`;
      restoreScroll();
      return;
    }
    const files = ((run.code_plan || {}).files || []);
    if (!files.length) {
      content.innerHTML = `<div class="muted">No code plan files.</div>`;
      return;
    }
    if (!state.filePath || !files.some((f) => f.path === state.filePath)) {
      state.filePath = files[0].path;
    }
    const cur = files.find((f) => f.path === state.filePath) || files[0];
    const tabsHtml = files.map((f) => `<button class="tab-btn ${f.path === state.filePath ? "active" : ""}" data-file="${esc(f.path)}">${esc(f.path)}</button>`).join("");
    content.innerHTML = `<div class="muted">run_cmd: ${esc(run.effective_run_cmd || (run.code_plan || {}).run_cmd || "-")}</div><div class="tabs">${tabsHtml}</div><div class="workbench-split"><pre>${esc(cur.content || "")}</pre><pre>${esc(run.stderr || run.stdout || "")}</pre></div>`;
    content.querySelectorAll("[data-file]").forEach((btn) => {
      btn.onclick = () => {
        state.filePath = btn.dataset.file || "";
        renderInspector();
      };
    });
    restoreScroll();
    return;
  }

  if (state.tab === "console") {
    if (!run) {
      content.innerHTML = `<div class="muted">No linked run. Check task release/action trace instead.</div><pre>${esc(JSON.stringify({
        release_reason: v.release_reason || task?.release_reason || "-",
        action_trace: traceRows.map((x) => ({ ts: x.ts, action: x.action, status: x.status, reason: x.reason }))
      }, null, 2))}</pre>`;
      restoreScroll();
      return;
    }
    const errRows = runs
      .filter((r) => Number(r.episode_id) === Number(run.episode_id))
      .map((r) => r.error_signature || "")
      .filter(Boolean);
    const grouped = {};
    errRows.forEach((e) => {
      grouped[e] = (grouped[e] || 0) + 1;
    });
    const hints = buildFailureHints(run.stderr || "");
    const isFail = String(run.exit_code) !== "0";

    content.innerHTML = `
      <div class="kv ${isFail ? "pulse-alert" : ""}">
        <div class="k">exit_code</div><div>${esc(run.exit_code)}</div>
        <div class="k">duration_s</div><div>${esc(Number(run.duration_s || 0).toFixed(3))}</div>
        <div class="k">timed_out</div><div>${esc(run.timed_out)}</div>
        <div class="k">error_signature</div><div>${esc(run.error_signature || "-")}</div>
      </div>
      <div class="muted">Failure Anatomy · error clusters</div>
      <pre>${esc(JSON.stringify(grouped, null, 2))}</pre>
      <div class="muted">Auto Diagnosis</div>
      <pre>${esc(JSON.stringify(hints, null, 2))}</pre>
      <div class="workbench-split">
        <div><div class="muted">stderr</div><pre>${esc(run.stderr || "")}</pre></div>
        <div><div class="muted">stdout</div><pre>${esc(run.stdout || "")}</pre></div>
      </div>
    `;
    restoreScroll();
    return;
  }

  if (state.tab === "artifacts") {
    const dc = run?.data_card || {};
    const mc = run?.method_card || {};
    const eRows = evidence.filter((e) => (run?.run_id && String(e.run_id || "") === String(run.run_id || "")) || (ep > 0 && Number(e.episode_id || 0) === ep)).slice(-20);
    const pRows = papers.filter((p) => ep > 0 && Number(p.episode_id || 0) === ep).slice(-20);

    content.innerHTML = `
      <div class="kv">
        <div class="k">submission</div><div>${esc(run?.submission_path || "-")}</div>
        <div class="k">workspace</div><div>${esc(run?.workspace_dir || "-")}</div>
        <div class="k">code_log</div><div>${esc(run?.code_log_path || "-")}</div>
        <div class="k">solver_log</div><div>${esc(run?.solver_log_path || "-")}</div>
      </div>
      <div class="muted">taskboard artifacts index</div>
      <pre>${esc(JSON.stringify(artifactRows, null, 2))}</pre>
      <div class="muted">data card</div>
      <pre>${esc(JSON.stringify({ task_name: dc.task_name, degraded: dc.degraded, split_stats: dc.split_stats, risk_flags: dc.risk_flags }, null, 2))}</pre>
      <div class="muted">method card</div>
      <pre>${esc(JSON.stringify({ task_name: mc.task_name, metric: mc.metric, category: mc.category, baselines: (mc.recommended_baselines || []).map((x) => x.name) }, null, 2))}</pre>
      <div class="muted">lineage run→evidence→paper</div>
      <pre>${esc(JSON.stringify({ run_id: run?.run_id || "", evidence_refs: eRows.map((x) => x.evidence_id || x.task_id || x.kind), paper_refs: pRows.map((x) => x.paper_id) }, null, 2))}</pre>
    `;
    restoreScroll();
    return;
  }

  if (state.tab === "reasoning") {
    const rows = traces
      .filter((t) => (state.episode <= 0 || Number(t.episode_id) === state.episode))
      .filter((t) => !v.owner || String(t.agent_id || "") === String(v.owner || ""))
      .slice(-60);
    content.innerHTML = `<div class="muted">structured trace (non-CoT)</div><pre>${esc(JSON.stringify(rows, null, 2))}</pre>`;
    restoreScroll();
    return;
  }

  if (state.tab === "paper") {
    const ep = Number(v.episode_id || run?.episode_id || 0);
    const rows = papers.filter((p) => ep <= 0 || Number(p.episode_id) === ep).slice(-20);
    content.innerHTML = rows.length
      ? rows
          .map(
            (p) => `<div class="panel" style="margin-bottom:8px;"><div class="panel-body"><b>${esc(p.paper_id || "-")}</b> <span class="muted">${esc(p.source || "")}</span><div class="kv" style="margin-top:6px;"><div class="k">agent</div><div>${esc(p.agent_id || "-")}</div><div class="k">fitness</div><div>${esc(p.fitness)}</div><div class="k">f1</div><div>${esc(p.f1)}</div><div class="k">publishable</div><div>${esc(p.publishable)}</div><div class="k">replication_ok</div><div>${esc(p.replication_ok)}</div></div></div></div>`
          )
          .join("")
      : `<div class="muted">No paper records.</div>`;
    restoreScroll();
    return;
  }

  if (state.tab === "compare") {
    const byEp = {};
    runs.forEach((r) => {
      if (state.episode > 0 && Number(r.episode_id) !== state.episode) return;
      const k = String(r.episode_id || 0);
      if (!byEp[k]) byEp[k] = { runs: 0, fail: 0, timeout: 0, avg_dur: 0, dev_scores: [] };
      byEp[k].runs += 1;
      byEp[k].avg_dur += Number(r.duration_s || 0);
      if (r.exit_code !== null && r.exit_code !== 0) byEp[k].fail += 1;
      if (r.timed_out) byEp[k].timeout += 1;
      if (r.dev_score != null) byEp[k].dev_scores.push(Number(r.dev_score));
    });

    Object.values(byEp).forEach((item) => {
      item.avg_dur = item.runs ? item.avg_dur / item.runs : 0;
      item.dev_avg = item.dev_scores.length ? item.dev_scores.reduce((a, b) => a + b, 0) / item.dev_scores.length : null;
    });

    content.innerHTML = `<div class="muted">episode compare (runs/errors/dev score)</div><pre>${esc(JSON.stringify(byEp, null, 2))}</pre>`;
    restoreScroll();
  }
}

function renderAll() {
  const payload = state.payload || {};
  const meta = payload.meta || {};
  const tb = payload.taskboard || {};
  const events = tb.events || [];
  const tasks = tb.task_snapshot || [];
  const agents = tb.agents || [];
  const alerts = meta.alerts || [];
  const timing = tb.timing || {};
  const teamSummary = (payload.team || {}).summary || {};
  const runs = (payload.runs || {}).runs || [];
  const runMap = Object.fromEntries(runs.map((r) => [String(r.run_id), r]));
  const serverInfo = payload.server || {};

  setupEpisodeSelect(events, runs);
  renderHeader(meta, timing, teamSummary, serverInfo);
  renderAlertCenter(alerts);
  renderTypeFilters(tasks);
  renderErrorFilters();
  renderAgents(agents);

  const cutoff = replayCutoffTs(events);
  renderTimeline(events, cutoff);
  renderTaskboard(tasks, cutoff, runMap);
  renderInspector();
}

async function bootstrap() {
  bindControls();
  bindColumnSplitters();
  bindMainSplitter();
  try {
    await fetchSnapshot();
  } catch (e) {
    document.getElementById("inspector-content").innerHTML = `<pre>${esc(String(e))}</pre>`;
  }
  connectSSE();
  window.addEventListener("resize", () => {
    applyColumnWidths();
    applyMainSplitRatio(state.mainSplitRatio);
    const payload = state.payload || {};
    const events = ((payload.taskboard || {}).events || []);
    renderTimeline(events, replayCutoffTs(events));
  });
}

bootstrap();
