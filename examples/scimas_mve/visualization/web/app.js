const stageOrder = ["prepare", "profile", "literature", "read", "hypothesize", "experiment", "review", "write", "replicate", "other"];

const state = {
  payload: null,
  selected: null,
  tab: "overview",
  episode: 0,
  mode: "live",
  replayPercent: 100,
  search: "",
  runSearch: "",
  agent: "",
  typeFilter: new Set(),
  errFilter: new Set(),
  snapshots: [],
};

function esc(s) {
  return String(s == null ? "" : s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}
function toLocal(ts) {
  if (!ts) return "-";
  try { return new Date(ts).toLocaleString(); } catch { return String(ts); }
}
function setHealth(dot, level) {
  dot.classList.remove("ok", "warn", "err");
  if (level === "red" || level === "error") dot.classList.add("err");
  else if (level === "yellow" || level === "warn") dot.classList.add("warn");
  else dot.classList.add("ok");
}

async function loadSnapshot() {
  const res = await fetch("/api/snapshot", { cache: "no-store" });
  if (!res.ok) throw new Error(`snapshot_http_${res.status}`);
  const data = await res.json();
  applyPayload(data);
}

function connectSSE() {
  const es = new EventSource("/api/events");
  es.addEventListener("snapshot", (evt) => {
    try { applyPayload(JSON.parse(evt.data || "{}")); } catch (_) {}
  });
  es.onerror = () => {
    setTimeout(() => {
      try { es.close(); } catch (_) {}
      connectSSE();
    }, 2000);
  };
}

function getReplayCutoff(events) {
  if (state.mode !== "replay") return null;
  const ts = events.map((e) => Date.parse(e.ts || "")).filter(Number.isFinite).sort((a, b) => a - b);
  if (!ts.length) return null;
  const idx = Math.max(0, Math.min(ts.length - 1, Math.floor((state.replayPercent / 100) * (ts.length - 1))));
  return ts[idx];
}

function matchFilters(obj, replayCutoff) {
  const ep = Number(obj.episode_id || 0);
  if (state.episode > 0 && ep !== state.episode) return false;
  if (replayCutoff != null) {
    const t = Date.parse(obj.ts || obj.last_update || "");
    if (Number.isFinite(t) && t > replayCutoff) return false;
  }
  if (state.agent && String(obj.owner || obj.agent_id || "").toLowerCase() !== state.agent.toLowerCase()) return false;
  const text = JSON.stringify(obj).toLowerCase();
  if (state.search && !text.includes(state.search)) return false;
  if (state.runSearch && !`${obj.run_id || ""} ${obj.task_id || ""} ${obj.owner || ""}`.toLowerCase().includes(state.runSearch)) return false;
  if (state.typeFilter.size > 0 && !state.typeFilter.has(String(obj.task_type || ""))) return false;
  if (state.errFilter.size > 0) {
    let ok = false;
    for (const e of state.errFilter) {
      if (text.includes(e)) { ok = true; break; }
    }
    if (!ok) return false;
  }
  return true;
}

function applyPayload(payload) {
  state.payload = payload || {};
  render();
}

function bindControls() {
  document.getElementById("search-input").oninput = (e) => { state.search = e.target.value.trim().toLowerCase(); render(); };
  document.getElementById("run-search").oninput = (e) => { state.runSearch = e.target.value.trim().toLowerCase(); render(); };
  document.getElementById("episode-select").onchange = (e) => { state.episode = Number(e.target.value || 0); render(); };
  document.getElementById("mode-select").onchange = (e) => {
    state.mode = e.target.value;
    document.getElementById("replay-slider").classList.toggle("hidden", state.mode !== "replay");
    render();
  };
  document.getElementById("replay-slider").oninput = (e) => { state.replayPercent = Number(e.target.value || 100); render(); };

  document.getElementById("snapshot-btn").onclick = () => {
    const snap = {
      ts: new Date().toISOString(),
      episode: state.episode,
      mode: state.mode,
      selected: state.selected,
      top_release: (((state.payload || {}).meta || {}).top_release_reason || ""),
    };
    state.snapshots.push(snap);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(snap, null, 2)], { type: "application/json" }));
    a.download = `scimas_snapshot_${Date.now()}.json`;
    a.click();
  };
  document.getElementById("export-btn").onclick = () => {
    const data = { payload: state.payload || {}, snapshots: state.snapshots };
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    a.download = `scimas_export_${Date.now()}.json`;
    a.click();
  };
  document.getElementById("inspector-toggle").onclick = () => {
    document.querySelector(".inspector").classList.toggle("hidden");
  };
}

function renderHeader(payload) {
  const meta = payload.meta || {};
  const dot = document.getElementById("health-dot");
  setHealth(dot, meta.health || "green");
  document.getElementById("health-text").textContent = `health:${meta.health || "green"}`;
  document.getElementById("alerts-badge").textContent = `alerts:${meta.alert_count || 0}`;
  document.getElementById("release-badge").textContent = `top release:${meta.top_release_reason || "-"}`;
  const eps = new Set([0]);
  ((payload.taskboard || {}).events || []).forEach((e) => { if (Number(e.episode_id || 0) > 0) eps.add(Number(e.episode_id)); });
  const select = document.getElementById("episode-select");
  const opts = [...eps].sort((a, b) => a - b).map((ep) => `<option value="${ep}">${ep === 0 ? "All Episodes" : `Episode ${ep}`}</option>`);
  select.innerHTML = opts.join("");
  select.value = String(state.episode);
}

function renderFilters(tasks) {
  const typeWrap = document.getElementById("type-filters");
  const errWrap = document.getElementById("error-filters");
  const types = [...new Set(tasks.map((t) => String(t.task_type || "")).filter(Boolean))].sort();
  typeWrap.innerHTML = types.map((t) => `<button class="chip ${state.typeFilter.has(t) ? "on" : ""}" data-type="${esc(t)}">${esc(t)}</button>`).join("");
  typeWrap.querySelectorAll(".chip").forEach((el) => {
    el.onclick = () => {
      const t = el.dataset.type || "";
      if (state.typeFilter.has(t)) state.typeFilter.delete(t); else state.typeFilter.add(t);
      render();
    };
  });
  const errs = ["lease_expired", "inner_action_failed", "timeout", "oom", "json_parse"];
  errWrap.innerHTML = errs.map((t) => `<button class="chip ${state.errFilter.has(t) ? "on" : ""}" data-err="${t}">${t}</button>`).join("");
  errWrap.querySelectorAll(".chip").forEach((el) => {
    el.onclick = () => {
      const t = el.dataset.err || "";
      if (state.errFilter.has(t)) state.errFilter.delete(t); else state.errFilter.add(t);
      render();
    };
  });
}

function renderAgents(agents) {
  const list = document.getElementById("agent-list");
  document.getElementById("agent-count").textContent = `${agents.length}`;
  list.innerHTML = agents.map((a) => {
    const active = state.agent === a.agent_id ? "active" : "";
    const cls = a.status === "running" ? "ok" : (a.status === "error" ? "err" : "warn");
    return `<div class="agent-row ${active}" data-aid="${esc(a.agent_id)}"><div><span class="status-dot ${cls}"></span>${esc(a.agent_id)}</div><div class="muted">${a.claimed}/${a.completed}/${a.released}</div></div>`;
  }).join("");
  list.querySelectorAll(".agent-row").forEach((row) => {
    row.onclick = () => {
      const aid = row.dataset.aid || "";
      state.agent = state.agent === aid ? "" : aid;
      render();
    };
  });
}

function renderAlerts(alerts) {
  const box = document.getElementById("alert-list");
  if (!alerts.length) {
    box.innerHTML = `<div class="muted">No active alert</div>`;
    return;
  }
  box.innerHTML = alerts.map((a) => `<div><span class="status-dot ${a.level === "error" ? "err" : "warn"}"></span><b>${esc(a.key)}</b> <span class="muted">${esc(a.value)}</span></div>`).join("");
}

function renderTimeline(events, replayCutoff) {
  const lanes = Object.fromEntries(stageOrder.map((s) => [s, []]));
  events.filter((e) => matchFilters(e, replayCutoff)).forEach((e) => {
    const s = stageOrder.includes(e.stage) ? e.stage : "other";
    lanes[s].push(e);
  });
  for (const s of stageOrder) lanes[s] = (lanes[s] || []).slice(-24);
  const root = document.getElementById("timeline");
  root.innerHTML = stageOrder.map((s) => {
    const items = lanes[s] || [];
    return `<div class="lane"><div class="lane-head">${s} <span class="muted">${items.length}</span></div><div class="lane-body">${
      items.map((e) => {
        const cls = e.event === "complete" ? "success" : (e.event === "release" ? "release fail" : "");
        return `<div class="event ${cls}" data-eid="${esc(e.id)}"><b>${esc(e.task_type || e.stage)}</b><small>${esc(e.task_id)} · ${esc(e.owner || "-")}</small><small>${esc(e.event)} · ${toLocal(e.ts)}</small><small>${esc(e.run_id || "")} C→S:${esc(e.claim_to_start_ticks ?? "-")} S→C:${esc(e.start_to_complete_ticks ?? "-")}</small></div>`;
      }).join("")
    }</div></div>`;
  }).join("");
  root.querySelectorAll(".event").forEach((el) => {
    el.onclick = () => {
      const e = events.find((x) => x.id === el.dataset.eid);
      state.selected = { type: "event", value: e };
      state.tab = "overview";
      renderInspector();
    };
  });
}

function renderTaskboard(tasks, replayCutoff) {
  const filtered = tasks.filter((t) => matchFilters(t, replayCutoff));
  const counts = { open: 0, claimed: 0, running: 0, completed: 0, released: 0 };
  filtered.forEach((x) => { counts[String(x.state || "unknown")] = (counts[String(x.state || "unknown")] || 0) + 1; });
  const c2s = filtered.map((x) => Number(x.claim_to_start_ticks)).filter((x) => Number.isFinite(x) && x >= 0);
  const s2c = filtered.map((x) => Number(x.start_to_complete_ticks)).filter((x) => Number.isFinite(x) && x >= 0);
  const avg = (arr) => arr.length ? (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(2) : "-";

  document.getElementById("task-pills").innerHTML = [
    `open:${counts.open || 0}`, `claimed:${counts.claimed || 0}`, `running:${counts.running || 0}`, `completed:${counts.completed || 0}`,
    `avg C→S:${avg(c2s)}`, `avg S→C:${avg(s2c)}`,
  ].map((x) => `<span class="pill">${x}</span>`).join("");
  document.getElementById("task-summary").textContent = `${filtered.length} rows`;

  const tbody = document.getElementById("task-tbody");
  tbody.innerHTML = filtered.map((t) => {
    const sel = state.selected && state.selected.type === "task" && state.selected.value.task_id === t.task_id ? "sel" : "";
    const deps = (t.depends_on || []).slice(0, 2).join("→") || "-";
    return `<tr class="${sel}" data-task="${esc(t.task_id)}"><td>${esc(t.task_id)}</td><td>${esc(t.task_type)}</td><td>${esc(t.state)}</td><td>${esc(t.owner || "-")}</td><td>${esc(t.lease_ttl)}/${esc(t.heartbeat)}</td><td>${esc(deps)}</td><td>${esc(t.claim_to_start_ticks ?? "-")}</td><td>${esc(t.start_to_complete_ticks ?? "-")}</td><td>${esc(t.release_reason || "-")}</td><td>${esc(t.run_id || "-")}</td></tr>`;
  }).join("");
  tbody.querySelectorAll("tr").forEach((row) => {
    row.onclick = () => {
      const t = tasks.find((x) => x.task_id === row.dataset.task);
      state.selected = { type: "task", value: t };
      state.tab = "overview";
      render();
    };
  });
}

function runForSelection(runs) {
  if (!state.selected) return null;
  const v = state.selected.value || {};
  if (state.selected.type === "run") return v;
  if (!v.run_id) return null;
  return runs.find((r) => String(r.run_id) === String(v.run_id) && (state.episode <= 0 || Number(r.episode_id) === state.episode))
    || runs.find((r) => String(r.run_id) === String(v.run_id));
}

function renderInspector() {
  const payload = state.payload || {};
  const runs = (payload.runs || {}).runs || [];
  const papers = payload.papers || [];
  const evidenceCards = payload.evidence_cards || [];
  const traces = payload.action_trace || [];
  const summary = document.getElementById("inspector-summary");
  const content = document.getElementById("inspector-content");
  const tabs = document.getElementById("inspector-tabs");
  const sel = state.selected;

  if (!sel) {
    summary.innerHTML = "No selection";
    tabs.innerHTML = "";
    content.innerHTML = `<div class="muted">Select timeline or task row.</div>`;
    return;
  }
  const run = runForSelection(runs);
  const v = sel.value || {};
  summary.innerHTML = `<div class="kv"><div class="k">episode</div><div>${esc(v.episode_id || run?.episode_id || "-")}</div><div class="k">task</div><div>${esc(v.task_type || run?.task_name || "-")}</div><div class="k">owner</div><div>${esc(v.owner || "-")}</div><div class="k">run</div><div>${esc(run?.run_id || v.run_id || "-")}</div></div>`;
  const names = ["overview", "code", "console", "artifacts", "reasoning", "paper", "compare"];
  tabs.innerHTML = names.map((n) => `<button class="tab ${state.tab === n ? "on" : ""}" data-tab="${n}">${n}</button>`).join("");
  tabs.querySelectorAll(".tab").forEach((b) => { b.onclick = () => { state.tab = b.dataset.tab; renderInspector(); }; });

  if (state.tab === "overview") {
    const eruns = runs.filter((r) => Number(r.episode_id) === Number(v.episode_id || run?.episode_id || 0)).slice(-12);
    content.innerHTML = `<div class="muted">linked runs</div><pre>${esc(JSON.stringify(eruns.map((r) => ({ run_id: r.run_id, exit_code: r.exit_code, dev_score: r.dev_score, duration_s: r.duration_s, error: r.error_signature })), null, 2))}</pre>`;
    return;
  }
  if (state.tab === "code") {
    if (!run) { content.innerHTML = `<div class="muted">No run linked.</div>`; return; }
    const files = ((run.code_plan || {}).files || []);
    content.innerHTML = `<div class="muted">run_cmd: ${esc((run.code_plan || {}).run_cmd || "-")}</div><pre>${esc(JSON.stringify(files, null, 2))}</pre>`;
    return;
  }
  if (state.tab === "console") {
    if (!run) { content.innerHTML = `<div class="muted">No run linked.</div>`; return; }
    const epErr = runs.filter((r) => Number(r.episode_id) === Number(run.episode_id)).map((r) => r.error_signature).filter(Boolean);
    const bucket = {};
    epErr.forEach((e) => { bucket[e] = (bucket[e] || 0) + 1; });
    content.innerHTML = `<div class="kv"><div class="k">exit_code</div><div>${esc(run.exit_code)}</div><div class="k">duration</div><div>${esc(run.duration_s)}</div><div class="k">timed_out</div><div>${esc(run.timed_out)}</div><div class="k">error_signature</div><div>${esc(run.error_signature || "-")}</div></div><div class="muted">error clusters</div><pre>${esc(JSON.stringify(bucket, null, 2))}</pre><div class="muted">stderr</div><pre>${esc(run.stderr || "")}</pre><div class="muted">stdout</div><pre>${esc(run.stdout || "")}</pre>`;
    return;
  }
  if (state.tab === "artifacts") {
    if (!run) { content.innerHTML = `<div class="muted">No run linked.</div>`; return; }
    const ev = evidenceCards.filter((e) => String(e.run_id || "") === String(run.run_id || "") || Number(e.episode_id || 0) === Number(run.episode_id || 0)).slice(-20);
    const pp = papers.filter((p) => Number(p.episode_id || 0) === Number(run.episode_id || 0)).slice(-10);
    content.innerHTML = `<div class="kv"><div class="k">submission</div><div>${esc(run.submission_path || "-")}</div><div class="k">workspace</div><div>${esc(run.workspace_dir || "-")}</div><div class="k">code_log</div><div>${esc(run.code_log_path || "-")}</div></div><div class="muted">data card</div><pre>${esc(JSON.stringify(run.data_card || {}, null, 2))}</pre><div class="muted">method card</div><pre>${esc(JSON.stringify(run.method_card || {}, null, 2))}</pre><div class="muted">lineage run→evidence→paper</div><pre>${esc(JSON.stringify({ run_id: run.run_id, evidence_refs: ev.map((x) => x.evidence_id || x.task_id), paper_refs: pp.map((x) => x.paper_id) }, null, 2))}</pre>`;
    return;
  }
  if (state.tab === "reasoning") {
    const rows = traces.filter((t) => state.episode <= 0 || Number(t.episode_id) === state.episode).slice(-80);
    content.innerHTML = `<div class="muted">structured action trace</div><pre>${esc(JSON.stringify(rows, null, 2))}</pre>`;
    return;
  }
  if (state.tab === "paper") {
    const ep = Number(v.episode_id || run?.episode_id || 0);
    const rows = papers.filter((p) => ep <= 0 || Number(p.episode_id) === ep).slice(-20);
    content.innerHTML = rows.length ? rows.map((p) => `<div><b>${esc(p.paper_id)}</b> <span class="muted">${esc(p.agent_id)} f1=${esc(p.f1)} publishable=${esc(p.publishable)} replication_ok=${esc(p.replication_ok)}</span></div>`).join("<hr/>") : `<div class="muted">No paper rows.</div>`;
    return;
  }
  if (state.tab === "compare") {
    const byEp = {};
    runs.forEach((r) => {
      if (state.episode > 0 && Number(r.episode_id) !== state.episode) return;
      const k = String(r.episode_id || 0);
      byEp[k] = byEp[k] || { runs: 0, fail: 0, timeout: 0, avg_dur: 0, dev: [] };
      byEp[k].runs += 1;
      byEp[k].avg_dur += Number(r.duration_s || 0);
      if (r.exit_code !== null && r.exit_code !== 0) byEp[k].fail += 1;
      if (r.timed_out) byEp[k].timeout += 1;
      if (r.dev_score != null) byEp[k].dev.push(Number(r.dev_score));
    });
    Object.values(byEp).forEach((x) => {
      x.avg_dur = x.runs ? x.avg_dur / x.runs : 0;
      x.dev_avg = x.dev.length ? x.dev.reduce((a, b) => a + b, 0) / x.dev.length : null;
    });
    content.innerHTML = `<pre>${esc(JSON.stringify(byEp, null, 2))}</pre>`;
  }
}

function render() {
  const payload = state.payload || {};
  const tb = payload.taskboard || {};
  const events = tb.events || [];
  const tasks = tb.task_snapshot || [];
  const agents = tb.agents || [];
  const replayCutoff = getReplayCutoff(events);

  renderHeader(payload);
  renderFilters(tasks);
  renderAgents(agents);
  renderAlerts((payload.meta || {}).alerts || []);
  renderTimeline(events, replayCutoff);
  renderTaskboard(tasks, replayCutoff);
  renderInspector();
}

async function bootstrap() {
  bindControls();
  try {
    await loadSnapshot();
  } catch (e) {
    document.getElementById("inspector-content").innerHTML = `<pre>${esc(String(e))}</pre>`;
  }
  connectSSE();
}

bootstrap();

