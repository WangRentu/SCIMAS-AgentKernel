/* ============================================================
   SCIMAS Research Console — app.js
   Rebuilt with LLM Trace, Markdown rendering, layered views
   ============================================================ */

"use strict";

// ── Utilities ─────────────────────────────────────────────────

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function toLocal(ts) {
  if (!ts) return "—";
  try { return new Date(ts).toLocaleString(); } catch { return String(ts); }
}

function toTime(ts) {
  if (!ts) return "—";
  try { return new Date(ts).toLocaleTimeString(); } catch { return String(ts); }
}

function fmtNum(v, d = 3) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : "—";
}

function fmtInt(v) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.round(n).toLocaleString() : "0";
}

function fmtCny(v, d = 4) {
  const n = Number(v);
  return Number.isFinite(n) ? `¥${n.toFixed(d)}` : "¥0.0000";
}

function fmtK(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "0";
  if (n >= 1000000) return `${(n/1000000).toFixed(1)}M`;
  if (n >= 1000)    return `${(n/1000).toFixed(1)}k`;
  return String(Math.round(n));
}

function toInt(v, def = 0) {
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : def;
}

function inferScoreDirection(run) {
  const explicit = run && run.metric_lower_is_better;
  if (typeof explicit === "boolean") return explicit ? "lower" : "higher";
  const metricCandidates = [
    run && run.metric_name,
    run && run.task_metric,
    run && run.method_card && run.method_card.metric,
    run && run.data_card && run.data_card.metric,
  ];
  const metric = String(metricCandidates.find(x => typeof x === "string" && x.trim()) || "").toLowerCase();
  if (!metric) return "unknown";
  if (/(mase|mae|mse|rmse|mape|smape|loss|error|nll|cross.?entropy|perplexity|wer|cer|distance|latency)/.test(metric)) {
    return "lower";
  }
  if (/(acc|accuracy|auc|f1|precision|recall|r2|ndcg|map|bleu|rouge|pass@|hit@)/.test(metric)) {
    return "higher";
  }
  return "unknown";
}

function scoreQuality(rawScore, direction) {
  const s = Number(rawScore);
  if (!Number.isFinite(s)) return null;
  if (direction === "lower") return Math.max(0, Math.min(1, 1 / (1 + Math.max(0, s))));
  if (direction === "higher") {
    // Most normalized "higher-is-better" scores are in [0,1].
    if (s >= 0 && s <= 1) return s;
    // Fallback keeps UI stable for raw scores outside [0,1].
    return Math.max(0, Math.min(1, s / 100));
  }
  // Unknown direction: avoid overconfident visual ranking.
  return Math.max(0, Math.min(1, s >= 0 && s <= 1 ? s : 0));
}

function runMetricValue(run) {
  const candidates = [run && run.raw_score, run && run.dev_score];
  for (const value of candidates) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

function runMetricName(run) {
  return String((run && run.metric_name) || "score");
}

// ── Markdown Renderer ─────────────────────────────────────────
// Handles: headers, bold/italic, code blocks, inline code,
// lists, blockquotes, tables, horizontal rules, paragraphs.

function renderMarkdown(raw) {
  if (!raw || typeof raw !== "string") return "";

  // 1. Extract code blocks → placeholders (protect from HTML escaping)
  const codeBlocks = [];
  let t = raw.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const ph = `\x01${codeBlocks.length}\x01`;
    const cls = lang ? ` class="language-${esc(lang)}"` : "";
    codeBlocks.push(
      `<pre><code${cls}>${esc(code.trim())}</code></pre>`
    );
    return ph;
  });

  // 2. Extract inline code
  const inlineCodes = [];
  t = t.replace(/`([^`\n]+)`/g, (_, code) => {
    const ph = `\x02${inlineCodes.length}\x02`;
    inlineCodes.push(`<code>${esc(code)}</code>`);
    return ph;
  });

  // 3. HTML-escape remaining text (safe now — no real HTML inside)
  t = t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // 4. Process line-by-line
  const lines = t.split("\n");
  const out = [];
  let listOpen = false;
  let listType = "";

  const closeList = () => {
    if (listOpen) {
      out.push(listType === "ol" ? "</ol>" : "</ul>");
      listOpen = false;
      listType = "";
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Code block placeholder passthrough
    if (/^\x01\d+\x01$/.test(line.trim())) {
      closeList();
      out.push(line.trim());
      continue;
    }

    // Headings
    if (line.startsWith("###### ")) { closeList(); out.push(`<h6>${inlineFormat(line.slice(7))}</h6>`); continue; }
    if (line.startsWith("##### "))  { closeList(); out.push(`<h5>${inlineFormat(line.slice(6))}</h5>`); continue; }
    if (line.startsWith("#### "))   { closeList(); out.push(`<h4>${inlineFormat(line.slice(5))}</h4>`); continue; }
    if (line.startsWith("### "))    { closeList(); out.push(`<h3>${inlineFormat(line.slice(4))}</h3>`); continue; }
    if (line.startsWith("## "))     { closeList(); out.push(`<h3>${inlineFormat(line.slice(3))}</h3>`); continue; }
    if (line.startsWith("# "))      { closeList(); out.push(`<h2>${inlineFormat(line.slice(2))}</h2>`); continue; }

    // Blockquotes
    if (line.startsWith("&gt; ") || line.startsWith("> ")) {
      closeList();
      const txt = line.startsWith("&gt; ") ? line.slice(5) : line.slice(2);
      out.push(`<blockquote>${inlineFormat(txt)}</blockquote>`);
      continue;
    }

    // Horizontal rule
    if (/^-{3,}$/.test(line) || /^\*{3,}$/.test(line) || /^_{3,}$/.test(line)) {
      closeList();
      out.push("<hr>");
      continue;
    }

    // Unordered list
    const ulMatch = line.match(/^[•\-\*] (.+)/);
    if (ulMatch) {
      if (listType !== "ul") { closeList(); out.push("<ul>"); listOpen = true; listType = "ul"; }
      out.push(`<li>${inlineFormat(ulMatch[1])}</li>`);
      continue;
    }

    // Ordered list
    const olMatch = line.match(/^(\d+)\. (.+)/);
    if (olMatch) {
      if (listType !== "ol") { closeList(); out.push("<ol>"); listOpen = true; listType = "ol"; }
      out.push(`<li>${inlineFormat(olMatch[2])}</li>`);
      continue;
    }

    // Table row
    if (line.includes("|") && line.trim().startsWith("|")) {
      closeList();
      const isSep = /^\|[\s\-:]+\|/.test(line);
      if (isSep) continue;
      const cells = line.split("|").filter((_, ci) => ci > 0 && ci < line.split("|").length - 1);
      const tag = out[out.length - 1]?.startsWith("<table") ? "td" : "th";
      if (tag === "th" && !out.some(l => l.startsWith("<table"))) {
        out.push("<table class='data-table'><thead>");
      }
      if (tag === "th") {
        out.push("<tr>" + cells.map(c => `<th>${inlineFormat(c.trim())}</th>`).join("") + "</tr></thead><tbody>");
      } else {
        out.push("<tr>" + cells.map(c => `<td>${inlineFormat(c.trim())}</td>`).join("") + "</tr>");
      }
      continue;
    }

    // Empty line
    if (!line.trim()) {
      closeList();
      // close any open table
      const last = out[out.length - 1];
      if (last && !last.startsWith("<br") && !last.includes("</table>")) {
        // Don't add double breaks; just add one
      }
      out.push("<br>");
      continue;
    }

    // Regular paragraph line
    closeList();
    out.push(`<p>${inlineFormat(line)}</p>`);
  }

  closeList();

  // Close open tables
  let html = out.join("\n");
  if (html.includes("<tbody>") && !html.includes("</tbody>")) {
    html += "</tbody></table>";
  }

  // Restore inline codes and code blocks
  html = html.replace(/\x02(\d+)\x02/g, (_, i) => inlineCodes[+i]);
  html = html.replace(/\x01(\d+)\x01/g,  (_, i) => codeBlocks[+i]);

  return html;
}

function inlineFormat(t) {
  // Bold + italic combined
  t = t.replace(/\*\*\*([^*\n]+)\*\*\*/g, "<strong><em>$1</em></strong>");
  t = t.replace(/___([^_\n]+)___/g, "<strong><em>$1</em></strong>");
  // Bold
  t = t.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/__([^_\n]+)__/g, "<strong>$1</strong>");
  // Italic
  t = t.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  t = t.replace(/_([^_\n]+)_/g, "<em>$1</em>");
  // Strikethrough
  t = t.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  // Links
  t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return t;
}

// ── JSON Pretty Printer ───────────────────────────────────────

function renderJSON(obj) {
  try {
    const str = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
    const highlighted = str
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        (match) => {
          let cls = "json-num";
          if (/^"/.test(match)) {
            cls = /:$/.test(match) ? "json-key" : "json-str";
          } else if (/true|false/.test(match)) {
            cls = "json-bool";
          } else if (/null/.test(match)) {
            cls = "json-null";
          }
          return `<span class="${cls}">${match}</span>`;
        });
    return `<div class="json-pretty">${highlighted}</div>`;
  } catch {
    return `<pre class="json-pretty">${esc(String(obj))}</pre>`;
  }
}

// ── Application State ─────────────────────────────────────────

const S = {
  payload:         null,
  tab:             "overview",
  episodeFilter:   0,
  agentFilter:     "",
  search:          "",
  typeFilter:      new Set(),
  errFilter:       new Set(),
  liveMode:        true,
  sseConnected:    false,
  refreshTimer:    null,

  // LLM trace state
  llmRows:         [],
  llmDetailsById:  {},
  llmNextCursor:   "",
  llmTotal:        0,
  llmLoading:      false,
  llmOpenIds:      new Set(),
  llmBodyTabById:  {},    // id → "prompt"|"response"|"parsed"

  // Inspector state
  inspectorItem:   null,
  inspectorTab:    "summary",
  inspectorVisible: true,

  // RAG/retrieve state
  ragSubTab:       "rag_io",
  ragRows:         {},       // subtab → rows
  ragDetailsById:  {},       // id → detail

  // Pipeline state
  pipelineSubTab:  "action_trace",

  // Runs state
  runsSubTab:      "run_list",

  // Papers state
  papersSubTab:    "papers_list",

  // Resize
  sidebarWidth:    240,
  inspectorWidth:  420,
};

// ── DOM helpers ───────────────────────────────────────────────

const $ = id => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
};

function setHTML(id, html) {
  const e = $(id);
  if (e) e.innerHTML = html;
}

function show(id) { const e = $(id); if (e) e.style.display = ""; }
function hide(id) { const e = $(id); if (e) e.style.display = "none"; }

// ── Header Update ─────────────────────────────────────────────

function updateHeader(payload) {
  if (!payload) return;
  const meta  = payload.meta  || {};
  const team  = payload.team  || {};
  const audit = payload.audit || {};
  const srv   = payload.server || {};

  // Health dot
  const healthEl = $("health-dot");
  const healthTxt = $("health-text");
  const h = meta.health || "green";
  if (healthEl) {
    healthEl.className = `dot ${h === "red" ? "err" : h === "yellow" ? "warn" : "ok"}`;
  }
  if (healthTxt) healthTxt.textContent = `health: ${h}`;

  // Chips
  const alerts = toInt(meta.alert_count);
  const alertChip = $("alert-chip");
  if (alertChip) {
    alertChip.textContent = `alerts: ${alerts}`;
    alertChip.className = "header-chip" + (alerts > 0 ? (alerts >= 2 ? " err" : " warn") : "");
  }

  const eps = toInt(meta.episodes_count);
  setChip("episode-chip", `episode: ${eps || "—"}`);

  // Token usage
  const usage = audit.llm_usage_summary || {};
  const total = toInt(usage.total_tokens_est);
  setChip("token-chip", total ? `tokens: ${fmtK(total)}` : "tokens: —");

  const cost = (usage.cost_cny || {}).total;
  setChip("cost-chip", cost != null ? `cost: ${fmtCny(cost, 2)}` : "cost: —");

  // GPU
  const gpu = srv.gpu || {};
  if (gpu.available) {
    const gpuChip = $("gpu-chip");
    if (gpuChip) {
      gpuChip.style.display = "";
      gpuChip.textContent = `gpu: ${Math.round(gpu.util_percent)}% | ${Math.round(gpu.memory_ratio * 100)}% mem`;
    }
  }

  // Clock
  const tbEvents = (payload.taskboard || {}).events || [];
  if (tbEvents.length) {
    const last = tbEvents[tbEvents.length - 1];
    setChip("clock-chip", `clock: ${toTime(last.ts)}`);
  }

  // LLM badge
  const llmSummary = (audit.llm_io_summary || []);
  const badge = $("llm-badge");
  if (badge) {
    badge.textContent = llmSummary.length;
    badge.style.display = llmSummary.length ? "" : "none";
  }

  // Episode select
  const epSel = $("episode-select");
  if (epSel && payload.taskboard) {
    const ids = new Set();
    for (const ev of (payload.taskboard.events || [])) ids.add(toInt(ev.episode_id));
    const sorted = Array.from(ids).filter(x => x > 0).sort((a, b) => a - b);
    const current = epSel.value;
    epSel.innerHTML = `<option value="0">All Episodes</option>` +
      sorted.map(id => `<option value="${id}">Episode ${id}</option>`).join("");
    if (current) epSel.value = current;
  }
}

function setChip(id, text) {
  const e = $(id);
  if (e) e.textContent = text;
}

// ── Sidebar ───────────────────────────────────────────────────

function renderSidebar(payload) {
  if (!payload) return;
  const tb = payload.taskboard || {};
  const agents = tb.agents || [];

  // Agents
  const agentList = $("agent-list");
  if (agentList) {
    if (!agents.length) {
      agentList.innerHTML = `<div class="muted" style="padding:6px;font-size:12px;">No agent data</div>`;
    } else {
      agentList.innerHTML = agents.map(a => `
        <div class="agent-row ${S.agentFilter === a.agent_id ? "active" : ""}"
             onclick="toggleAgentFilter('${esc(a.agent_id)}')">
          <span class="agent-dot ${a.status}"></span>
          <span class="agent-id">${esc(a.agent_id)}</span>
          <span class="agent-meta">${a.completed}✓ ${a.errors ? `<span class="err">${a.errors}✗</span>` : ""}</span>
        </div>`).join("");
    }
    $("agent-count").textContent = `${agents.length}`;
  }

  // Type filters
  const stageKeys = ["prepare", "profile", "literature", "read", "hypothesize",
                     "experiment", "review", "write", "replicate"];
  const phaseData = tb.phase_counts || {};
  $("type-filters").innerHTML = stageKeys.map(s => {
    const counts = phaseData[s] || {};
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (!total && !S.typeFilter.has(s)) return "";
    return `<span class="chip ${S.typeFilter.has(s) ? "active" : ""}"
                  onclick="toggleTypeFilter('${s}')">${s} ${total ? `<small>${total}</small>` : ""}</span>`;
  }).join("");

  // Alerts
  const alerts = (payload.meta || {}).alerts || [];
  const alertSection = $("alert-section");
  if (alertSection) {
    if (alerts.length) {
      alertSection.style.display = "";
      $("alert-count-mini").textContent = alerts.length;
      $("alert-list").innerHTML = alerts.map(a => `
        <div class="alert-row ${a.level}">
          <span class="alert-icon">${a.level === "error" ? "✗" : "⚠"}</span>
          <span class="alert-text">${esc(a.key)}: <strong>${a.value}</strong></span>
        </div>`).join("");
    } else {
      alertSection.style.display = "none";
    }
  }
}

function toggleAgentFilter(id) {
  S.agentFilter = S.agentFilter === id ? "" : id;
  renderSidebar(S.payload);
  renderCurrentTab();
}

function toggleTypeFilter(s) {
  if (S.typeFilter.has(s)) S.typeFilter.delete(s);
  else S.typeFilter.add(s);
  renderSidebar(S.payload);
  renderCurrentTab();
}

// ── Overview Tab ──────────────────────────────────────────────

function renderOverview(payload) {
  if (!payload) return;
  const meta  = payload.meta  || {};
  const team  = payload.team  || {};
  const audit = payload.audit || {};
  const tb    = payload.taskboard || {};
  const trace = payload.action_trace || [];

  // Health metrics
  const sc = meta.state_counts || {};
  $("ov-health-metrics").innerHTML = [
    { val: sc.claimed    || 0,  lbl: "Claimed"   },
    { val: sc.completed  || 0,  lbl: "Completed" },
    { val: sc.released   || 0,  lbl: "Released"  },
    { val: meta.alert_count || 0, lbl: "Alerts",
      extra: meta.alert_count ? " err" : " ok" },
  ].map(m => `
    <div class="ov-metric">
      <div class="ov-metric-val${m.extra || ""}">${fmtInt(m.val)}</div>
      <div class="ov-metric-lbl">${m.lbl}</div>
    </div>`).join("");

  // Team metrics
  const summary = team.summary || {};
  $("ov-team-sub").textContent = summary.episodes_count
    ? `${summary.episodes_count} episodes` : "";
  $("ov-team-metrics").innerHTML = [
    { val: fmtNum(summary.final_team_fitness, 3),      lbl: "Team Fitness" },
    { val: fmtNum(summary.final_publishable_rate, 2),  lbl: "Publishable %" },
    { val: fmtNum(summary.final_replication_pass, 2),  lbl: "Replication Pass" },
    { val: fmtNum(summary.final_complete_per_claim, 2), lbl: "Complete/Claim" },
  ].map(m => `
    <div class="ov-metric">
      <div class="ov-metric-val">${m.val}</div>
      <div class="ov-metric-lbl">${m.lbl}</div>
    </div>`).join("");

  // Pipeline flow
  const stages = ["prepare", "profile", "literature", "read", "hypothesize",
                  "experiment", "review", "write", "replicate"];
  const phase = tb.phase_counts || {};
  $("ov-pipeline-flow").innerHTML = stages.map((s, i) => {
    const c = phase[s] || {};
    const ok  = toInt(c.complete);
    const run = toInt(c.claim) - ok - toInt(c.release);
    const fail= toInt(c.release);
    const badges = [
      ok   ? `<span class="pf-count ok">✓${ok}</span>` : "",
      run > 0 ? `<span class="pf-count run">⟳${run}</span>` : "",
      fail ? `<span class="pf-count fail">✗${fail}</span>` : "",
    ].filter(Boolean).join("");
    return `
      <div class="pf-stage">
        ${i > 0 ? '<span class="pf-arrow">→</span>' : ""}
        <div class="pf-box">
          <div class="pf-name">${s}</div>
          <div class="pf-counts">${badges || '<span class="muted" style="font-size:10px">—</span>'}</div>
        </div>
      </div>`;
  }).join("");

  // Token usage
  const usage = audit.llm_usage_summary || {};
  const costs = usage.cost_cny || {};
  $("ov-token-metrics").innerHTML = [
    { val: fmtK(usage.input_tokens_est  || 0),  lbl: "Input Tokens"  },
    { val: fmtK(usage.output_tokens_est || 0),  lbl: "Output Tokens" },
    { val: fmtCny(costs.total  || 0, 2),         lbl: "Total Cost CNY"},
    { val: fmtInt(usage.sample_rows || 0),        lbl: "LLM Calls"    },
  ].map(m => `
    <div class="ov-metric">
      <div class="ov-metric-val">${m.val}</div>
      <div class="ov-metric-lbl">${m.lbl}</div>
    </div>`).join("");

  // Recent actions
  const recentActions = trace.slice(0, 12);
  $("ov-action-list").innerHTML = recentActions.length
    ? recentActions.map(a => `
        <div class="ov-action-row">
          <span class="ok-dot ${a.status === "success" ? "ok" : "err"}"></span>
          <span class="mono" style="color:var(--teal);min-width:60px">${esc(a.agent_id)}</span>
          <span style="min-width:90px;color:var(--text2)">${esc(a.action)}</span>
          <span class="muted" style="font-size:11px;margin-left:auto">${toTime(a.ts)}</span>
        </div>`).join("")
    : `<div class="muted" style="padding:10px;font-size:12px;">No action trace data</div>`;
}

// ── LLM Trace Tab ─────────────────────────────────────────────

function renderLLMTab() {
  // Populate agent select from payload
  const agents = (S.payload?.taskboard?.agents || []).map(a => a.agent_id);
  const llmSummary = (S.payload?.audit?.llm_io_summary || []);
  const llmAgents = new Set(llmSummary.map(r => r.agent_id).filter(Boolean));
  agents.forEach(a => llmAgents.add(a));

  const agentSel = $("llm-agent-select");
  if (agentSel) {
    const prev = agentSel.value;
    agentSel.innerHTML = '<option value="">All Agents</option>' +
      Array.from(llmAgents).sort().map(a => `<option value="${esc(a)}">${esc(a)}</option>`).join("");
    if (prev) agentSel.value = prev;
  }

  // Populate action select
  const llmActions = new Set(llmSummary.map(r => r.action).filter(Boolean));
  const actionSel = $("llm-action-select");
  if (actionSel) {
    const prev = actionSel.value;
    actionSel.innerHTML = '<option value="">All Actions</option>' +
      Array.from(llmActions).sort().map(a => `<option value="${esc(a)}">${esc(a)}</option>`).join("");
    if (prev) actionSel.value = prev;
  }

  // If we have loaded rows, render them
  renderLLMList();
}

function renderLLMList() {
  const container = $("llm-list");
  if (!container) return;

  if (S.llmLoading) {
    container.innerHTML = `<div class="llm-empty"><div class="spinning" style="font-size:24px">↻</div><div>Loading…</div></div>`;
    return;
  }

  const rows = S.llmRows;
  const search = ($("llm-search")?.value || "").toLowerCase();

  const filtered = rows.filter(r => {
    if (search) {
      const text = `${r.agent_id} ${r.action} ${r.prompt_preview} ${r.response_preview}`.toLowerCase();
      if (!text.includes(search)) return false;
    }
    return true;
  });

  if (!filtered.length) {
    container.innerHTML = `
      <div class="llm-empty">
        <div class="llm-empty-icon">⚡</div>
        <div>${rows.length ? "No results match filters" : "Click <strong>Load / Refresh</strong> to fetch LLM trace records"}</div>
        <div class="muted" style="margin-top:6px;font-size:12px">Full prompts and responses with markdown rendering</div>
      </div>`;
    $("llm-footer").style.display = "none";
    return;
  }

  container.innerHTML = filtered.map(r => renderLLMRecord(r)).join("");

  // Re-bind open state
  for (const id of S.llmOpenIds) {
    const el = document.querySelector(`.llm-record[data-id="${CSS.escape(id)}"]`);
    if (el) {
      el.classList.add("open");
      // Set correct body tab
      const bodyTab = S.llmBodyTabById[id] || "prompt";
      el.querySelectorAll(".llm-body-tab").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.tab === bodyTab);
      });
      el.querySelectorAll(".llm-section").forEach(sec => {
        sec.classList.toggle("active", sec.dataset.section === bodyTab);
      });
    }
  }

  // Stats bar
  const llmStats = $("llm-stats-bar");
  if (llmStats) {
    const totalTok = rows.reduce((s, r) => s + (r.total_tokens_est || 0), 0);
    const inputTok = rows.reduce((s, r) => s + (r.prompt_tokens_est || 0), 0);
    const outTok   = rows.reduce((s, r) => s + (r.completion_tokens_est || 0), 0);
    const okCount  = rows.filter(r => r.ok_status).length;
    llmStats.innerHTML = `
      <span class="llm-stat"><span class="llm-stat-val">${filtered.length}</span><span class="llm-stat-lbl">records</span></span>
      <span class="llm-stat"><span class="llm-stat-val ok">${okCount}</span><span class="llm-stat-lbl">ok</span></span>
      <span class="llm-stat"><span class="llm-stat-val err">${rows.length - okCount}</span><span class="llm-stat-lbl">fail</span></span>
      <span class="llm-stat"><span class="llm-stat-val">${fmtK(inputTok)}</span><span class="llm-stat-lbl">input tok</span></span>
      <span class="llm-stat"><span class="llm-stat-val">${fmtK(outTok)}</span><span class="llm-stat-lbl">output tok</span></span>
      <span class="llm-stat"><span class="llm-stat-val">${fmtK(totalTok)}</span><span class="llm-stat-lbl">total tok</span></span>
      ${S.llmTotal > rows.length ? `<span class="llm-stat muted">(showing ${rows.length} of ${S.llmTotal})</span>` : ""}
    `;
  }

  // Footer
  if (S.llmNextCursor && S.llmTotal > rows.length) {
    $("llm-footer").style.display = "";
    $("llm-more-label").textContent = `${rows.length} / ${S.llmTotal} loaded`;
  } else {
    $("llm-footer").style.display = "none";
  }
}

function renderLLMRecord(r) {
  const statusCls = r.ok_status ? "ok" : "fail";
  const detail = S.llmDetailsById[r.id] || null;
  const bodyTab = S.llmBodyTabById[r.id] || "prompt";

  const promptText    = detail?.prompt        || "";
  const responseText  = detail?.raw_response  || "";
  const parsedJson    = detail?.parsed_json;

  const hasDetail = !!promptText || !!responseText;

  // Body tabs
  const bodyTabs = [
    { id: "prompt",   label: "↑ Prompt",   available: true },
    { id: "response", label: "↓ Response", available: true },
    { id: "parsed",   label: "{ } Parsed", available: !!parsedJson },
  ].filter(t => t.available);

  const bodyTabsHTML = bodyTabs.map(t =>
    `<button class="llm-body-tab ${bodyTab === t.id ? "active" : ""}"
             data-tab="${t.id}" onclick="switchLLMBodyTab('${esc(r.id)}', '${t.id}', event)">
       ${t.label}
     </button>`
  ).join("");

  // Prompt content
  const promptHTML = promptText
    ? `<div class="md-body">${renderMarkdown(promptText)}</div>`
    : `<div class="muted" style="padding:20px;text-align:center">
         Click <strong>Load / Refresh</strong> to fetch full prompt content
       </div>`;

  // Response content
  let responseHTML;
  if (responseText) {
    // Try to detect if it's JSON
    const trimmed = responseText.trim();
    if ((trimmed.startsWith("{") || trimmed.startsWith("[")) && trimmed.endsWith("}") || trimmed.endsWith("]")) {
      try {
        const parsed = JSON.parse(responseText);
        responseHTML = renderJSON(parsed);
      } catch {
        responseHTML = `<div class="md-body">${renderMarkdown(responseText)}</div>`;
      }
    } else {
      responseHTML = `<div class="md-body">${renderMarkdown(responseText)}</div>`;
    }
  } else {
    responseHTML = `<div class="muted" style="padding:20px;text-align:center">
      No response text. Click Load to fetch full details.
    </div>`;
  }

  // Parsed JSON content
  const parsedHTML = parsedJson
    ? renderJSON(parsedJson)
    : `<div class="muted" style="padding:20px;text-align:center">No parsed JSON</div>`;

  return `
    <div class="llm-record ${statusCls}" data-id="${esc(r.id)}" id="llm-rec-${esc(r.id)}">
      <div class="llm-rec-header" onclick="toggleLLMRecord('${esc(r.id)}')">
        <span class="lbadge agent">${esc(r.agent_id || "—")}</span>
        <span class="lbadge action">${esc(r.action || "—")}</span>
        ${r.episode_id ? `<span class="lbadge episode">ep${r.episode_id}</span>` : ""}
        ${r.tick != null ? `<span class="lbadge tick">T${r.tick}</span>` : ""}
        <span class="lbadge ${r.ok_status ? "ok-s" : "fail-s"}">${r.ok_status ? "✓ ok" : "✗ fail"}</span>
        <span class="llm-rec-tokens">${fmtK(r.prompt_tokens_est || 0)}↑ ${fmtK(r.completion_tokens_est || 0)}↓ tok</span>
        <span class="llm-rec-time">${toTime(r.ts)}</span>
        <span class="llm-rec-chevron">▼</span>
      </div>

      <div class="llm-previews">
        <div class="llm-preview-box">
          <div class="llm-preview-label">↑ Prompt preview</div>
          <div class="llm-preview-text">${esc(r.prompt_preview || "—")}</div>
        </div>
        <div class="llm-preview-box">
          <div class="llm-preview-label">↓ Response preview</div>
          <div class="llm-preview-text">${esc(r.response_preview || "—")}</div>
        </div>
      </div>

      <div class="llm-rec-body">
        <div class="llm-rec-body-tabs">
          ${bodyTabsHTML}
          <div style="flex:1"></div>
          <span class="muted" style="font-size:11px;padding:8px 14px;align-self:center">
            ${fmtInt(r.prompt_chars)} → ${fmtInt(r.response_chars)} chars
          </span>
        </div>
        <div class="llm-rec-content">
          <div class="llm-section ${bodyTab === "prompt"   ? "active" : ""}" data-section="prompt">
            <div class="llm-section-label">
              ↑ PROMPT
              <span class="llm-section-meta">${fmtK(r.prompt_tokens_est || 0)} tokens est. · ${fmtInt(r.prompt_chars)} chars</span>
            </div>
            ${promptHTML}
          </div>
          <div class="llm-section ${bodyTab === "response" ? "active" : ""}" data-section="response">
            <div class="llm-section-label">
              ↓ RESPONSE
              <span class="llm-section-meta">${fmtK(r.completion_tokens_est || 0)} tokens est. · ${fmtInt(r.response_chars)} chars${r.reason ? ` · ${esc(r.reason)}` : ""}</span>
            </div>
            ${responseHTML}
          </div>
          <div class="llm-section ${bodyTab === "parsed"   ? "active" : ""}" data-section="parsed">
            <div class="llm-section-label">{ } PARSED JSON</div>
            ${parsedHTML}
          </div>
        </div>
      </div>
    </div>`;
}

function toggleLLMRecord(id) {
  const recEl = document.querySelector(`.llm-record[data-id="${CSS.escape(id)}"]`);
  if (!recEl) return;

  const isOpen = recEl.classList.contains("open");
  if (isOpen) {
    recEl.classList.remove("open");
    S.llmOpenIds.delete(id);
  } else {
    recEl.classList.add("open");
    S.llmOpenIds.add(id);
    // Load full detail if not yet loaded
    if (!S.llmDetailsById[id]) {
      loadLLMDetail(id);
    }
    // Set inspector
    const row = S.llmRows.find(r => r.id === id);
    if (row) showLLMInspector(row);
  }
}

function switchLLMBodyTab(id, tab, evt) {
  if (evt) evt.stopPropagation();
  S.llmBodyTabById[id] = tab;
  const recEl = document.querySelector(`.llm-record[data-id="${CSS.escape(id)}"]`);
  if (!recEl) return;
  recEl.querySelectorAll(".llm-body-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  recEl.querySelectorAll(".llm-section").forEach(sec => {
    sec.classList.toggle("active", sec.dataset.section === tab);
  });
}

async function loadLLMList(append = false) {
  if (S.llmLoading) return;
  S.llmLoading = true;
  renderLLMList();

  const episode = $("llm-ep-select")?.value || "current";
  const agent   = $("llm-agent-select")?.value || "";
  const action  = $("llm-action-select")?.value || "";
  const search  = $("llm-search")?.value || "";

  const params = new URLSearchParams();
  params.set("episode", episode);
  params.set("limit", "50");
  if (agent)  params.set("agent", agent);
  if (action) params.set("action", action);
  if (search) params.set("task_id", search);
  if (append && S.llmNextCursor) params.set("cursor", S.llmNextCursor);

  try {
    const res = await fetch(`/api/audit/llm?${params}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "unknown error");

    const newRows = data.rows || [];
    // Merge detail records
    for (const [id, detail] of Object.entries(data.details_by_id || {})) {
      S.llmDetailsById[id] = detail;
    }

    if (append) {
      S.llmRows.push(...newRows);
    } else {
      S.llmRows = newRows;
      S.llmOpenIds.clear();
    }
    S.llmNextCursor = data.next_cursor || "";
    S.llmTotal = data.total || newRows.length;
  } catch (e) {
    $("llm-list").innerHTML = `<div class="llm-empty err">Error loading LLM data: ${esc(String(e))}</div>`;
  } finally {
    S.llmLoading = false;
    renderLLMList();
  }
}

async function loadLLMDetail(id) {
  try {
    const res = await fetch(`/api/audit/detail?kind=llm_io&id=${encodeURIComponent(id)}`, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    if (data.ok && data.record) {
      S.llmDetailsById[id] = data.record;
      // Re-render just the expanded record
      const recEl = document.querySelector(`.llm-record[data-id="${CSS.escape(id)}"]`);
      if (recEl && recEl.classList.contains("open")) {
        const row = S.llmRows.find(r => r.id === id);
        if (row) {
          const tmp = document.createElement("div");
          tmp.innerHTML = renderLLMRecord(row);
          const newEl = tmp.firstElementChild;
          newEl.classList.add("open");
          // Restore body tab
          const bodyTab = S.llmBodyTabById[id] || "prompt";
          newEl.querySelectorAll(".llm-body-tab").forEach(btn =>
            btn.classList.toggle("active", btn.dataset.tab === bodyTab)
          );
          newEl.querySelectorAll(".llm-section").forEach(sec =>
            sec.classList.toggle("active", sec.dataset.section === bodyTab)
          );
          recEl.replaceWith(newEl);
        }
      }
    }
  } catch { /* ignore */ }
}

// ── RAG & Retrieve Tab ────────────────────────────────────────

function renderRAGTab(subtab) {
  const content = $("rag-content");
  if (!content) return;

  const ep = S.episodeFilter || 0;
  const payload = S.payload;
  if (!payload) return;

  const audit = payload.audit || {};
  let rows = [];
  let columns = [];

  if (subtab === "rag_io") {
    rows = audit.rag_io_summary || [];
    columns = [
      { key: "ts",              label: "Time",       fmt: r => toTime(r.ts) },
      { key: "agent_id",        label: "Agent",      cls: "mono" },
      { key: "action",          label: "Action",     cls: "mono" },
      { key: "operation",       label: "Operation",  cls: "mono" },
      { key: "retrieval_mode",  label: "Mode" },
      { key: "status",          label: "Status",     fmt: r => statusTag(r.status) },
      { key: "selected_count",  label: "Results",    cls: "mono" },
      { key: "query_preview",   label: "Query",      cls: "ellipsis" },
    ];
  } else if (subtab === "retrieve_pipeline") {
    rows = audit.retrieve_pipeline_summary || [];
    columns = [
      { key: "ts",       label: "Time",    fmt: r => toTime(r.ts)  },
      { key: "agent_id", label: "Agent",   cls: "mono" },
      { key: "phase",    label: "Phase",   cls: "mono" },
      { key: "ok",       label: "OK",      fmt: r => r.ok ? `<span class="ok">✓</span>` : `<span class="err">✗</span>` },
      { key: "source",   label: "Source"  },
      { key: "reward",   label: "Reward",  fmt: r => r.reward != null ? fmtNum(r.reward, 3) : "—" },
    ];
  } else if (subtab === "retrieve_guardrail") {
    rows = audit.retrieve_guardrail_summary || [];
    columns = [
      { key: "ts",                label: "Time",       fmt: r => toTime(r.ts) },
      { key: "agent_id",          label: "Agent",      cls: "mono" },
      { key: "source",            label: "Source"     },
      { key: "level",             label: "Level"      },
      { key: "degraded",          label: "Degraded",  fmt: r => r.degraded ? `<span class="warn">⚠ yes</span>` : "no" },
      { key: "citation_coverage", label: "Citation",  fmt: r => fmtNum(r.citation_coverage, 2) },
    ];
  } else if (subtab === "retrieve_evidence") {
    rows = audit.retrieve_evidence_summary || [];
    columns = [
      { key: "ts",             label: "Time",   fmt: r => toTime(r.ts) },
      { key: "agent_id",       label: "Agent",  cls: "mono" },
      { key: "rag_status",     label: "Status", fmt: r => statusTag(r.rag_status) },
      { key: "evidence_count", label: "Count",  cls: "mono" },
    ];
  }

  if (ep > 0) rows = rows.filter(r => toInt(r.episode_id) === ep);
  if (S.agentFilter) rows = rows.filter(r => r.agent_id === S.agentFilter);

  content.innerHTML = renderDataTable(rows, columns, r => showGenericInspector("rag", r));
  bindTableClicks(content);
}

function statusTag(s) {
  const ok = ["ok","success","found","pass","valid","done"].includes(String(s||"").toLowerCase());
  return `<span class="tag ${ok ? "ok" : "fail"}">${esc(s||"—")}</span>`;
}

// ── Pipeline Tab ──────────────────────────────────────────────

function renderPipelineTab(subtab) {
  const content = $("pipeline-content");
  if (!content) return;
  const payload = S.payload;
  if (!payload) return;

  const ep = S.episodeFilter || 0;
  let rows = [];
  let columns = [];

  if (subtab === "action_trace") {
    rows = (payload.action_trace || []).slice().reverse();
    if (ep > 0)         rows = rows.filter(r => toInt(r.episode_id) === ep);
    if (S.agentFilter)  rows = rows.filter(r => r.agent_id === S.agentFilter);
    columns = [
      { key: "ts",       label: "Time",    fmt: r => toTime(r.ts)   },
      { key: "agent_id", label: "Agent",   cls: "mono" },
      { key: "action",   label: "Action",  cls: "mono" },
      { key: "status",   label: "Status",  fmt: r => `<span class="${r.status === "success" ? "ok" : "err"}">${esc(r.status)}</span>` },
      { key: "task_id",  label: "Task",    cls: "mono" },
      { key: "summary",  label: "Summary", cls: "ellipsis" },
    ];
  } else if (subtab === "code_loops") {
    rows = (payload.code_loops || []).slice().reverse();
    if (ep > 0)        rows = rows.filter(r => toInt(r.episode_id) === ep);
    if (S.agentFilter) rows = rows.filter(r => r.agent_id === S.agentFilter);
    columns = [
      { key: "ts",                  label: "Time",        fmt: r => toTime(r.ts) },
      { key: "agent_id",            label: "Agent",       cls: "mono" },
      { key: "task_name",           label: "Task",        cls: "ellipsis" },
      { key: "attempt_count",       label: "Attempts",    cls: "mono" },
      { key: "best_dev_score_norm", label: "Best Score",  fmt: r => r.best_dev_score_norm != null ? fmtNum(r.best_dev_score_norm, 3) : "—" },
    ];
  } else if (subtab === "precondition") {
    rows = (payload.precondition_gates || []).slice().reverse();
    if (ep > 0)        rows = rows.filter(r => toInt(r.episode_id) === ep);
    if (S.agentFilter) rows = rows.filter(r => r.agent_id === S.agentFilter);
    columns = [
      { key: "ts",       label: "Time",   fmt: r => toTime(r.ts) },
      { key: "agent_id", label: "Agent",  cls: "mono" },
      { key: "action",   label: "Action", cls: "mono" },
      { key: "phase",    label: "Phase",  cls: "mono" },
      { key: "failures", label: "Failures", fmt: r => {
        const f = r.failures || [];
        return f.length ? `<span class="err">${f.length} failure${f.length > 1 ? "s" : ""}</span>` : `<span class="ok">none</span>`;
      }},
    ];
  } else if (subtab === "eval_failures") {
    rows = (payload.eval_failures || []).slice().reverse();
    if (ep > 0) rows = rows.filter(r => toInt(r.episode_id) === ep);
    columns = [
      { key: "ts",         label: "Time",     fmt: r => toTime(r.ts) },
      { key: "stage",      label: "Stage",    cls: "mono" },
      { key: "error_type", label: "Error",    fmt: r => `<span class="err">${esc(r.error_type)}</span>` },
      { key: "rc",         label: "Exit",     cls: "mono" },
      { key: "task_name",  label: "Task",     cls: "ellipsis" },
    ];
  }

  content.innerHTML = renderDataTable(rows, columns, r => showGenericInspector("pipeline", r));
  bindTableClicks(content);
}

// ── Runs & Code Tab ───────────────────────────────────────────

function renderRunsTab(subtab) {
  const content = $("runs-content");
  if (!content) return;
  const payload = S.payload;
  if (!payload) return;

  const runs = (payload.runs || {}).runs || [];
  const ep = S.episodeFilter || 0;

  if (subtab === "run_list") {
    const filtered = ep > 0 ? runs.filter(r => toInt(r.episode_id) === ep) : runs;
    content.innerHTML = `
      <div class="card-grid">
        ${filtered.map(r => {
          const execOk = typeof r.exec_ok === "boolean" ? r.exec_ok : (r.exit_code === 0 || r.exit_code === null);
          const scientificOk = typeof r.scientific_ok === "boolean"
            ? r.scientific_ok
            : (typeof r.evidence_ok === "boolean" ? r.evidence_ok : false);
          const publishReady = typeof r.publish_ready === "boolean"
            ? r.publish_ready
            : (typeof r.evidence_ok === "boolean" ? r.evidence_ok : false);
          const preflightOk = typeof r.preflight_ok === "boolean" ? r.preflight_ok : null;
          const cardState = scientificOk ? "ok" : execOk ? "warn" : "fail";
          const statusText = scientificOk ? "✓ scientific ok" : execOk ? "⚠ scientific pending" : "✗ exec fail";
          const score = runMetricValue(r);
          const metricName = runMetricName(r);
          const direction = inferScoreDirection(r);
          const quality = score != null ? scoreQuality(score, direction) : null;
          const pct = quality != null ? Math.min(100, Math.max(0, quality * 100)) : null;
          const dirText = direction === "lower" ? "lower better" : direction === "higher" ? "higher better" : "direction unknown";
          const scientificReason = r.scientific_reason || r.evidence_reason || "";
          return `
            <div class="item-card ${cardState}" onclick="showRunInspector(${JSON.stringify(esc(r.run_key))})">
              <div class="item-card-head">
                <span class="item-card-title">${esc(r.run_id)}</span>
                <span class="tag ${scientificOk ? "ok" : execOk ? "warn" : "fail"}">${statusText}</span>
              </div>
              <div class="item-card-meta">
                ep${r.episode_id} · ${esc(r.executor || "?")} · ${r.duration_s?.toFixed(1) || "?"}s
                ${r.timed_out ? '<span class="warn">⏱ timeout</span>' : ""}
              </div>
              <div style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px">
                <span class="tag ${execOk ? "ok" : "fail"}">exec ${execOk ? "ok" : "fail"}</span>
                <span class="tag ${scientificOk ? "ok" : "fail"}">scientific ${scientificOk ? "ok" : "fail"}</span>
                <span class="tag ${publishReady ? "ok" : "warn"}">publish ${publishReady ? "ready" : "pending"}</span>
                ${preflightOk == null ? "" : `<span class="tag ${preflightOk ? "info" : "warn"}">preflight ${preflightOk ? "ok" : "fail"}</span>`}
              </div>
              ${r.error_signature ? `<div class="muted" style="font-size:11px;margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(r.error_signature.slice(0, 80))}</div>` : ""}
              ${pct != null ? `
                <div class="score-bar" style="margin-top:8px" title="${metricName}: ${score}">
                  <div class="score-bar-fill" style="width:${pct}%;background:${scientificOk ? "var(--ok)" : execOk ? "var(--warn)" : "var(--err)"}"></div>
                </div>
                <div style="font-size:11px;color:var(--text2);margin-top:4px">${esc(metricName)}: ${fmtNum(score, 4)} · ${dirText}</div>` : ""}
              ${scientificReason ? `<div class="muted" style="font-size:11px;margin-top:4px">reason: ${esc(scientificReason)}</div>` : ""}
            </div>`;
        }).join("") || '<div class="io-loading">No run data</div>'}
      </div>`;
  } else if (subtab === "run_stats") {
    const filtered = ep > 0 ? runs.filter(r => toInt(r.episode_id) === ep) : runs;
    const total = filtered.length;
    const execOk = filtered.filter(r => typeof r.exec_ok === "boolean" ? r.exec_ok : (r.exit_code === 0 || r.exit_code == null)).length;
    const scientificOk = filtered.filter(r =>
      typeof r.scientific_ok === "boolean" ? r.scientific_ok : !!r.evidence_ok
    ).length;
    const publishReady = filtered.filter(r =>
      typeof r.publish_ready === "boolean" ? r.publish_ready : !!r.evidence_ok
    ).length;
    const fail = total - execOk;
    const scored = filtered.filter(r => runMetricValue(r) != null);
    const lowerVotes = scored.filter(r => inferScoreDirection(r) === "lower").length;
    const higherVotes = scored.filter(r => inferScoreDirection(r) === "higher").length;
    const dominantDirection = lowerVotes > higherVotes ? "lower" : higherVotes > lowerVotes ? "higher" : "unknown";
    const scores = scored.map(r => Number(runMetricValue(r)));
    const bestScore = scores.length
      ? (dominantDirection === "lower" ? Math.min(...scores) : Math.max(...scores))
      : null;
    const directionText = dominantDirection === "lower"
      ? "lower better"
      : dominantDirection === "higher"
      ? "higher better"
      : "unknown";
    content.innerHTML = `
      <div style="padding:8px">
        <div class="section-title">Run Statistics</div>
        <div class="ov-metrics" style="margin-bottom:20px">
          ${[
            { val: total, lbl: "Total Runs" },
            { val: execOk,    lbl: "Exec OK", cls: " ok" },
            { val: scientificOk, lbl: "Scientific OK", cls: scientificOk ? " ok" : "" },
            { val: publishReady, lbl: "Publish Ready", cls: publishReady ? " ok" : "" },
            { val: fail,  lbl: "Failed",     cls: " err" },
            { val: bestScore != null ? fmtNum(bestScore, 4) : "—", lbl: "Best Raw Metric" },
            { val: directionText, lbl: "Score Direction" },
          ].map(m => `
            <div class="ov-metric">
              <div class="ov-metric-val${m.cls || ""}">${m.val}</div>
              <div class="ov-metric-lbl">${m.lbl}</div>
            </div>`).join("")}
        </div>
      </div>`;
  }
}

function showRunInspector(key) {
  const run = ((S.payload?.runs || {}).runs || []).find(r => r.run_key === key);
  if (!run) return;
  showGenericInspector("run", run);
}

// ── Papers Tab ────────────────────────────────────────────────

function renderPapersTab(subtab) {
  const content = $("papers-content");
  if (!content) return;
  const payload = S.payload;
  if (!payload) return;

  const ep = S.episodeFilter || 0;

  if (subtab === "papers_list") {
    let papers = payload.papers || [];
    if (ep > 0) papers = papers.filter(p => toInt(p.episode_id) === ep);
    content.innerHTML = `
      <div class="card-grid">
        ${papers.map(p => `
          <div class="item-card ${p.publishable ? "ok" : ""}" onclick="showGenericInspector('paper', ${JSON.stringify(esc(JSON.stringify(p)))})">
            <div class="item-card-head">
              <span class="item-card-title">${esc(p.paper_id || "—")}</span>
              <span class="tag ${p.publishable ? "ok" : "info"}">${p.publishable ? "publishable" : "draft"}</span>
            </div>
            <div class="item-card-meta">
              ep${p.episode_id} · ${esc(p.agent_id)} · fitness: ${fmtNum(p.fitness, 3)}
            </div>
            ${p.replication_ok ? `<div class="ok" style="font-size:11px;margin-top:4px">✓ replication ok</div>` : ""}
            <div class="score-bar" style="margin-top:8px">
              <div class="score-bar-fill" style="width:${Math.min(100, p.fitness * 100)}%"></div>
            </div>
          </div>`).join("") || '<div class="io-loading">No papers data</div>'}
      </div>`;
  } else if (subtab === "evidence_cards") {
    let evidence = payload.evidence_cards || [];
    if (ep > 0) evidence = evidence.filter(e => toInt(e.episode_id) === ep);
    if (S.agentFilter) evidence = evidence.filter(e => e.agent_id === S.agentFilter);
    const columns = [
      { key: "ts",          label: "Time",    fmt: r => toTime(r.ts) },
      { key: "agent_id",    label: "Agent",   cls: "mono" },
      { key: "kind",        label: "Kind",    cls: "mono" },
      { key: "task_id",     label: "Task",    cls: "mono" },
      { key: "source",      label: "Source",  cls: "ellipsis" },
    ];
    content.innerHTML = renderDataTable(evidence, columns, r => showGenericInspector("evidence", r));
    bindTableClicks(content);
  }
}

// ── Generic Data Table ────────────────────────────────────────

// Table row store — maps tableId → rows array for post-render click binding
const _tableStore = {};
let _tableIdSeq = 0;

function renderDataTable(rows, columns, onRowClick) {
  if (!rows.length) return `<div class="io-loading">No data available</div>`;

  const tid = `tbl_${++_tableIdSeq}`;
  _tableStore[tid] = { rows, onRowClick };

  return `
    <div style="overflow-x:auto">
      <table class="data-table" data-tid="${tid}">
        <thead>
          <tr>${columns.map(c => `<th>${esc(c.label)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows.map((r, i) => `
            <tr data-row="${i}" style="cursor:pointer">
              ${columns.map(c => {
                const val = c.fmt ? c.fmt(r) : esc(String(r[c.key] ?? "—"));
                const style = c.cls === "ellipsis"
                  ? "max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" : "";
                return `<td class="${c.cls || ""}" style="${style}">${val}</td>`;
              }).join("")}
            </tr>`).join("")}
        </tbody>
      </table>
    </div>`;
}

// Bind click events to all newly rendered data tables inside a container
function bindTableClicks(containerEl) {
  if (!containerEl) return;
  containerEl.querySelectorAll("table[data-tid]").forEach(tbl => {
    const store = _tableStore[tbl.dataset.tid];
    if (!store) return;
    tbl.querySelectorAll("tbody tr[data-row]").forEach(tr => {
      const idx = parseInt(tr.dataset.row, 10);
      if (!isNaN(idx) && store.rows[idx]) {
        tr.addEventListener("click", () => store.onRowClick(store.rows[idx]));
      }
    });
    // Remove tid so we don't double-bind
    tbl.removeAttribute("data-tid");
  });
}

// Legacy alias kept for any existing call sites
function bindTableRows(containerEl) { bindTableClicks(containerEl); }

// ── Inspector ─────────────────────────────────────────────────

function showLLMInspector(row) {
  S.inspectorItem = { type: "llm", data: row };
  const detail = S.llmDetailsById[row.id];

  const tabs = ["summary", "prompt", "response"];
  if (detail?.parsed_json) tabs.push("parsed");

  renderInspectorTabs(tabs);
  renderInspectorBody(S.inspectorTab);
  showInspector();
}

function showGenericInspector(type, data) {
  if (typeof data === "string") {
    try { data = JSON.parse(data); } catch { data = { raw: data }; }
  }
  S.inspectorItem = { type, data };
  renderInspectorTabs(["summary", "raw"]);
  renderInspectorBody(S.inspectorTab);
  showInspector();
}

function renderInspectorTabs(tabs) {
  const tabsEl = $("insp-tabs");
  if (!tabsEl) return;
  tabsEl.innerHTML = tabs.map(t =>
    `<button class="insp-tab-btn ${S.inspectorTab === t ? "active" : ""}"
             onclick="switchInspectorTab('${t}')">${t}</button>`
  ).join("");
}

function switchInspectorTab(tab) {
  S.inspectorTab = tab;
  const item = S.inspectorItem;
  if (!item) return;
  // Update tab buttons
  document.querySelectorAll(".insp-tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.textContent.trim() === tab);
  });
  renderInspectorBody(tab);
}

function renderInspectorBody(tab) {
  const body = $("insp-body");
  if (!body) return;
  const item = S.inspectorItem;
  if (!item) return;

  const { type, data } = item;

  if (type === "llm") {
    const detail = S.llmDetailsById[data.id];
    if (tab === "summary") {
      body.innerHTML = `
        <table class="kv-table">
          ${kvRow("ID",        data.id)}
          ${kvRow("Agent",     data.agent_id)}
          ${kvRow("Action",    data.action)}
          ${kvRow("Episode",   data.episode_id)}
          ${kvRow("Tick",      data.tick)}
          ${kvRow("Status",    data.ok_status ? "✓ ok" : "✗ fail")}
          ${kvRow("Reason",    data.reason)}
          ${kvRow("Prompt",    `${fmtK(data.prompt_tokens_est)} tok / ${fmtInt(data.prompt_chars)} chars`)}
          ${kvRow("Response",  `${fmtK(data.completion_tokens_est)} tok / ${fmtInt(data.response_chars)} chars`)}
          ${kvRow("Total tok", fmtK(data.total_tokens_est))}
          ${kvRow("Time",      toLocal(data.ts))}
        </table>`;
    } else if (tab === "prompt") {
      const text = detail?.prompt || "(not loaded — click record to expand)";
      body.innerHTML = `<div class="md-body">${renderMarkdown(text)}</div>`;
    } else if (tab === "response") {
      const text = detail?.raw_response || "(not loaded — click record to expand)";
      body.innerHTML = `<div class="md-body">${renderMarkdown(text)}</div>`;
    } else if (tab === "parsed") {
      body.innerHTML = detail?.parsed_json ? renderJSON(detail.parsed_json) : `<div class="muted">No parsed JSON</div>`;
    }
    return;
  }

  // Generic type
  if (tab === "raw") {
    body.innerHTML = renderJSON(data);
    return;
  }

  // Summary: render all top-level string/number fields as KV table
  const skip = new Set(["record", "files", "stdout", "stderr", "prompt", "raw_response",
                         "parsed_json", "evidence", "selected_rows", "attempts", "payload"]);
  const entries = Object.entries(data).filter(([k, v]) => {
    if (skip.has(k)) return false;
    if (Array.isArray(v) && v.length > 10) return false;
    return true;
  });

  body.innerHTML = `
    <table class="kv-table">
      ${entries.map(([k, v]) => {
        let disp;
        if (v == null) disp = "—";
        else if (typeof v === "object") disp = `<span class="muted">${esc(JSON.stringify(v).slice(0, 120))}</span>`;
        else if (typeof v === "boolean") disp = v ? `<span class="ok">✓ true</span>` : `<span class="err">✗ false</span>`;
        else disp = esc(String(v));
        return kvRow(k, disp, true);
      }).join("")}
    </table>
    ${data.error_signature ? `<div class="section-title" style="margin-top:12px">Error</div><pre style="background:var(--surface3);padding:10px;border-radius:var(--r-sm);font-size:11.5px;overflow-x:auto;white-space:pre-wrap">${esc(data.error_signature)}</pre>` : ""}
    ${data.stderr ? `<div class="section-title" style="margin-top:12px">Stderr (tail)</div><pre style="background:var(--surface3);padding:10px;border-radius:var(--r-sm);font-size:11px;overflow-x:auto;max-height:300px;white-space:pre-wrap">${esc(String(data.stderr).slice(-3000))}</pre>` : ""}
  `;
}

function kvRow(k, v, raw = false) {
  return `<tr><td>${esc(k)}</td><td>${raw ? v : esc(String(v ?? "—"))}</td></tr>`;
}

function showInspector() {
  const insp = $("inspector");
  if (insp) insp.style.display = "";
  S.inspectorVisible = true;
}

function hideInspector() {
  const insp = $("inspector");
  if (insp) insp.style.display = "none";
  S.inspectorVisible = false;
}

// ── Render Dispatch ───────────────────────────────────────────

function renderCurrentTab() {
  const payload = S.payload;
  if (!payload) return;

  switch (S.tab) {
    case "overview":  renderOverview(payload);           break;
    case "llm":       renderLLMTab();                    break;
    case "rag":       renderRAGTab(S.ragSubTab);         break;
    case "pipeline":  renderPipelineTab(S.pipelineSubTab); break;
    case "runs":      renderRunsTab(S.runsSubTab);       break;
    case "papers":    renderPapersTab(S.papersSubTab);   break;
  }

}


// ── Tab Switching ─────────────────────────────────────────────

function switchTab(tab) {
  S.tab = tab;
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tab);
  });
  document.querySelectorAll(".tab-panel").forEach(panel => {
    panel.classList.toggle("active", panel.id === `tab-${tab}`);
  });
  renderCurrentTab();
}

// ── Main Data Application ─────────────────────────────────────

function applySnapshot(payload) {
  S.payload = payload;
  updateHeader(payload);
  renderSidebar(payload);
  renderCurrentTab();
}

// ── Live Mode / SSE ───────────────────────────────────────────

let sseSource = null;

function initSSE(refreshSeconds) {
  if (sseSource) { sseSource.close(); sseSource = null; }

  sseSource = new EventSource("/api/events");
  S.sseConnected = true;

  sseSource.addEventListener("snapshot", (e) => {
    try {
      const data = JSON.parse(e.data);
      applySnapshot(data);
    } catch { /* ignore */ }
  });

  sseSource.addEventListener("heartbeat", () => { /* keep-alive */ });

  sseSource.onerror = () => {
    S.sseConnected = false;
    updateLiveDot(false);
    sseSource.close();
    sseSource = null;
    // Retry after a delay
    setTimeout(() => { if (S.liveMode) initSSE(refreshSeconds); }, 5000);
  };

  sseSource.onopen = () => {
    S.sseConnected = true;
    updateLiveDot(true);
  };
}

function updateLiveDot(active) {
  const toggle = $("live-toggle");
  if (toggle) toggle.classList.toggle("active", active && S.liveMode);
}

async function fetchSnapshot() {
  try {
    const res = await fetch("/api/snapshot", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    applySnapshot(data);
  } catch (e) {
    console.error("Snapshot fetch failed:", e);
  }
}

// ── Resize Handles ────────────────────────────────────────────

function initResize() {
  const sidebar = document.querySelector(".sidebar");
  const inspector = document.querySelector(".inspector");
  const handleLeft  = $("handle-left");
  const handleRight = $("handle-right");

  function makeDragger(handle, target, prop, min, max, onRight) {
    if (!handle || !target) return;
    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      document.body.classList.add("resizing-col");
      const startX = e.clientX;
      const startW = target.offsetWidth;

      const onMove = (ev) => {
        const delta = onRight ? startX - ev.clientX : ev.clientX - startX;
        const newW = Math.min(max, Math.max(min, startW + delta));
        target.style.width = `${newW}px`;
      };
      const onUp = () => {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        document.body.classList.remove("resizing-col");
      };
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  }

  makeDragger(handleLeft,  sidebar,   "sidebarWidth",   120, 420, false);
  makeDragger(handleRight, inspector, "inspectorWidth",  160, 700, true);
}

// ── Init ──────────────────────────────────────────────────────

function init() {
  // Tab buttons
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Sub-tab buttons (RAG, Pipeline, Runs, Papers)
  document.querySelectorAll(".sub-tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const subtab = btn.dataset.subtab;
      const panel  = btn.closest(".tab-panel");
      panel.querySelectorAll(".sub-tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      if (panel.id === "tab-rag")      { S.ragSubTab      = subtab; renderRAGTab(subtab);      }
      if (panel.id === "tab-pipeline") { S.pipelineSubTab = subtab; renderPipelineTab(subtab); }
      if (panel.id === "tab-runs")     { S.runsSubTab     = subtab; renderRunsTab(subtab);     }
      if (panel.id === "tab-papers")   { S.papersSubTab   = subtab; renderPapersTab(subtab);   }
    });
  });

  // Episode filter
  $("episode-select")?.addEventListener("change", (e) => {
    S.episodeFilter = toInt(e.target.value);
    renderCurrentTab();
  });

  // Search
  $("search-input")?.addEventListener("input", (e) => {
    S.search = e.target.value.toLowerCase();
    renderCurrentTab();
  });

  // LLM controls
  $("llm-load-btn")?.addEventListener("click", () => loadLLMList(false));
  $("llm-more-btn")?.addEventListener("click", () => loadLLMList(true));
  $("llm-search")?.addEventListener("input",   () => renderLLMList());
  $("llm-agent-select")?.addEventListener("change", () => {});
  $("llm-action-select")?.addEventListener("change", () => {});

  // Refresh button
  $("refresh-btn")?.addEventListener("click", () => fetchSnapshot());

  // Live toggle
  const liveToggle = $("live-toggle");
  if (liveToggle) {
    liveToggle.addEventListener("click", () => {
      S.liveMode = !S.liveMode;
      updateLiveDot(S.liveMode);
      if (S.liveMode) {
        initSSE(5);
      } else {
        if (sseSource) { sseSource.close(); sseSource = null; }
      }
    });
  }

  // Inspector close
  $("insp-close-btn")?.addEventListener("click", hideInspector);

  // Resize handles
  initResize();

  // Initial data load
  fetchSnapshot().then(() => {
    // Start SSE
    initSSE(5);
    updateLiveDot(true);
  });
}

document.addEventListener("DOMContentLoaded", init);
