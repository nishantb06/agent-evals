/* Ollive evals UI — live chat + upload eval */

const state = {
  chatId: null,
  personaApplied: false,
  conversation: [],
  scores: [], // parallel to assistant turns: {hallucination, bias_harm, jailbreak}
  uploadTurns: null,
  lastEvalResult: null,
};

const CRITERIA = [
  { key: "hallucination", short: "H", label: "Hallucination" },
  { key: "bias_harm", short: "B", label: "Bias / Harm" },
  { key: "jailbreak", short: "J", label: "Jailbreak" },
];

function $(id) { return document.getElementById(id); }

function scoreClass(score) {
  if (score == null) return "err";
  if (score >= 8) return "good";
  if (score >= 5) return "mid";
  return "bad";
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!res.ok) {
    const msg = (data && (data.detail || data.error)) || res.statusText;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

function updateAverages(targetId, scoreList) {
  const el = $(targetId);
  if (!el) return;
  for (const c of CRITERIA) {
    const chip = el.querySelector(`[data-k="${c.key}"]`);
    if (!chip) continue;
    const vals = scoreList
      .map((s) => s && s[c.key] && s[c.key].score)
      .filter((n) => typeof n === "number");
    if (!vals.length) {
      chip.textContent = `${c.short} —`;
      chip.className = "avg-chip";
      continue;
    }
    const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
    chip.textContent = `${c.short} ${avg.toFixed(1)}`;
    chip.className = `avg-chip ${scoreClass(Math.round(avg))}`;
  }
}

function renderScoreChips(scores, detailId) {
  if (!scores) {
    return `<div class="score-row"><span class="score-chip err">scoring…</span></div>`;
  }
  const chips = CRITERIA.map((c) => {
    const s = scores[c.key] || {};
    const val = s.score;
    const label = val == null ? `${c.short} ?` : `${c.short} ${val}`;
    return `<button type="button" class="score-chip ${scoreClass(val)}" data-detail="${detailId}" data-crit="${c.key}">${label}</button>`;
  }).join("");

  const panels = CRITERIA.map((c) => {
    const s = scores[c.key] || {};
    const viol = (s.violations || []).map((v) => `<li>${escapeHtml(v)}</li>`).join("");
    let kb = "";
    if (c.key === "hallucination" && s.kb_chunks_used && s.kb_chunks_used.length) {
      kb = `<div><strong>KB chunks:</strong> ${s.kb_chunks_used.map((k) =>
        escapeHtml(`${k.source}#${k.index + 1} (score ${k.score})`)
      ).join(", ")}</div>`;
    }
    const err = s.error ? `<div><strong>error:</strong> ${escapeHtml(s.error)}</div>` : "";
    return `<div class="score-detail" id="${detailId}-${c.key}">
      <div><strong>${c.label}</strong> — score ${s.score == null ? "n/a" : s.score}</div>
      <div>${escapeHtml(s.rationale || "")}</div>
      ${viol ? `<ul>${viol}</ul>` : ""}
      ${kb}${err}
    </div>`;
  }).join("");

  return `<div class="score-row">${chips}</div>${panels}`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function appendBubble(logEl, role, content, { scores, detailId } = {}) {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  let html = `<div class="role">${role}</div><div class="body">${escapeHtml(content)}</div>`;
  if (role === "assistant") {
    html += renderScoreChips(scores, detailId || `d-${Date.now()}`);
  }
  div.innerHTML = html;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function wireScoreChips(root) {
  root.querySelectorAll(".score-chip[data-detail]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = `${btn.dataset.detail}-${btn.dataset.crit}`;
      const panel = document.getElementById(id);
      if (!panel) return;
      const open = panel.classList.contains("open");
      root.querySelectorAll(".score-detail").forEach((p) => p.classList.remove("open"));
      if (!open) panel.classList.add("open");
    });
  });
}

function renderLiveLog() {
  const log = $("chatLog");
  log.innerHTML = "";
  let asstIdx = 0;
  for (const turn of state.conversation) {
    if (turn.role === "assistant") {
      const detailId = `live-${asstIdx}`;
      appendBubble(log, "assistant", turn.content, {
        scores: state.scores[asstIdx],
        detailId,
      });
      asstIdx += 1;
    } else {
      appendBubble(log, "user", turn.content);
    }
  }
  wireScoreChips(log);
  updateAverages("liveAverages", state.scores);
}

async function sendMessage(e) {
  e.preventDefault();
  const input = $("messageInput");
  const msg = input.value.trim();
  if (!msg) return;
  input.value = "";
  $("sendBtn").disabled = true;

  state.conversation.push({ role: "user", content: msg });
  renderLiveLog();

  const body = { message: msg, chat_id: state.chatId };
  if (!state.personaApplied) {
    const persona = $("personaInput").value.trim();
    if (persona) body.persona = persona;
  }

  try {
    const res = await api("/api/chat", { method: "POST", body: JSON.stringify(body) });
    state.chatId = res.chat_id;
    state.personaApplied = true;
    $("chatIdLabel").textContent = res.chat_id;
    state.conversation.push({
      role: "assistant",
      content: res.answer,
      run_id: res.run_id,
    });
    state.scores.push(null);
    renderLiveLog();

    const asstIdx = state.scores.length - 1;
    try {
      const scores = await api("/api/judge", {
        method: "POST",
        body: JSON.stringify({ conversation: state.conversation }),
      });
      state.scores[asstIdx] = scores;
    } catch (err) {
      state.scores[asstIdx] = {
        hallucination: { score: null, rationale: String(err), error: "judge_failed", violations: [] },
        bias_harm: { score: null, rationale: String(err), error: "judge_failed", violations: [] },
        jailbreak: { score: null, rationale: String(err), error: "judge_failed", violations: [] },
      };
    }
    renderLiveLog();
  } catch (err) {
    appendBubble($("chatLog"), "assistant", `Error: ${err.message}`);
  } finally {
    $("sendBtn").disabled = false;
    input.focus();
  }
}

function newChat() {
  state.chatId = null;
  state.personaApplied = false;
  state.conversation = [];
  state.scores = [];
  $("chatIdLabel").textContent = "—";
  renderLiveLog();
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((p) => {
    p.classList.toggle("active", p.id === `tab-${name}`);
  });
}

function onFileSelected(e) {
  const file = e.target.files && e.target.files[0];
  $("runEvalBtn").disabled = !file;
  $("downloadBtn").disabled = true;
  state.uploadTurns = null;
  state.lastEvalResult = null;
  $("uploadStatus").textContent = file ? file.name : "";
  $("scoreTable").innerHTML = "";
  $("uploadAverages").innerHTML = "";
  $("uploadTranscript").innerHTML = "";
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      if (!Array.isArray(data)) throw new Error("expected a JSON array");
      state.uploadTurns = data;
      const log = $("uploadTranscript");
      log.innerHTML = "";
      for (const t of data) {
        appendBubble(log, t.role === "assistant" ? "assistant" : "user", t.content || "");
      }
      $("uploadStatus").textContent = `${file.name} · ${data.length} turns`;
    } catch (err) {
      $("uploadStatus").textContent = `parse error: ${err.message}`;
      $("runEvalBtn").disabled = true;
    }
  };
  reader.readAsText(file);
}

async function runUploadEval() {
  if (!state.uploadTurns) return;
  $("runEvalBtn").disabled = true;
  $("uploadStatus").classList.add("status-busy");
  $("uploadStatus").textContent = "judging…";
  try {
    const result = await api("/api/eval-file", {
      method: "POST",
      body: JSON.stringify(state.uploadTurns),
    });
    state.lastEvalResult = result;
    $("downloadBtn").disabled = false;
    $("uploadStatus").classList.remove("status-busy");
    $("uploadStatus").textContent = `done · saved ${result.saved_path || ""}`;

    const avgEl = $("uploadAverages");
    avgEl.innerHTML = CRITERIA.map((c) => {
      const v = result.averages && result.averages[c.key];
      const label = v == null ? `${c.short} —` : `${c.short} ${Number(v).toFixed(1)}`;
      return `<span class="avg-chip ${scoreClass(v == null ? null : Math.round(v))}" data-k="${c.key}">${label}</span>`;
    }).join("");

    const table = $("scoreTable");
    table.innerHTML = "";
    (result.turns || []).forEach((t, i) => {
      const card = document.createElement("div");
      card.className = "score-card";
      const detailId = `up-${i}`;
      card.innerHTML = `
        <div class="preview">turn ${t.turn_index} · ${escapeHtml(t.content_preview || "")}</div>
        ${renderScoreChips(t.scores, detailId)}
      `;
      table.appendChild(card);
    });
    wireScoreChips(table);
  } catch (err) {
    $("uploadStatus").classList.remove("status-busy");
    $("uploadStatus").textContent = `error: ${err.message}`;
  } finally {
    $("runEvalBtn").disabled = false;
  }
}

function downloadResults() {
  if (!state.lastEvalResult) return;
  const blob = new Blob([JSON.stringify(state.lastEvalResult, null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = state.lastEvalResult.saved_path || "eval-results.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

async function checkHealth() {
  try {
    const h = await api("/api/health");
    const el = $("healthStatus");
    if (h.gateway_ok) {
      el.textContent = `gateway ok · ${h.kb_chunks} kb chunks`;
      el.style.color = "var(--ok)";
    } else {
      el.textContent = `gateway down (${h.gateway_error || "?"}) · ${h.kb_chunks} kb chunks`;
      el.style.color = "var(--bad)";
    }
  } catch (err) {
    $("healthStatus").textContent = `health check failed: ${err.message}`;
    $("healthStatus").style.color = "var(--bad)";
  }
}

function init() {
  document.querySelectorAll(".tab").forEach((t) => {
    t.addEventListener("click", () => switchTab(t.dataset.tab));
  });
  $("chatForm").addEventListener("submit", sendMessage);
  $("newChatBtn").addEventListener("click", newChat);
  $("fileInput").addEventListener("change", onFileSelected);
  $("runEvalBtn").addEventListener("click", runUploadEval);
  $("downloadBtn").addEventListener("click", downloadResults);
  checkHealth();
}

init();
