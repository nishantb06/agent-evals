/* Session graph viewer — chat → session navigation + memory hits. */

const SKILL_COLOR = {
  planner: "#5eb1ff",
  retriever: "#7ddea0",
  researcher: "#ffc857",
  distiller: "#c792ea",
  summariser: "#82aaff",
  critic: "#ff8f8f",
  formatter: "#89ddff",
  coder: "#f78c6c",
  sandbox_executor: "#c3e88d",
  browser: "#b2ccd6",
};

const STANDALONE = "__standalone__";

let currentSession = null;
let currentChat = null; // { id, runs, conversation } or null for standalone
let network = null;
let selectedNodeId = null;

const statusEl = document.getElementById("status");
const panelEl = document.getElementById("panel");
const memoryPanelEl = document.getElementById("memoryPanel");
const tipEl = document.getElementById("tooltip");
const chatSelectEl = document.getElementById("chatSelect");
const selectEl = document.getElementById("sessionSelect");
const queryTextEl = document.getElementById("queryText");
const chatMetaTextEl = document.getElementById("chatMetaText");
const networkEl = document.getElementById("network");

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function short(v, n = 180) {
  const s = typeof v === "string" ? v : JSON.stringify(v);
  if (!s) return "—";
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function truncateLabel(query, max = 80) {
  const q = (query || "").replace(/\s+/g, " ").trim();
  if (!q) return "(empty query)";
  return q.length > max ? q.slice(0, max) + "…" : q;
}

function substituteQuery(value, query) {
  if (value == null) return value;
  const q = query || "";
  if (typeof value === "string") {
    if (value === "USER_QUERY") return q || "USER_QUERY";
    return value.split("USER_QUERY").join(q || "USER_QUERY");
  }
  if (Array.isArray(value)) {
    return value.map((v) => substituteQuery(v, query));
  }
  if (typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      out[k] = substituteQuery(v, query);
    }
    return out;
  }
  return value;
}

function formatInputs(inputs, query) {
  const list = inputs || [];
  if (!list.length) return "(none)";
  return list
    .map((inp) => {
      if (inp === "USER_QUERY") {
        const q = (query || "").replace(/\s+/g, " ").trim();
        return q ? `USER_QUERY → "${short(q, 100)}"` : "USER_QUERY";
      }
      return String(inp);
    })
    .join(", ");
}

function inferEdges(data) {
  const edges = [];
  const seen = new Set();
  const add = (source, target, kind) => {
    const key = source + "->" + target;
    if (seen.has(key) || source === target) return;
    seen.add(key);
    edges.push({ source, target, kind });
  };

  for (const e of data.edges || []) {
    add(e.source, e.target, "recorded");
  }

  for (const n of data.nodes || []) {
    for (const inp of n.inputs || []) {
      if (typeof inp === "string" && inp.startsWith("n:")) {
        add(inp, n.id, "input");
      }
    }
  }

  const byId = [...(data.nodes || [])].sort(
    (a, b) => parseInt(a.id.slice(2), 10) - parseInt(b.id.slice(2), 10)
  );
  for (const n of byId) {
    const succs = n.result?.successors || [];
    if (!succs.length) continue;
    const nNum = parseInt(n.id.slice(2), 10);
    const pool = byId.filter((c) => parseInt(c.id.slice(2), 10) > nNum);
    let i = 0;
    for (const spec of succs) {
      while (i < pool.length && pool[i].skill !== spec.skill) i++;
      if (i >= pool.length) break;
      add(n.id, pool[i].id, "spawn");
      pool.splice(i, 1);
    }
  }
  return edges;
}

function previewOutput(result) {
  if (!result) return "—";
  if (result.error) return "error: " + result.error;
  const o = result.output || {};
  if (o.final_answer) return o.final_answer;
  if (o.rationale) return o.rationale;
  if (o.summary) return o.summary;
  if (o.findings) return o.findings;
  return short(o, 160);
}

function enrichNode(gnode) {
  if (!currentSession) return gnode;
  const extra = currentSession.nodes?.[gnode.id] || {};
  return {
    ...gnode,
    prompt_sent: gnode.prompt_sent ?? extra.prompt_sent ?? null,
    started_at: gnode.started_at ?? extra.started_at ?? null,
    completed_at: gnode.completed_at ?? extra.completed_at ?? null,
    retries: gnode.retries ?? extra.retries ?? 0,
    result: gnode.result || extra.result || null,
    status: gnode.status || extra.status || "—",
  };
}

function renderMemoryPanel(session) {
  if (!session) {
    memoryPanelEl.innerHTML = `
      <h2>Memory / FAISS hits</h2>
      <p class="hint">Load a session to see memory hits for that run.</p>`;
    return;
  }
  const hits = session.memory_hits || [];
  const source = session.memory_hits_source || "none";
  const badge =
    source === "memory_hits.json"
      ? "from memory_hits.json"
      : source === "prompt_sent"
        ? "parsed from prompt"
        : "none";
  const retriever = session.retriever_chunks || [];

  let hitsHtml = "";
  if (!hits.length) {
    hitsHtml = `<p class="hint">No MEMORY HITS in this run${
      source === "none" ? " (nothing stored / nothing to parse)." : "."
    }</p>
    <p class="hint">FAISS similarity scores are not persisted — not shown.</p>`;
  } else {
    hitsHtml = hits
      .map((h, i) => {
        const chunk = h.chunk || h.raw || "";
        const body = chunk
          ? `<details>
              <summary>chunk / text (${chunk.length.toLocaleString()} chars)</summary>
              <pre class="chunk">${esc(chunk)}</pre>
            </details>`
          : `<p class="hint">No chunk text on this hit.</p>`;
        return `<div class="hit-card">
          <div><b>#${i + 1}</b> [${esc(h.kind || "?")}] ${esc(short(h.descriptor || "", 160))}</div>
          <div class="meta">source: ${esc(h.source || "—")}${
            h.id ? ` · id: ${esc(h.id)}` : ""
          }</div>
          ${body}
        </div>`;
      })
      .join("");
  }

  let retrieverHtml = "";
  if (retriever.length) {
    retrieverHtml =
      `<h3>Retriever chunks</h3>` +
      retriever
        .map((c, i) => {
          const text = c.chunk || c.preview || c.summary || JSON.stringify(c);
          return `<div class="hit-card">
            <div><b>retriever #${i + 1}</b> ${esc(c.source || "")}</div>
            <details open>
              <summary>text</summary>
              <pre class="chunk">${esc(typeof text === "string" ? text : JSON.stringify(text, null, 2))}</pre>
            </details>
          </div>`;
        })
        .join("");
  }

  memoryPanelEl.innerHTML = `
    <h2>Memory / FAISS hits (${hits.length})
      <span class="badge">${esc(badge)}</span>
    </h2>
    ${hitsHtml}
    ${retrieverHtml}
  `;
}

function renderPanel(n) {
  if (!n) {
    panelEl.innerHTML = `<p class="hint">Click a node for details.</p>`;
    return;
  }
  const query = currentSession?.query || "";
  const r = n.result || {};
  const prompt = n.prompt_sent || null;
  const status = n.status || "—";
  const displayInputs = substituteQuery(n.inputs || [], query);
  const promptDisplay = prompt ? substituteQuery(prompt, query) : null;
  const hitNote =
    currentSession?.memory_hits?.length
      ? `<p class="hint">This session has ${currentSession.memory_hits.length} memory hit(s) — see panel above.</p>`
      : "";

  panelEl.innerHTML = `
    <p class="hint" style="margin-top:0"><b>session query</b>: ${esc(query || "(empty)")}</p>
    ${hitNote}
    <h2>${esc(n.id)} · ${esc(n.skill)}</h2>
    <div class="kv">
      <dt>status</dt><dd><span class="pill ${esc(status)}">${esc(status)}</span></dd>
      <dt>provider</dt><dd>${esc(r.provider || "—")}</dd>
      <dt>elapsed</dt><dd>${r.elapsed_s != null ? Number(r.elapsed_s).toFixed(2) + "s" : "—"}</dd>
      <dt>success</dt><dd>${r.success == null ? "—" : r.success}</dd>
      <dt>retries</dt><dd>${esc(n.retries ?? 0)}</dd>
      <dt>label</dt><dd>${esc((n.metadata && n.metadata.label) || "—")}</dd>
      <dt>question</dt><dd>${esc((n.metadata && n.metadata.question) || "—")}</dd>
    </div>
    <h3>inputs</h3>
    <pre>${esc(JSON.stringify(displayInputs, null, 2))}</pre>
    <h3>metadata</h3>
    <pre>${esc(JSON.stringify(n.metadata || {}, null, 2))}</pre>
    <h3>output</h3>
    <pre>${esc(JSON.stringify(r.output || {}, null, 2))}</pre>
    ${r.error ? `<h3>error</h3><pre>${esc(r.error)}</pre>` : ""}
    <h3>successors</h3>
    <pre>${esc(JSON.stringify(r.successors || [], null, 2))}</pre>
    <h3>prompt_sent</h3>
    ${
      promptDisplay
        ? `<details class="prompt-fold">
            <summary>Show / hide prompt (${promptDisplay.length.toLocaleString()} chars)</summary>
            <pre class="prompt">${esc(promptDisplay)}</pre>
          </details>`
        : `<pre>(no prompt_sent in nodes/ for this node)</pre>`
    }
  `;
}

function showTip(n, evt) {
  const query = currentSession?.query || "";
  const r = n.result || {};
  const question = n.metadata?.question;
  tipEl.innerHTML = `
    <strong>${esc(n.id)}</strong> · ${esc(n.skill)}
    <div class="muted">${esc(n.status)} · ${esc(r.provider || "—")} · ${
      r.elapsed_s != null ? Number(r.elapsed_s).toFixed(1) + "s" : "—"
    }</div>
    ${
      question
        ? `<div style="margin-top:6px"><b>question</b>: ${esc(short(question, 140))}</div>`
        : ""
    }
    <div style="margin-top:6px"><b>inputs</b>: ${esc(formatInputs(n.inputs, query))}</div>
    <div style="margin-top:4px"><b>out</b>: ${esc(short(previewOutput(r), 220))}</div>
  `;
  tipEl.style.display = "block";
  moveTip(evt);
}

function moveTip(evt) {
  if (!evt) return;
  const x = Math.min(evt.clientX + 14, window.innerWidth - 400);
  const y = Math.min(evt.clientY + 14, window.innerHeight - 140);
  tipEl.style.left = x + "px";
  tipEl.style.top = y + "px";
}

function hideTip() {
  tipEl.style.display = "none";
}

function clearCanvas() {
  if (network) {
    network.destroy();
    network = null;
  }
  selectedNodeId = null;
  panelEl.innerHTML = `<p class="hint">Pick a chat, then a session. Hover a node for a summary; click for full details.</p>`;
  renderMemoryPanel(null);
}

function drawSession(session) {
  currentSession = session;
  renderMemoryPanel(session);
  const data = session.graph || { nodes: [], edges: [] };
  const edgesRaw = inferEdges(data);
  const nodes = new vis.DataSet(
    (data.nodes || []).map((raw) => {
      const n = enrichNode(raw);
      const color = SKILL_COLOR[n.skill] || "#8b9bb0";
      const failed = n.status === "failed" || (n.result && n.result.success === false);
      return {
        id: n.id,
        label: `${n.id}\n${n.skill}`,
        color: {
          background: color,
          border: failed ? "#ff5c5c" : "#0c1016",
          highlight: { background: color, border: "#fff" },
        },
        font: { color: "#0c1016", multi: true, bold: true },
        borderWidth: failed ? 3 : 2,
        shape: "box",
        margin: 10,
        _raw: n,
      };
    })
  );
  const edges = new vis.DataSet(
    edgesRaw.map((e, i) => ({
      id: "e" + i,
      from: e.source,
      to: e.target,
      arrows: "to",
      dashes: e.kind !== "recorded",
      color: { color: e.kind === "recorded" ? "#7a8aa0" : "#4a5a70" },
      width: 1.5,
      label: e.kind === "spawn" ? "spawn" : e.kind === "input" ? "in" : "",
      font: { size: 9, color: "#8b9bb0", strokeWidth: 0 },
    }))
  );

  if (network) network.destroy();
  network = new vis.Network(
    networkEl,
    { nodes, edges },
    {
      layout: {
        hierarchical: {
          enabled: true,
          direction: "UD",
          sortMethod: "directed",
          levelSeparation: 90,
          nodeSpacing: 140,
        },
      },
      physics: false,
      interaction: { hover: true, tooltipDelay: 99999 },
      edges: {
        smooth: { type: "cubicBezier", forceDirection: "vertical", roundness: 0.4 },
      },
    }
  );

  network.on("hoverNode", (params) => {
    const node = nodes.get(params.node);
    showTip(node._raw, params.event?.srcEvent || window.event);
  });
  network.on("blurNode", hideTip);
  network.on("click", (params) => {
    if (!params.nodes.length) return;
    const node = nodes.get(params.nodes[0]);
    selectedNodeId = node.id;
    renderPanel(node._raw);
  });

  const nCount = (data.nodes || []).length;
  const eCount = edgesRaw.length;
  const recorded = (data.edges || []).length;
  const hitN = (session.memory_hits || []).length;
  statusEl.textContent =
    `${nCount} nodes · ${eCount} edges (${recorded} recorded, ${eCount - recorded} inferred)` +
    ` · ${hitN} memory hit(s)`;
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || res.statusText || String(res.status));
  }
  return res.json();
}

function fillSessionSelect(sessions, preferId) {
  selectEl.innerHTML = "";
  if (!sessions.length) {
    selectEl.innerHTML = `<option value="">No sessions</option>`;
    return null;
  }
  for (const s of sessions) {
    const opt = document.createElement("option");
    const id = s.id || s.run_id;
    opt.value = id;
    const q = s.query || "";
    const missing = s.exists === false ? " [missing]" : "";
    opt.textContent = `${id}: ${truncateLabel(q)}${missing}`;
    if (s.exists === false) opt.disabled = true;
    selectEl.appendChild(opt);
  }
  const ids = sessions.map((s) => s.id || s.run_id);
  const pick = ids.includes(preferId) ? preferId : ids.find((id) => {
    const s = sessions.find((x) => (x.id || x.run_id) === id);
    return !s || s.exists !== false;
  }) || ids[0];
  selectEl.value = pick;
  return pick;
}

function updateChatMeta(sessionId) {
  if (!currentChat || currentChat.id === STANDALONE) {
    chatMetaTextEl.textContent = "Standalone sessions (not linked to a chat)";
    return;
  }
  const runs = currentChat.runs || [];
  const idx = runs.findIndex((r) => r.run_id === sessionId);
  const turn = idx >= 0 ? `turn ${idx + 1}/${runs.length}` : "";
  chatMetaTextEl.textContent = `${currentChat.id}${turn ? " · " + turn : ""}`;
}

async function loadSelectedSession() {
  const id = selectEl.value;
  if (!id) {
    clearCanvas();
    return;
  }
  hideTip();
  statusEl.textContent = `Loading ${id}…`;
  try {
    const session = await fetchJSON(`/api/sessions/${encodeURIComponent(id)}`);
    queryTextEl.textContent = session.query || "(empty query)";
    updateChatMeta(id);
    clearCanvas();
    drawSession(session);
  } catch (err) {
    statusEl.textContent = `Failed: ${err.message}`;
    queryTextEl.textContent = "—";
    clearCanvas();
  }
}

async function onChatChanged(preferSessionId) {
  const chatId = chatSelectEl.value;
  clearCanvas();
  queryTextEl.textContent = "Select a session";
  if (!chatId) {
    selectEl.innerHTML = `<option value="">Select a chat first</option>`;
    chatMetaTextEl.textContent = "—";
    return;
  }

  statusEl.textContent = "Loading chat…";
  try {
    if (chatId === STANDALONE) {
      currentChat = { id: STANDALONE, runs: [] };
      const sessions = await fetchJSON("/api/sessions/standalone");
      const pick = fillSessionSelect(sessions, preferSessionId);
      chatMetaTextEl.textContent = "Standalone sessions (not linked to a chat)";
      if (pick) await loadSelectedSession();
      else statusEl.textContent = "0 standalone sessions";
      return;
    }

    const chat = await fetchJSON(`/api/chats/${encodeURIComponent(chatId)}`);
    currentChat = chat;
    const sessions = (chat.runs || []).map((r) => ({
      id: r.run_id,
      run_id: r.run_id,
      query: r.query,
      exists: r.exists,
    }));
    const pick = fillSessionSelect(sessions, preferSessionId);
    if (pick) await loadSelectedSession();
    else {
      statusEl.textContent = "Chat has no linked sessions";
      chatMetaTextEl.textContent = chat.id;
    }
  } catch (err) {
    statusEl.textContent = `Failed: ${err.message}`;
    selectEl.innerHTML = `<option value="">Error</option>`;
  }
}

async function loadChatList(preferChatId, preferSessionId) {
  statusEl.textContent = "Loading chats…";
  const chats = await fetchJSON("/api/chats");
  const prev = preferChatId || chatSelectEl.value;
  chatSelectEl.innerHTML = "";

  const standOpt = document.createElement("option");
  standOpt.value = STANDALONE;
  standOpt.textContent = "Standalone sessions";
  chatSelectEl.appendChild(standOpt);

  for (const c of chats) {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = `${c.id}: ${truncateLabel(c.preview)} (${c.turn_count} turns)`;
    chatSelectEl.appendChild(opt);
  }

  if (!chats.length && !prev) {
    chatSelectEl.value = STANDALONE;
  } else {
    const ids = [STANDALONE, ...chats.map((c) => c.id)];
    chatSelectEl.value = ids.includes(prev) ? prev : chats[0]?.id || STANDALONE;
  }
  await onChatChanged(preferSessionId);
}

chatSelectEl.addEventListener("change", () => onChatChanged());
selectEl.addEventListener("change", () => loadSelectedSession());
document.getElementById("refreshBtn").addEventListener("click", () => {
  loadChatList(chatSelectEl.value, selectEl.value);
});

networkEl.addEventListener("mousemove", (e) => {
  if (tipEl.style.display === "block") moveTip(e);
});

loadChatList().catch((err) => {
  statusEl.textContent = `Failed to list chats: ${err.message}`;
  chatSelectEl.innerHTML = `<option value="">Error</option>`;
});
