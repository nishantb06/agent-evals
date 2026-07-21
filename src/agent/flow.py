"""Growing-graph orchestrator.

The agent's loop is a NetworkX DiGraph. Each node is a skill; edges
carry typed AgentResult payloads. The graph GROWS at runtime via five
actors: the Planner's seed plan, dynamic successors from any skill,
static `internal_successors` from the yaml, Critic auto-insertion on
edges out of `critic:true` skills, and Planner re-invocation on node
failure (gated by `recovery.plan_recovery`). The Planner is
tool-blind — it names skills, never tools.

Persistence lives in persistence.py; skill execution in skills.py;
failure-policy in recovery.py; sandbox in sandbox.py.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass

import networkx as nx

import memory as memory_svc
from gateway import ensure_gateway
from persistence import SessionStore
from recovery import handle_critic_verdict, plan_recovery
from schemas import AgentResult, NodeState
from skills import SkillRegistry, run_skill

MAX_NODES = 60  # hard cap so a Planner loop cannot grow forever


# ── Graph ────────────────────────────────────────────────────────────────────

class Graph:
    """NetworkX DiGraph wrapper. Nodes are str ids `n:<i>`; each node carries
    `skill`, `inputs` (list of str), and `status`."""

    def __init__(self):
        self.g = nx.DiGraph()
        self._counter = 0

    def add_node(self, skill: str, inputs: list[str], metadata: dict | None = None) -> str:
        self._counter += 1
        nid = f"n:{self._counter}"
        self.g.add_node(nid, skill=skill, inputs=list(inputs),
                        metadata=dict(metadata or {}), status="pending")
        for inp in inputs:
            if inp.startswith("n:") and inp in self.g.nodes:
                self.g.add_edge(inp, nid)
        return nid

    def mark(self, nid: str, status: str) -> None:
        self.g.nodes[nid]["status"] = status

    def ready_nodes(self) -> list[str]:
        # A predecessor counts as "satisfied" when it is either complete or
        # skipped (the latter is how a Critic-fail removes a child from the
        # critical path without blocking unrelated branches downstream).
        out = []
        for nid, d in self.g.nodes(data=True):
            if d["status"] != "pending":
                continue
            preds = list(self.g.predecessors(nid))
            if all(self.g.nodes[p]["status"] in ("complete", "skipped") for p in preds):
                out.append(nid)
        return out

    def has_running(self) -> bool:
        return any(d["status"] == "running" for _, d in self.g.nodes(data=True))

    def extend_from(self, src_nid: str, result: AgentResult,
                    *, registry: SkillRegistry) -> list[str]:
        """Splice in dynamic successors, static internal_successors, and
        critic auto-insertion. Returns the list of new node ids.

        Resolves label-based input references (`n:<label>`) against the
        `metadata.label` of nodes added in the same batch. The Planner is
        encouraged to name its nodes by label so it can reference them
        without knowing the integer ids the orchestrator will hand out."""
        added: list[str] = []
        src_def = registry.get(self.g.nodes[src_nid]["skill"])

        # Pass 1: add the new nodes; build a label → assigned-id map.
        label_to_id: dict[str, str] = {}
        pending: list[tuple[str, list[str]]] = []
        for spec in result.successors:
            label = (spec.metadata or {}).get("label")
            new_id = self.add_node(spec.skill, inputs=[],
                                   metadata=spec.metadata)
            added.append(new_id)
            if isinstance(label, str) and label:
                label_to_id[label] = new_id
            pending.append((new_id, list(spec.inputs)))

        # Pass 2: resolve inputs now that every sibling has an id. Translate
        # `n:<label>` to `n:<assigned-id>` if the label matches; pass numeric
        # `n:<i>` references through; pass anything else through unchanged.
        # NOTE: an empty `raw_inputs` is now a legitimate Planner signal for
        # a fan-out worker scoped via `metadata.question` (see planner.md).
        # We do NOT substitute the parent in that case — doing so would dump
        # the parent's full output (which for the Planner contains every
        # sibling's question) back into the worker's INPUTS block and undo
        # the scoping. The structural parent edge is preserved separately
        # below so the graph topology is still correct.
        for new_id, raw_inputs in pending:
            resolved: list[str] = []
            for inp in raw_inputs:
                # `n:<label>` or `n:<int>` form (preferred).
                if inp.startswith("n:"):
                    suffix = inp[2:]
                    if suffix in label_to_id:
                        resolved.append(label_to_id[suffix])
                        continue
                    if suffix.isdigit() and inp in self.g.nodes:
                        resolved.append(inp)
                        continue
                # Bare label form — the Planner sometimes drops the n: prefix.
                if inp in label_to_id:
                    resolved.append(label_to_id[inp])
                    continue
                # Special literal — the user query is always available.
                if inp == "USER_QUERY":
                    resolved.append(inp)
                    continue
                # Artifact handle — pass through, the input renderer handles it.
                if inp.startswith("art:"):
                    resolved.append(inp)
                    continue
                # Unresolvable input — fall back to the parent so the child
                # has at least one upstream dependency to wait on. This still
                # leaks the parent's output into INPUTS, but only when the
                # Planner emitted a bad input name; it is not the fan-out
                # path. A future round may want to fail loudly here instead.
                resolved.append(src_nid)
            self.g.nodes[new_id]["inputs"] = resolved
            for inp in resolved:
                if inp.startswith("n:") and inp in self.g.nodes:
                    self.g.add_edge(inp, new_id)
            # Fan-out worker case: planner emitted inputs=[] on purpose. No
            # data dependency, but we still record the structural parent
            # edge so the executor's `ready_nodes` ordering and replay
            # topology stay coherent.
            if not raw_inputs:
                self.g.add_edge(src_nid, new_id)

        for child_skill in src_def.internal_successors:
            nid = self.add_node(child_skill, inputs=[src_nid])
            added.append(nid)

        # Critic auto-insertion: place a Critic before each newly-added
        # child so the child only runs after Critic passes.
        if src_def.critic and added:
            for child_nid in list(added):
                self.g.remove_edge(src_nid, child_nid)
                critic_nid = self.add_node(
                    "critic", inputs=[src_nid],
                    metadata={"target": src_nid, "child": child_nid},
                )
                self.g.add_edge(critic_nid, child_nid)
                added.append(critic_nid)

        return added


# ── Executor ─────────────────────────────────────────────────────────────────

class Executor:
    def __init__(self, registry: SkillRegistry | None = None):
        ensure_gateway()
        self.registry = registry or SkillRegistry()

    async def run(self, query: str, *, session_id: str | None = None,
                  resume: bool = False,
                  chat_history: list[dict] | None = None,
                  chat_context: str | None = None,
                  persona: str | None = None) -> str:
        sid = session_id or f"run-{uuid.uuid4().hex[:8]}"
        store = SessionStore(sid)
        if resume:
            existing = store.read_graph()
            if existing is None:
                raise RuntimeError(f"cannot resume {sid}: no graph.pkl on disk")
            graph_obj = existing
            graph = Graph.__new__(Graph)
            graph.g = graph_obj
            graph._counter = max(
                [int(n.split(":")[1]) for n in graph.g.nodes if n.startswith("n:")] or [0]
            )
            for _, d in graph.g.nodes(data=True):
                if d["status"] == "running":
                    d["status"] = "pending"
            if not query:
                query = store.read_query()
        else:
            store.write_query(query)
            graph = Graph()
            graph.add_node("planner", inputs=["USER_QUERY"])

        print(f"\n{'═' * 78}\nsession {sid}  ─  query: {query}\n{'═' * 78}")
        # Read memory ONCE at session start; the same hits flow into every
        # skill's prompt so every cognitive role sees a consistent view.
        # chat_history (prior turns) feeds the keyword fallback only; the
        # ordered transcript still lives in ChatStore, not Memory.
        memory_hits = memory_svc.read(query, history=chat_history) or []
        if memory_hits:
            print(f"[memory.read] {len(memory_hits)} hit(s) visible to every skill this run")
        try:
            store.write_memory_hits(memory_hits)
        except Exception as e:
            print(f"[memory_hits] write skipped: {e!r}")
        try:
            memory_svc.remember(query, source="user_query", run_id=sid)
        except Exception as e:
            print(f"[memory.remember] skipped: {e!r}")

        formatter_answer: str | None = None
        executed_count = 0
        # Per-target cap for critic-fail recovery.
        recovered_branches: dict[str, bool] = {}
        # When the cap fires, the branch is skipped and the final answer
        # reflects missing data. Track every second-or-later critic-fail
        # here so the final log can surface it.
        critic_fail_cap_hit: list[str] = []

        while True:
            ready = graph.ready_nodes()
            if not ready and not graph.has_running():
                break
            if executed_count + len(ready) > MAX_NODES:
                print(f"[flow] node cap {MAX_NODES} hit at {executed_count}; stopping")
                break

            for nid in ready:
                graph.mark(nid, "running")
            store.write_graph(graph.g)

            outcomes = await asyncio.gather(*[
                self._run_one(
                    nid, graph, sid, query, store, memory_hits,
                    chat_context=chat_context, persona=persona,
                )
                for nid in ready
            ])

            for nid, result, prompt in outcomes:
                executed_count += 1
                graph.g.nodes[nid]["result"] = result
                graph.mark(nid, "complete" if result.success else "failed")
                store.write_node(NodeState(
                    node_id=nid, skill=graph.g.nodes[nid]["skill"],
                    status=graph.g.nodes[nid]["status"],
                    inputs=graph.g.nodes[nid]["inputs"],
                    result=result, prompt_sent=prompt,
                    started_at=time.time() - result.elapsed_s,
                    completed_at=time.time(),
                ))
                print(f"[{nid}] {graph.g.nodes[nid]['skill']:18s} "
                      f"{graph.g.nodes[nid]['status']:8s} "
                      f"({result.elapsed_s:.1f}s)"
                      + (f"  err={result.error[:80]}" if result.error else ""))

                if result.success:
                    if graph.g.nodes[nid]["skill"] == "critic":
                        if handle_critic_verdict(nid, result, graph,
                                                 recovered_branches,
                                                 critic_fail_cap_hit):
                            continue
                        # verdict == pass: the child is now ready to run.
                    graph.extend_from(nid, result, registry=self.registry)
                    if graph.g.nodes[nid]["skill"] == "formatter":
                        fa = result.output.get("final_answer")
                        if isinstance(fa, str) and fa.strip():
                            formatter_answer = fa
                else:
                    failed_skill = graph.g.nodes[nid]["skill"]
                    decision = plan_recovery(
                        failed_skill=failed_skill,
                        error_text=result.error or "",
                        failed_node_id=nid,
                    )
                    if decision.action == "skip":
                        print(f"  ↪ {nid} failed ({decision.reason}, "
                              f"skill={failed_skill}): {decision.note}")
                        continue
                    # action == "replan"
                    rec_nid = graph.add_node(
                        "planner", inputs=["USER_QUERY"],
                        metadata={"failure_report": decision.failure_report,
                                  "recovers": nid,
                                  "recovery_reason": decision.reason},
                    )
                    print(f"  ↪ recovery ({decision.reason}): planner node "
                          f"{rec_nid} queued for {nid}")

            store.write_graph(graph.g)

        if formatter_answer is None:
            for nid in reversed(list(graph.g.nodes)):
                d = graph.g.nodes[nid]
                if d["status"] == "complete" and isinstance(d.get("result"), AgentResult):
                    formatter_answer = json.dumps(d["result"].output)[:2000]
                    break

        if critic_fail_cap_hit:
            # Without this the cap firing was invisible and the user would
            # just see a thin formatter answer with no explanation of why.
            print(f"\n[flow] WARNING: critic-fail cap hit on "
                  f"{len(critic_fail_cap_hit)} branch(es): "
                  f"{', '.join(critic_fail_cap_hit)}. "
                  f"The final answer reflects missing data from these "
                  f"branches because the Critic rejected the re-planned "
                  f"output too.")
        print(f"\n{'═' * 78}\nFINAL: {(formatter_answer or '')[:600]}\n{'═' * 78}\n")
        return formatter_answer or ""

    async def _run_one(self, nid: str, graph: Graph, sid: str, query: str,
                       store: SessionStore, memory_hits: list,
                       *, chat_context: str | None = None,
                       persona: str | None = None) -> tuple[str, AgentResult, str]:
        skill_name = graph.g.nodes[nid]["skill"]
        skill = self.registry.get(skill_name)
        fr = graph.g.nodes[nid].get("metadata", {}).get("failure_report")
        store.write_node(NodeState(node_id=nid, skill=skill_name, status="running",
                                   inputs=graph.g.nodes[nid]["inputs"],
                                   started_at=time.time()))
        try:
            result, prompt = await run_skill(
                skill, nid, graph.g.nodes, sid, query, fr,
                memory_hits=memory_hits,
                chat_context=chat_context,
                persona=persona,
            )
        except Exception as e:  # pragma: no cover - dispatcher fault path
            result = AgentResult(success=False, agent_name=skill_name,
                                 error=f"exception: {type(e).__name__}: {e}")
            prompt = "(exception before prompt-render)"
        return nid, result, prompt


# ── CLI ──────────────────────────────────────────────────────────────────────

@dataclass
class CliArgs:
    """Parsed argv for flow.py. mode is one of: repl | resume | oneshot."""

    mode: str
    chat_id: str | None = None
    persona: str | None = None
    resume_sid: str | None = None
    query: str = ""


class CliParseError(ValueError):
    """Raised when argv is malformed (missing flag value, unknown flag)."""


def _parse_cli(argv: list[str]) -> CliArgs:
    """Parse flow.py argv into CliArgs. Order of --chat / --persona is free.

    Raises CliParseError on missing values or unknown flags when flags are
    mixed with a one-shot query.
    """
    chat_id: str | None = None
    persona: str | None = None
    positional: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--chat":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise CliParseError("--chat requires a chat id")
            chat_id = argv[i + 1]
            i += 2
            continue
        if tok == "--persona":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise CliParseError("--persona requires a persona string")
            persona = argv[i + 1]
            i += 2
            continue
        if tok == "--resume":
            if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
                raise CliParseError("--resume requires a session id")
            resume_sid = argv[i + 1]
            query = " ".join(argv[i + 2 :])
            return CliArgs(mode="resume", resume_sid=resume_sid, query=query)
        if tok.startswith("--"):
            raise CliParseError(f"unknown flag: {tok}")
        positional.append(tok)
        i += 1

    if chat_id is not None or persona is not None or not positional:
        if positional:
            raise CliParseError(
                "positional query cannot be combined with --chat / --persona; "
                "use the REPL or a bare one-shot query"
            )
        return CliArgs(mode="repl", chat_id=chat_id, persona=persona)

    return CliArgs(mode="oneshot", query=" ".join(positional))


def _run_repl(chat_id: str | None, persona: str | None = None) -> None:
    """Interactive multi-turn chat. Each line → one graph session via handle_turn."""
    from chat import handle_turn, new_chat_id

    cid = chat_id or new_chat_id("cli")
    print(f"chat {cid}")
    if persona and persona.strip():
        preview = persona.strip().replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80] + "…"
        print(f"persona: {preview}")
    print("Multi-turn REPL. Commands: /help  /chat  /quit")
    print("Each message runs a fresh graph session under this chat.")
    print("Persona is set at launch via --persona (persists in meta.json).\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit", "quit", "exit"):
            break
        if line == "/help":
            print("  /chat   print current chat id")
            print("  /quit   leave the REPL")
            print("  otherwise: send a user message (one graph run)")
            print("  launch with --persona \"...\" to set chat persona")
            continue
        if line == "/chat":
            print(f"  chat_id = {cid}")
            continue
        try:
            result = asyncio.run(
                handle_turn(cid, line, channel="cli", persona=persona)
            )
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}")
            continue
        cid = result.chat_id
        print(f"agent> {result.answer}")
        print(f"[run {result.run_id}  chat {result.chat_id}]\n")


def main() -> None:
    try:
        parsed = _parse_cli(sys.argv[1:])
    except CliParseError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    if parsed.mode == "repl":
        _run_repl(parsed.chat_id, persona=parsed.persona)
        return
    if parsed.mode == "resume":
        asyncio.run(
            Executor().run(
                parsed.query,
                session_id=parsed.resume_sid,
                resume=True,
            )
        )
        return
    # One-shot: flow.py "your question"
    asyncio.run(Executor().run(parsed.query))


if __name__ == "__main__":
    main()
