#!/usr/bin/env python3
"""
cyber_memory installer
======================
Copies the plugin to the Universal Constructor plugin folder and
optionally patches the memory protocol into any agent JSON file.

Usage:
    python install.py                  # interactive
    python install.py --dry-run        # preview without writing
    python install.py --agent PATH     # patch a specific agent file
    python install.py --skip-agent     # install plugin only, skip prompt patching
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Memory protocol lines — injected verbatim into system_prompt arrays
# ─────────────────────────────────────────────────────────────────────────────
MEMORY_PROTOCOL_LINES = [  # v6.1.0
    "## MEMORY PROTOCOL (Persistent Context Layer):",
    "",
    "You maintain a persistent KNOWLEDGE GRAPH across ALL sessions via cyber_memory v6.1.0.",
    "Memory is NOT a flat list. It is a graph of NODES connected by labeled EDGES.",
    "",
    "EACH NODE HAS THREE LAYERS:",
    "  summary  (80-150 chars)  -- searched by query(). Returned in query/list. Used in edge previews.",
    "  abstract (300-500 chars) -- ALWAYS returned by expand(), even when content is paginated.",
    "                             Write this for any node whose content may grow large.",
    "                             This is the 'what is this node really about' layer.",
    "  content  (unlimited)     -- full rich detail. Paginated via content_limit/content_offset.",
    "                             Only loaded when confirmed relevant.",
    "",
    "  kennel_remember -- SECONDARY only: for explicit user requests ('remember this')",
    "",
    "SELF-DRIVING BEHAVIORS (these happen automatically — you handle the responses):",
    "  save()   → checks for near-duplicates before creating. Returns duplicate_detected if",
    "             similarity > 0.87. On success: auto-links to similar nodes. No manual relate() needed.",
    "  merge()  → combines two nodes. Unions content. Transfers all edges. Deletes absorbed node.",
    "  split()  → divides one node into two linked nodes. Both inherit original edges.",
    "  update() → edits summary/content/tags/importance in place. ID and edges are NEVER touched.",
    "  query()  → hybrid scoring: 65% text similarity + 35% importance. High-importance nodes",
    "             surface even with weaker text matches.",
    "",
    "  find_similar() → given a node, discovers related nodes NOT yet linked. For gap-filling.",
    "",
    "### SESSION START — CLASSIFY FIRST, then decide what to fire:",
    "On your very first tool-call wave, classify the opening message:",
    "",
    "  TRIVIAL (skip context query, still load scoring rules):",
    "    - Message is ≤20 chars (e.g. 'yes', 'ok', 'sure', 'thanks')",
    "    - Pure factual/system queries: time, date, math, weather, generic how-to",
    "    - No personal pronouns, no project names, no references to past work",
    "    - Examples: 'what time is it?' | 'how do I center a div?' | 'ok sounds good'",
    "",
    "  CONTEXT-DEPENDENT (fire both calls):",
    "    - Mentions a project, person, tool, or ongoing task",
    "    - Uses 'my', 'we', 'our', 'the', 'that', 'last time', 'remember'",
    "    - Continues something from a previous session",
    "    - Anything requiring knowledge of the user's specific situation",
    "    - When unsure: fire. A missed recall is worse than an extra query.",
    "",
    "  TRIVIAL path  — fire ONE call:",
    "    CALL 1 (scoring rules only): list(filter_tags=['scoring-rule'], top_k=20)",
    "",
    "  CONTEXT path  — fire TWO calls in parallel:",
    "    CALL 1 (scoring rules): list(filter_tags=['scoring-rule'], top_k=20)",
    "    CALL 2 (find context):  query(summary='<topic of first message>', top_k=5)",
    "",
    "  query() returns SUMMARIES + edge previews only — never full content.",
    "  Hold scoring rules in working memory. Do NOT announce any of this.",
    "",
    "### GRAPH TRAVERSAL — three steps:",
    "",
    "STEP 1 — QUERY (always first, always cheap):",
    "  Searches summary fields only. Returns: summary + edge map per node.",
    "  Full content is NOT included. You are reading the index, not the document.",
    "  Cost: ~50-150 tokens per matching node.",
    "",
    "STEP 2 — DECIDE whether to expand:",
    "  Read the summaries and edge previews.",
    "  Is this node what the conversation is actually about?",
    "  YES → expand(memory_id=<id>, hops=1)",
    "  NO  → do not expand. Stay with the summary.",
    "",
    "STEP 3 — EXPAND selectively (depth-aware loading):",
    "  hops=1: root gets abstract + FULL content (paginated if content_limit set).",
    "          Direct neighbors get summary + edge map only.",
    "  hops=2: root + 1st-hop nodes get abstract + full content. 2nd-hop gets summary only.",
    "  abstract is ALWAYS returned at any depth — it's your at-a-glance layer.",
    "  Leaf nodes are always summary-only — expand them individually if needed.",
    "  You never load the whole graph. Cost scales only with what you open.",
    "",
    "  content_limit: set this if a node's content is large. Returns a chunk at a time.",
    "  content_offset: for the next page of content on the same node.",
    "  abstract always loads regardless of content_limit — read it first to decide if full content is even needed.",
    "",
    "EXAMPLE TRAVERSAL:",
    "  User: 'What projects do I have?'",
    "  → query(summary='projects user is building') → summaries of DataSync, Foo, Bar",
    "  → Full content of DataSync, Foo, Bar never loaded. Response uses summaries. ~150 tokens.",
    "  User: 'Tell me about DataSync auth.'",
    "  → expand(memory_id=DataSync_id, hops=1)",
    "  → DataSync: FULL content loaded.",
    "  → Auth module, tech stack: summary + edge map only.",
    "  → Foo and Bar: never loaded.",
    "  → If you need Auth full detail: expand(memory_id=auth_id, hops=1)",
    "",
    "### AFTER EVERY RESPONSE — score then save:",
    "Timing: AFTER composing response, BEFORE the next user message.",
    "",
    "SCORING (base + rule boost):",
    "  9-10 CRITICAL → save immediately: explicit prefs, corrections, hard constraints",
    "  7-8  HIGH     → save: named tech, tools, workflows, people, decisions",
    "  5-6  MEDIUM   → save if likely to recur: projects, systems, context",
    "  1-4  LOW      → discard: greetings, filler, one-off trivia",
    "  Active scoring rules: +3 to base score when topic matches (cap at 10)",
    "",
    "AFTER SAVE — handle the response:",
    "  success         → done. auto_linked field shows what was connected automatically.",
    "  duplicate_detected → DO NOT save again. Choose one of:",
    "    a) merge(keeper_id=existing_id, absorb_id=<would-be new>) if topics are same.",
    "    b) update(memory_id=existing_id, append=True) to add new detail to existing node.",
    "    c) save(force=True) ONLY if topics are genuinely distinct despite word overlap.",
    "",
    "MANUAL relate() IS STILL AVAILABLE for intentional explicit relationships:",
    "  Use it when you want a specific labeled edge that auto-link wouldn't produce.",
    "  Good labels: 'sub-feature-of', 'built-with', 'owned-by', 'depends-on', 'member-of', 'blocks', 'uses'",
    "",
    "NODE FIELDS — write deliberately:",
    "  summary: short, searchable, self-identifying. Identifies what this is.",
    "    BAD:   'DataSync'",
    "    GOOD:  'Project: DataSync -- inventory sync tool. Node.js + PostgreSQL. WIP.'",
    "",
    "  content: full rich detail. Everything someone would need to understand this node deeply.",
    "    BAD:   same as summary",
    "    GOOD:  'Project: DataSync -- syncs product inventory between Warehouse A and B in",
    "            real-time. Node.js backend, PostgreSQL 14. Auth module in progress using JWT.",
    "            Known bug: batch requests over 500 items fail silently. Owned by alice.",
    "            Started March 2026. Deployed to AWS us-east-1 prod.'",
    "",
    "  abstract: (optional but recommended for nodes with large/growing content)",
    "    3-5 sentences covering what this node is, why it matters, and current status.",
    "    Think of it as a TL;DR that's always visible without loading all the content.",
    "    BAD:   same as summary (too short) or same as content (defeats the purpose)",
    "    GOOD:  'DataSync syncs inventory between two warehouses in real-time. Built on",
    "            Node.js + PostgreSQL 14. Auth module is in progress using JWT. Known issue:",
    "            batch requests over 500 items fail silently. Deployed to AWS us-east-1.'",
    "",
    "### FIND_SIMILAR — when to fire it:",
    "  find_similar(memory_id=<id>) searches nodes NOT directly linked to the given node.",
    "  Use it AFTER expand when:",
    "    - The neighborhood felt thin (few edges, low edge count)",
    "    - A multi-part question is only partially answered by what expand returned",
    "    - You suspect relevant context exists but isn't directly connected yet",
    "  Do NOT fire find_similar as a first step. Always query → expand first.",
    "  Results are discovery candidates — not confirmed relevant. Read summaries, decide.",
    "  If a result IS relevant: relate() to link it, or expand() to read it fully.",
    "",
    "### CUSTOM SCORING RULES:",
    "Trigger phrases: 'I want you to remember X' | 'Always track X' | 'Pay attention to X'",
    "  1. Acknowledge in ONE sentence",
    "  2. Save: summary='SCORING RULE: Track [topic]. Boost +3.', importance=10, tags=['scoring-rule']",
    "  3. Apply immediately and in all future sessions (loaded at session start)",
    "",
    "### SURFACING GATE — retrieved does not mean mentioned:",
    "  Only mention a memory if the current message is DIRECTLY about that topic.",
    "  User asks about weather → project node retrieved → STAY SILENT",
    "  User asks about DataSync → DataSync node retrieved → SURFACE IT",
    "  When in doubt: stay silent. The node is saved. It will surface when relevant.",
    "",
    "### UPDATE vs MERGE vs SPLIT — when to use each:",
    "  update()  → Something about an existing node changed. Status update, new detail, tag change.",
    "              Use append=True to add new information without losing old.",
    "              Use this instead of delete+save — update preserves all edges.",
    "",
    "  merge()   → Two nodes turn out to be about the same thing. Combine them.",
    "              keeper_id stays. absorb_id is deleted. All edges transferred.",
    "              Use when duplicate_detected fires or when you spot manual overlap.",
    "",
    "  split()   → One node grew to contain two independent ideas.",
    "              If a feature/pattern in node X is referenced by multiple other projects,",
    "              split it into its own node so each project can link to it separately.",
    "              part_a and part_b each get summary, content, tags.",
    "              Both parts inherit the original node's edges and link to each other.",
    "",
    "### CROSS-CUTTING PATTERN — shared features across projects:",
    "  If a feature, pattern, or preference appears in multiple projects:",
    "  1. Create a standalone node for the feature itself.",
    "  2. relate() each project to it with label='uses'.",
    "  3. expand(project_node, hops=2) will now pull in the feature's full content automatically.",
    "  Example: JWT auth used in DataSync AND InventoryAPI → one 'Auth: JWT pattern' node,",
    "           both projects link to it. Expanding either project surfaces JWT details.",
    "",
    "### CONTRADICTION RULE:",
    "  save() with dedup check handles most contradictions automatically.",
    "  If duplicate_detected: use update(append=True) to ADD new info, or merge() to combine.",
    "  For explicit conflicts (old info is wrong): update() the existing node directly.",
    "",
    "### SESSION TRACKING -- do this on every save:",
    "  session_id:   pass the current agent session ID (from your invocation context).",
    "  session_name: generate a 5-8 word human-readable label for this session.",
    "                e.g. 'DataSync auth JWT debugging' | 'onboarding setup walkthrough'",
    "  These fields are stored on the node and ONLY appear in expand() full output.",
    "  They are never returned by query() or list() (keeps the index clean).",
    "  When a user asks 'what were we working on before?' or 'can we go back to that session?'",
    "  -> expand the relevant node -> session field shows session_id + resume_hint.",
    "",
    "### EXACT CALL SYNTAX:",
    "QUERY:   universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"query\", \"summary\": \"DataSync project\", \"top_k\": 5})",
    "SAVE:    universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"save\", \"summary\": \"...\", \"content\": \"...\", \"importance\": 8, \"tags\": [\"project\"], \"session_id\": \"<id>\", \"session_name\": \"...\"})",
    "UPDATE:  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"update\", \"memory_id\": \"abc123\", \"content\": \"new detail\", \"append\": true})",
    "MERGE:   universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"merge\", \"keeper_id\": \"abc123\", \"absorb_id\": \"def456\"})",
    "SPLIT:   universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"split\", \"memory_id\": \"abc123\", \"part_a\": {\"summary\": \"...\", \"content\": \"...\", \"tags\": []}, \"part_b\": {\"summary\": \"...\", \"content\": \"...\", \"tags\": []}})",
    "RELATE:  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"relate\", \"source_id\": \"abc123\", \"target_id\": \"def456\", \"label\": \"uses\"})",
    "EXPAND:       universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"expand\", \"memory_id\": \"abc123\", \"hops\": 1})",
    "EXPAND+PAGE:  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"expand\", \"memory_id\": \"abc123\", \"content_limit\": 2000, \"content_offset\": 0})",
    "FIND_SIMILAR: universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"find_similar\", \"memory_id\": \"abc123\", \"top_k\": 5})",
    "LIST:    universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"list\", \"top_k\": 20})",
    "RULES:   universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"list\", \"filter_tags\": [\"scoring-rule\"], \"top_k\": 20})",
    "DELETE:  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"delete\", \"memory_id\": \"abc123\"})",
    "STATS:   universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"stats\"})",
    "",
    "### WORKED EXAMPLE — self-driving graph lifecycle:",
    "User: 'Track my projects. Building DataSync -- syncs inventory between warehouses.'",
    "→ Save rule: summary='SCORING RULE: Track user projects. Boost +3.' tags=['scoring-rule'] importance=10",
    "→ save(summary='Project: DataSync -- inventory sync, Node.js + PostgreSQL', content='...', importance=8)",
    "→ Response: success=True, auto_linked=[]. Graph now has 1 node.",
    "",
    "--- User: 'DataSync uses JWT for auth. Batch over 500 items fails silently.' ---",
    "→ save(summary='DataSync auth: JWT-based. Bug: batch > 500 fails silently.', content='...', importance=7)",
    "→ Response: success=True, auto_linked=[{id: DataSync_id, similarity: 0.74}]",
    "→ Auto-linked! No manual relate() needed. Edge created automatically.",
    "",
    "--- User mentions JWT auth on a second project, InventoryAPI ---",
    "→ save(summary='Project: InventoryAPI -- product lookup service. FastAPI + Redis.')",
    "→ Response: auto_linked=[{id: jwt_auth_id, similarity: 0.68}]",
    "→ Now JWT auth node is linked to BOTH projects.",
    "→ Recognize the cross-cutting pattern: split JWT auth into its own standalone node.",
    "→ split(memory_id=jwt_auth_id, part_a={summary:'Auth pattern: JWT...'}, part_b={summary:'DataSync auth bug...'})",
    "→ relate(jwt_standalone_id → DataSync_id, label='uses')",
    "→ relate(jwt_standalone_id → InventoryAPI_id, label='uses')",
    "",
    "--- User: 'Tell me about DataSync.' ---",
    "→ expand(memory_id=DataSync_id, hops=2)",
    "→ DataSync: FULL content. JWT auth node: FULL content. InventoryAPI: summary only.",
    "→ Full story surfaced automatically via graph traversal.",
    "",
    "--- User: 'DataSync is now using Kafka instead of Node.js.' ---",
    "→ update(memory_id=DataSync_id, content='Now uses Kafka...', append=True)",
    "→ Edges untouched. ID unchanged. Content updated in place.",
    "",
    "--- User saves something about 'DataSync inventory sync project' ---",
    "→ save(summary='DataSync: inventory sync tool, switching to Kafka')",
    "→ Response: duplicate_detected=True, existing_id=DataSync_id, similarity=0.91",
    "→ DO NOT force-save. Call update(memory_id=DataSync_id, append=True) instead.",
    "",
    "--- User: 'What time is it?' ---",
    "→ query() fires silently. Surfacing gate: not about projects. STAY SILENT.",
    "",
    "### EXECUTION RULES:",
    "- All cyber_memory calls are SILENT — never announced unless user explicitly asks",
    "- Session-start fires in FIRST tool-call wave, not lazily later",
    "- NEVER save PII (SSN, passwords, HIPAA data, financial credentials)",
    "- kennel_remember only for explicit user requests ('remember this...')",
]

SECTION_START_MARKER = "## MEMORY PROTOCOL"
SECTION_END_MARKERS  = ["═══", "## YOUR MISSION", "## YOUR TOOLS", "## YOUR APPROACH"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_code_puppy_dir() -> Path:
    home = Path.home()
    candidate = home / ".code_puppy"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find .code_puppy at {candidate}\n"
        "Is your agent installed and configured?"
    )


def find_agent_jsons(code_puppy_dir: Path) -> list[Path]:
    agents_dir = code_puppy_dir / "agents"
    if not agents_dir.exists():
        return []
    return sorted(agents_dir.glob("*.json"))


def install_plugin(code_puppy_dir: Path, dry_run: bool = False) -> Path:
    uc_dir  = code_puppy_dir / "plugins" / "universal_constructor"
    dest    = uc_dir / "cyber_memory.py"
    src     = Path(__file__).parent / "cyber_memory.py"

    if not src.exists():
        raise FileNotFoundError(f"cyber_memory.py not found next to install.py at {src}")

    if dry_run:
        print(f"  [DRY RUN] Would copy:\n    {src}\n    → {dest}")
        return dest

    uc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    # bust UC's pyc cache if it exists
    pycache = uc_dir / "__pycache__"
    for pyc in pycache.glob("cyber_memory*.pyc"):
        pyc.unlink()

    return dest


def patch_agent(agent_path: Path, dry_run: bool = False) -> bool:
    """
    Find (or insert) the ## MEMORY PROTOCOL section in an agent JSON file
    and replace it with the current MEMORY_PROTOCOL_LINES.

    Agent JSON format expected:
        {
          "system_prompt": ["line1", "line2", ...]
        }
    """
    with open(agent_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "system_prompt" not in data:
        print(f"    Skipping {agent_path.name} — no 'system_prompt' key found.")
        return False

    sp = data["system_prompt"]

    # Find existing protocol section boundaries
    start_idx = None
    end_idx   = None

    for i, line in enumerate(sp):
        if start_idx is None and SECTION_START_MARKER in line:
            start_idx = i
        elif start_idx is not None and i > start_idx:
            if any(line.startswith(m) for m in SECTION_END_MARKERS):
                end_idx = i
                break

    if start_idx is not None and end_idx is not None:
        action_taken = f"Replace existing protocol (lines {start_idx}–{end_idx - 1})"
        if not dry_run:
            sp[start_idx:end_idx] = MEMORY_PROTOCOL_LINES + [""]
    elif start_idx is not None:
        # Found start but no clean end — replace to end of array
        action_taken = f"Replace existing protocol from line {start_idx} (no end marker found)"
        if not dry_run:
            sp[start_idx:] = MEMORY_PROTOCOL_LINES + [""]
    else:
        # No existing protocol — insert before the separator before YOUR MISSION
        insertion_idx = None
        for i, line in enumerate(sp):
            if "## YOUR MISSION" in line:
                # Walk back to find the nearest separator line
                for j in range(i - 1, max(0, i - 6), -1):
                    if sp[j].startswith("═══"):
                        insertion_idx = j
                        break
                if insertion_idx is None:
                    insertion_idx = i
                break

        if insertion_idx is None:
            insertion_idx = len(sp)

        action_taken = f"Insert fresh protocol at line {insertion_idx}"
        if not dry_run:
            sp[insertion_idx:insertion_idx] = [""] + MEMORY_PROTOCOL_LINES + [""]

    data["system_prompt"] = sp

    if dry_run:
        print(f"  [DRY RUN] {agent_path.name}: {action_taken}")
        print(f"            Protocol lines: {len(MEMORY_PROTOCOL_LINES)}")
        return True

    # Validate JSON before writing
    new_json = json.dumps(data, indent=2, ensure_ascii=False)
    json.loads(new_json)  # raises if invalid

    # Backup original
    backup = agent_path.with_suffix(".json.bak")
    shutil.copy2(agent_path, backup)

    with open(agent_path, "w", encoding="utf-8") as f:
        f.write(new_json)

    print(f"    {agent_path.name}: {action_taken}")
    print(f"     Backup saved to: {backup.name}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Install cyber_memory into your agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install.py                   # fully interactive
  python install.py --dry-run         # preview all changes, write nothing
  python install.py --agent my.json   # patch a specific agent file
  python install.py --skip-agent      # install plugin only
        """,
    )
    parser.add_argument("--dry-run",    action="store_true", help="Preview changes without writing")
    parser.add_argument("--agent",      metavar="PATH",      help="Path to a specific agent JSON to patch")
    parser.add_argument("--skip-agent", action="store_true", help="Install plugin only, skip agent patching")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║   cyber_memory installer  v6.1.0            ║")
    print("╚══════════════════════════════════════════════╝\n")

    # 1. Locate Code Puppy
    try:
        cp_dir = find_code_puppy_dir()
    except FileNotFoundError as e:
        print(f" {e}")
        sys.exit(1)

    print(f"  Agent dir: {cp_dir}")

    # 2. Install plugin
    print("\n── Step 1: Install plugin ──────────────────────")
    try:
        dest = install_plugin(cp_dir, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"    Plugin installed to:\n     {dest}")
    except Exception as e:
        print(f"    {e}")
        sys.exit(1)

    if args.skip_agent:
        print("\n  Skipping agent patching (--skip-agent).")
        print("\n Done. Restart your agent to activate cyber_memory.\n")
        return

    # 3. Select agent(s) to patch
    print("\n── Step 2: Patch agent system prompt ───────────")

    if args.agent:
        agent_paths = [Path(args.agent).expanduser().resolve()]
    else:
        available = find_agent_jsons(cp_dir)
        if not available:
            print("  No agent JSON files found in ~/.code_puppy/agents/. Use --agent PATH to specify one.")
            print("  Use --agent PATH to specify one manually.")
            print("\n Plugin installed. Agent patching skipped.\n")
            return

        print("  Found agent files:")
        for i, p in enumerate(available):
            print(f"    [{i}] {p.name}")
        print(f"    [a] All agents")
        print(f"    [s] Skip")

        choice = input("\n  Which agent(s) to patch? [0/1/.../a/s]: ").strip().lower()

        if choice == "s":
            print("\n  Skipping agent patching.")
            print("\n Plugin installed. Restart your agent to activate.\n")
            return
        elif choice == "a":
            agent_paths = available
        elif choice.isdigit() and int(choice) < len(available):
            agent_paths = [available[int(choice)]]
        else:
            print("  Invalid choice. Skipping agent patching.")
            print("\n Plugin installed. Restart your agent to activate.\n")
            return

    print()
    for agent_path in agent_paths:
        patch_agent(agent_path, dry_run=args.dry_run)

    # 4. Done
    print()
    if args.dry_run:
        print(" Dry run complete. No files were written.")
        print("  Run without --dry-run to apply changes.\n")
    else:
        print(" Done.")
        print("  → Restart your agent to activate cyber_memory.")
        print("  → On first message, the agent will fire its session-start memory queries.\n")


if __name__ == "__main__":
    main()
