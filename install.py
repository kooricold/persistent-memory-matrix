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
MEMORY_PROTOCOL_LINES = [
    "## MEMORY PROTOCOL (Persistent Context Layer):",
    "",
    "You maintain a persistent KNOWLEDGE GRAPH across ALL sessions via cyber_memory v5.2.0.",
    "Memory is NOT a flat list. It is a graph of NODES connected by labeled EDGES.",
    "",
    "EACH NODE HAS TWO LAYERS:",
    "  summary  (80-150 chars) -- searched by query(), returned by query(), used in edge previews.",
    "                             General context. Enough to identify relevance.",
    "  content  (unlimited)    -- full rich detail. ONLY returned by expand().",
    "                             Loaded only when the node is confirmed relevant.",
    "",
    "  kennel_remember -- SECONDARY only: for explicit user requests ('remember this')",
    "",
    "### SESSION START — fire ONCE, on your VERY FIRST tool-call wave:",
    "Before you do anything else, make TWO calls in parallel:",
    "  CALL 1 (scoring rules): universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"list\", \"filter_tags\": [\"scoring-rule\"], \"top_k\": 20})",
    "  CALL 2 (find context):  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"query\", \"summary\": \"<topic of first message>\", \"top_k\": 5})",
    "query() returns SUMMARIES + edge previews only — never full content.",
    "Edge previews show graph shape. Hold scoring rules in working memory. Do NOT announce this.",
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
    "  hops=1: root gets FULL content. Direct neighbors get summary + edge map only.",
    "  hops=2: root + 1st-hop nodes get FULL content. 2nd-hop gets summary only.",
    "  Leaf nodes are always summary-only — expand them individually if needed.",
    "  You never load the whole graph. Cost scales only with what you open.",
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
    "### AFTER EVERY RESPONSE — score, save, relate:",
    "Timing: AFTER composing response, BEFORE the next user message.",
    "",
    "SCORING (base + rule boost):",
    "  9-10 CRITICAL → save immediately: explicit prefs, corrections, hard constraints",
    "  7-8  HIGH     → save: named tech, tools, workflows, people, decisions",
    "  5-6  MEDIUM   → save if likely to recur: projects, systems, context",
    "  1-4  LOW      → discard: greetings, filler, one-off trivia",
    "  Active scoring rules: +3 to base score when topic matches (cap at 10)",
    "",
    "AFTER SAVING — relate the new node to existing ones:",
    "  Good labels: 'sub-feature-of', 'built-with', 'owned-by', 'depends-on', 'member-of', 'related', 'blocks'",
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
    "### CONTRADICTION RULE:",
    "  Before saving importance 7+, query first. If conflict found: delete old, save updated, re-link.",
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
    "SAVE:    universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"save\", \"summary\": \"Project: DataSync -- inventory sync, Node.js + PostgreSQL\", \"content\": \"Full detail here...\", \"importance\": 8, \"tags\": [\"project\"], \"session_id\": \"<current session id>\", \"session_name\": \"DataSync auth debugging\"})",
    "RELATE:  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"relate\", \"source_id\": \"abc123\", \"target_id\": \"def456\", \"label\": \"built-with\"})",
    "EXPAND:  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"expand\", \"memory_id\": \"abc123\", \"hops\": 1})",
    "LIST:    universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"list\", \"top_k\": 20})",
    "RULES:   universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"list\", \"filter_tags\": [\"scoring-rule\"], \"top_k\": 20})",
    "DELETE:  universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"delete\", \"memory_id\": \"abc123\"})",
    "STATS:   universal_constructor(action='call', tool_name='cyber_memory', tool_args={\"action\": \"stats\"})",
    "",
    "### WORKED EXAMPLE — full graph lifecycle:",
    "User: 'Track my projects. Building DataSync -- syncs inventory between warehouses.'",
    "→ Save rule: summary='SCORING RULE: Track user projects. Boost +3.' tags=['scoring-rule'] importance=10",
    "→ Save node A: summary='Project: DataSync -- inventory sync, Node.js + PostgreSQL'",
    "               content='Syncs product inventory between Warehouse A and B. Auth WIP.'",
    "               session_id='<current id>' session_name='DataSync project kickoff'",
    "--- User: 'DataSync uses JWT for auth. Batch over 500 items fails silently.' ---",
    "→ Save node B: summary='DataSync auth: JWT-based, in progress. Bug: batch > 500 fails.'",
    "               content='Auth module using JWT. Currently in development. Batch > 500 silent failure.'",
    "               session_id='<current id>' session_name='DataSync auth debugging'",
    "→ relate(B_id → A_id, label='sub-feature-of')",
    "--- Next session: 'What projects do I have?' ---",
    "→ query(summary='user projects') → returns summary of A + edge preview for B",
    "→ Response: 'DataSync -- inventory sync tool.' Full content of A never loaded.",
    "--- User: 'What is the DataSync auth status?' ---",
    "→ expand(memory_id=A_id, hops=1)",
    "→ A: FULL content loaded + session: {session_name: 'DataSync project kickoff', resume_hint: '...'}",
    "→ B: summary + edge map only. Session info NOT included (B not yet expanded).",
    "--- User: 'Can we go back to the session where we fixed the auth bug?' ---",
    "→ expand(memory_id=B_id, hops=1)",
    "→ B: FULL content + session: {session_name: 'DataSync auth debugging', session_id: '...', resume_hint: 'Resume session...'}",
    "→ Surface the session_id to user so they can resume it.",
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
    print("║   cyber_memory installer  v5.1.0            ║")
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
