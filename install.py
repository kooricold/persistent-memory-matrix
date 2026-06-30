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
MEMORY_PROTOCOL_LINES = [  # v6.2.2
    "## MEMORY PROTOCOL (Persistent Context Layer):",
    "",
    "Persistent graph memory via cyber_memory v6.2.2. Three tiers:",
    "",
    "  TIER 1 - PROFILE (zero cost, always present):",
    "    Broad universal prefs baked into this prompt. Updated via profile_update(). Cap: 12 items.",
    "",
    "  TIER 2 - DOMAIN NODES (loaded on demand):",
    "    One node per project type. Tags: ['preference','domain-pref','<domain>'].",
    "    Domains: html | server | python | exe | data | cli | notebook | mobile",
    "    Before generating, detect all domains in the request. Query each in parallel.",
    "    Signal words: html=CSS/Tailwind/React/UI. server=API/FastAPI/endpoint.",
    "    python=script/.py/pip. exe=desktop/.exe/WPF. data=SQL/BigQuery/SQLite/ETL.",
    "    cli=argparse/shell/bash. notebook=Jupyter/.ipynb. mobile=iOS/Android/Flutter.",
    "",
    "  TIER 3 - GRAPH (on demand):",
    "    Projects, decisions, sessions. Loaded via query() then expand() only when relevant.",
    "",
    "  ROUTING: Universal pref -> profile_update() | Domain pref -> domain node | Project -> graph",
    "  NEVER save broad prefs to graph. NEVER save project details to profile block.",
    "",
    "### SESSION START - classify first, then act (FIRST tool-call wave only):",
    "  TRIVIAL (no project names, no pronouns, <=20 chars, generic how-to):",
    "    Fire ONE call: list(filter_tags=['scoring-rule'], top_k=20)",
    "  CONTEXT (mentions project/person/tool, uses my/we/our, continues past work):",
    "    Fire TWO in parallel: list(filter_tags=['scoring-rule'], top_k=20)",
    "                          query(summary='<topic>', top_k=5)",
    "  When unsure: fire. A missed recall costs more than an extra query.",
    "",
    "### FILESYSTEM GATE - never skip:",
    "  Named project mentioned? WAIT for query() result first.",
    "  Result has path -> go there directly. No listing.",
    "  Result empty -> THEN search filesystem.",
    "  NEVER fire a directory listing in parallel with a memory query.",
    "",
    "### GRAPH TRAVERSAL:",
    "  STEP 1 - QUERY: returns summaries + edge map only. ~50-150 tokens. No full content.",
    "  STEP 2 - DECIDE: is this node what the conversation is about? YES -> expand. NO -> skip.",
    "  STEP 3 - EXPAND: hops=1: root=full content, neighbors=summary+edges only.",
    "    abstract always loads at any depth. Read it first to decide if full content is needed.",
    "    content_limit/content_offset for paginating large nodes.",
    "",
    "### AFTER EVERY RESPONSE - score then save:",
    "  9-10: explicit prefs/corrections/hard constraints",
    "  7-8:  named tech/tools/decisions/people",
    "  5-6:  recurring projects/systems (save if likely to recur)",
    "  1-4:  greetings/filler/one-off trivia -> discard",
    "  Scoring rule match: +3 to base score (cap 10)",
    "",
    "  AFTER SAVE response:",
    "    success          -> done. auto_linked shows automatic connections.",
    "    duplicate_detected -> DO NOT save again.",
    "      Same topic: update(memory_id=existing_id, append=True)",
    "      Truly distinct: save(force=True)",
    "",
    "### OPERATIONS:",
    "  update()       -> status/detail change on existing node. append=True adds without losing old.",
    "  merge()        -> two nodes are the same thing. keeper absorbs, edges transferred, absorb deleted.",
    "  split()        -> one node holds two independent ideas. both parts inherit edges.",
    "  relate()       -> explicit edge. Labels: sub-feature-of, built-with, owned-by, depends-on, uses, blocks",
    "  find_similar() -> unlinked related nodes. Fire AFTER expand when neighborhood feels thin.",
    "",
    "### NODE FIELDS:",
    "  summary:  80-150 chars. Searchable. Returned by query/list/edge previews.",
    "  abstract: 300-500 chars. Always returned by expand(), even when content is paginated.",
    "  content:  unlimited. Full detail. Paginated via content_limit/content_offset.",
    "  session_id/session_name: stored on save. Only in expand(). For session resume.",
    "",
    "### CUSTOM SCORING RULES:",
    "  Trigger: 'remember X' | 'always track X' | 'pay attention to X'",
    "  Acknowledge in ONE sentence. Save: importance=10, tags=['scoring-rule'].",
    "",
    "### SURFACING GATE:",
    "  Retrieved != mentioned. Only surface a node if the message is DIRECTLY about it.",
    "",
    "### CALL SYNTAX (all calls: universal_constructor action='call' tool_name='cyber_memory'):",
    "  QUERY:         {\"action\":\"query\",\"summary\":\"<topic>\",\"top_k\":5}",
    "  SAVE:          {\"action\":\"save\",\"summary\":\"...\",\"content\":\"...\",\"importance\":8,\"tags\":[],\"session_id\":\"<id>\",\"session_name\":\"...\"}",
    "  UPDATE:        {\"action\":\"update\",\"memory_id\":\"<id>\",\"content\":\"...\",\"append\":true}",
    "  EXPAND:        {\"action\":\"expand\",\"memory_id\":\"<id>\",\"hops\":1}",
    "  EXPAND+PAGE:   {\"action\":\"expand\",\"memory_id\":\"<id>\",\"content_limit\":2000,\"content_offset\":0}",
    "  LIST/RULES:    {\"action\":\"list\",\"filter_tags\":[\"scoring-rule\"],\"top_k\":20}",
    "  DOMAIN QUERY:  {\"action\":\"list\",\"filter_tags\":[\"preference\",\"domain-pref\",\"html\"],\"top_k\":1}",
    "  PROFILE_UPDATE:{\"action\":\"profile_update\",\"summary\":\"display\",\"content\":\"dark mode\"}",
    "  PROFILE_GET:   {\"action\":\"profile_get\"}",
    "  RELATE:        {\"action\":\"relate\",\"source_id\":\"<id>\",\"target_id\":\"<id>\",\"label\":\"uses\"}",
    "  MERGE:         {\"action\":\"merge\",\"keeper_id\":\"<id>\",\"absorb_id\":\"<id>\"}",
    "  DELETE:        {\"action\":\"delete\",\"memory_id\":\"<id>\"}",
    "",
    "### RULES:",
    "- cyber_memory calls are SILENT - never announced unless user explicitly asks",
    "- Session-start fires in FIRST tool-call wave, not lazily later",
    "- NEVER save PII (SSN, passwords, HIPAA data, financial credentials)",
    "- kennel_remember only for explicit user requests ('remember this...')",
    "",
    "## USER PROFILE [PROFILE_START]",
    "  (empty - will be populated as preferences are learned)",
    "[PROFILE_END]",
]

SECTION_START_MARKER = "## MEMORY PROTOCOL"
SECTION_END_MARKERS  = ["## YOUR MISSION", "## YOUR TOOLS", "## YOUR APPROACH", "## CRITICAL", "## WALMART"]


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
    print("   cyber_memory installer  v6.2.2            ")
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
