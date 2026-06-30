# cyber_memory

Graph-based persistent memory for LLM agents.

Instead of loading all memory at once (expensive, slow, token-heavy), cyber_memory
builds a knowledge graph of nodes and edges. You query cheap summaries to find what
you need, then expand selectively into rich detail only when the conversation actually
requires it.

---

## The Problem It Solves

Standard flat memory: "give me everything relevant" loads hundreds of memories every
session. Token cost is high. Most of it is noise for any given conversation.

cyber_memory: "give me the index, then drill down into what I actually need." A
question about weather never loads your project notes. A question about DataSync
loads only DataSync and its direct relations -- not every other project.

---

## How the Graph Works

Every memory is a **node**. Nodes connect to other nodes through labeled **edges**.
There are no fixed tiers, no boxes. A node can relate to anything that genuinely
connects to it.

```
[User: always FastAPI, never Flask]
        |
        | preference-of
        v
[Project: DataSync] ----sub-feature-of----> [Auth Module: JWT, WIP, bug batch > 500]
        |                                            |
        | built-with                                 | depends-on
        v                                            v
[Node.js + PostgreSQL]                    [jsonwebtoken v9]
        |
        | hosted-on
        v
[AWS us-east-1 -- prod environment]
```

**query()** is cheap -- returns node summaries with 80-char edge previews. You can
read the shape of the graph without loading neighbor content.

**expand()** opens one node's full neighborhood. hops=1 loads the root + direct
neighbors. hops=2 goes one level deeper. You cap at what the question actually needs.

**relate()** links two nodes with a label you choose. Labels are free-form strings:
sub-feature-of, built-with, owns, depends-on, member-of, blocks, related, uses.

---

## Requirements

- An agent system that supports the Universal Constructor plugin interface
- Python 3.10+
- scikit-learn (for TF-IDF search -- usually already present in your environment)

Optional upgrade (auto-detected if available):

- sentence-transformers with all-MiniLM-L6-v2 cached locally -- enables true
  semantic search instead of TF-IDF. The tool auto-upgrades silently if the model
  is cached. Falls back to TF-IDF if not.

---

## Installation

```
git clone https://github.com/kooricold/persistent-memory-matrix
cd persistent-memory-matrix
python install.py
```

The installer will:

1. Copy `cyber_memory.py` into your Universal Constructor plugin folder
   (`~/.code_puppy/plugins/universal_constructor/`)
2. List your available agent JSON files and let you choose which to patch
3. Inject the full memory protocol into the selected agent's `system_prompt`
4. Back up the original agent file before modifying it

Then restart your agent. The memory graph is live on next session start.

### Flags

```
python install.py                   # fully interactive
python install.py --dry-run         # preview all changes, write nothing
python install.py --agent PATH      # patch a specific agent file directly
python install.py --skip-agent      # install plugin only, don't patch any agent
```

### Manual install (no script)

1. Copy `cyber_memory.py` to `~/.code_puppy/plugins/universal_constructor/`
2. Add the contents of the `MEMORY_PROTOCOL_LINES` list in `install.py` into your
   agent's `system_prompt` array
3. Restart your agent

---

## Action Reference

| Action   | Required args              | What it does                                           | Cost   |
|----------|----------------------------|--------------------------------------------------------|--------|
| `save`   | `content`, `importance`    | Store a new node                                       | low    |
| `query`  | `content`                  | TF-IDF search, returns node summaries + edge previews  | low    |
| `relate` | `source_id`, `target_id`, `label` | Link two nodes bidirectionally                | low    |
| `expand` | `memory_id`                | Load root + N hops of neighbors (default hops=1)       | medium |
| `list`   | --                         | All nodes sorted by importance                         | low    |
| `stats`  | --                         | Graph overview (nodes, edges, mode)                    | low    |
| `delete` | `memory_id`                | Remove node + clean all dangling edges                 | low    |

All calls go through the Universal Constructor:

```
universal_constructor(
    action='call',
    tool_name='cyber_memory',
    tool_args={"action": "query", "content": "DataSync project", "top_k": 5}
)
```

---

## Node Content Guidelines

A node must make sense when read in complete isolation. No shorthand.

```
BAD:   "DataSync"
BAD:   "User likes FastAPI"

GOOD:  "Project: DataSync -- syncs product inventory between Warehouse A and B
        in real-time. Node.js backend, PostgreSQL database. Auth module in progress."

GOOD:  "User pref: FastAPI for all Python backends. Never Flask.
        In use since 2022. Preference extends to all new projects."
```

---

## How the Protocol Operates

### Session Start

On the first tool-call wave of every conversation, the agent fires two parallel calls:

1. Load scoring rules: `list` with `filter_tags=["scoring-rule"]`
2. Find relevant context: `query` with the topic of the first user message

Scoring rules are held in working memory for the session. They boost the importance
score of matching content by +3 (capped at 10).

### After Every Response

The agent scores what it just learned:

- 9-10 CRITICAL: explicit preferences, corrections, hard constraints
- 7-8 HIGH: named tech, tools, decisions, key people
- 5-6 MEDIUM: named projects, systems, reusable context
- 1-4 LOW: discard (greetings, filler, one-off questions)

Active scoring rules add +3 to the base score. If score >= 5, the agent saves a node.
After saving, it calls `relate()` to connect the new node to any existing nodes it
relates to.

### Custom Scoring Rules

User says: "I want you to remember my projects and what they do."
Agent response: "Got it -- I'll track your projects going forward." (one sentence)
Agent saves: `SCORING RULE: Track all user projects. Boost +3 when projects mentioned.`
             tags=["scoring-rule"], importance=10

This rule persists forever. It loads at session start automatically.

### Surfacing Gate

Retrieval (running query) and surfacing (mentioning in response) are two separate
decisions. A node being retrieved does NOT mean it gets mentioned.

```
User asks about weather
  -> query() runs silently
  -> project node retrieved with low relevance
  -> surfacing gate: is this about a project? NO
  -> answer weather, nothing surfaced

User asks about DataSync
  -> query() returns DataSync node with high relevance
  -> surfacing gate: is this about DataSync? YES
  -> surface the DataSync context in response
```

When in doubt, stay silent. The node is saved. It surfaces when the conversation
actually calls for it.

---

## Typical Token Budget

| Scenario                            | Calls                  | Approx tokens |
|-------------------------------------|------------------------|---------------|
| Session start (10 nodes in graph)   | list + query           | ~200-400      |
| "What projects do I have?"          | query                  | ~150-300      |
| "Tell me about DataSync"            | expand hops=1          | ~400-800      |
| "DataSync auth status?"             | expand hops=1          | ~300-600      |
| "What time is it?" (unrelated)      | query, nothing surfaced | ~50           |

Compare with naive "load everything": 50 nodes could easily be 5,000-15,000 tokens
every session regardless of what's being asked.

---

## Graph Storage

All data lives in `~/.code_puppy/cyber_memory/memories.json`. It's a flat JSON array
of node objects. No database required.

```json
[
  {
    "id": "7c0d0cc7",
    "content": "Project: DataSync -- syncs product inventory...",
    "importance": 8,
    "tags": ["project"],
    "relations": [
      {
        "target_id": "70be1cfe",
        "label": "sub-feature-of",
        "preview": "DataSync auth module: JWT-based authentication..."
      }
    ],
    "timestamp": "2026-06-15T09:30:00Z"
  }
]
```

Relations are stored on both ends of the edge (bidirectional). Previews are
80-char snippets of the neighbor's content -- enough to evaluate relevance without
loading the full node.

---

## Search Engine

The tool has three search modes, used in priority order:

1. **sentence-transformers** (best): True semantic similarity. Auto-used if
   `all-MiniLM-L6-v2` is cached in `~/.code_puppy/cyber_memory/model_cache/`.
   No internet needed if cached.

2. **TF-IDF via scikit-learn** (default): Keyword + n-gram similarity. Good
   enough for most cases. No downloads required.

3. **Hash fallback**: Pure Python, no dependencies. Used if scikit-learn
   is unavailable. Rough but functional.

Run `stats` to see which mode is active:

```
universal_constructor(action='call', tool_name='cyber_memory', tool_args={"action": "stats"})
```

---

## Upgrading

To upgrade the plugin, pull the latest version and re-run:

```
git pull
python install.py --skip-agent
```

---

## Contributing

The entire system is one Python file (`cyber_memory.py`) and one installer
(`install.py`). Both are self-contained and dependency-light by design.

Pull requests welcome. Keep it dependency-light.
