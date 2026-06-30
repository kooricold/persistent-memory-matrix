
TOOL_META = {
    'name': 'cyber_memory',
    'namespace': '',
    'description': (
        "Graph-based persistent memory. Nodes have a short SUMMARY (searched + returned by query) "
        "and full CONTENT (only loaded by expand). Query is cheap — returns summaries + edge previews. "
        "Expand is selective — loads full content of root + inner hops, summaries only at leaf level."
    ),
    'enabled': True,
    'version': '5.2.0',
    'author': 'user',
    'created_at': '2026-06-15T09:27:06.828478'
}


def cyber_memory(
    action: str,
    summary: str = "",
    content: str = "",
    top_k: int = 5,
    importance: int = 5,
    tags: list = None,
    memory_id: str = None,
    filter_tags: list = None,
    source_id: str = None,
    target_id: str = None,
    label: str = "",
    hops: int = 1,
    session_id: str = "",
    session_name: str = "",
) -> dict:
    """
    Graph-based persistent memory for LLM agents. v5.2.0

    TWO-LAYER NODE MODEL:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  summary  (80-150 chars)  -- searched by query(), returned by        │
    │                              query(), used in edge previews           │
    │  content  (unlimited)     -- full rich detail, ONLY returned by      │
    │                              expand() on inner/root nodes             │
    └──────────────────────────────────────────────────────────────────────┘

    EXAMPLE NODE:
      summary: "Project: DataSync -- inventory sync tool, Node.js + PostgreSQL"
      content: "Project: DataSync -- syncs product inventory between Warehouse A
                and B in real-time. Node.js backend, PostgreSQL 14. Auth module
                in progress (JWT). Known bug: batch > 500 fails silently.
                Owned by alice. Started 2026. Deployed to prod."

    QUERY vs EXPAND:
      query()  -- cheap: searches summary, returns summary + edge previews
      expand() -- selective: root + inner hops get full content;
                  leaf layer (outermost hop) gets summary + edge map only

    GRAPH STRUCTURE:
      [DataSync] --sub-feature-of--> [Auth Module: JWT, WIP]
      [DataSync] --built-with------> [Node.js + PostgreSQL]
      [Auth]     --depends-on------> [jsonwebtoken v9]

    Actions:
    - save:   Store a node with summary + content.
    - query:  Search summaries. Returns summary + edge previews only.
    - relate: Link two nodes bidirectionally with a labeled edge.
    - expand: Root + inner hops get full content. Leaf layer gets summary only.
    - list:   All nodes by importance. Returns summaries only.
    - stats:  Graph overview.
    - delete: Remove node + clean all dangling edges.

    Args:
        action:       save | query | relate | expand | list | stats | delete
        summary:      Short searchable description (80-150 chars). Used by query.
        content:      Full rich detail. Only returned by expand.
        top_k:        Max results for query/list (default 5)
        importance:   1-10 for save (default 5)
        tags:         Tags for save (e.g. ['project', 'tech'])
        memory_id:    Node ID for expand/delete
        filter_tags:  Filter list/query to nodes with ALL these tags
        source_id:    relate -- source node ID
        target_id:    relate -- target node ID
        label:        relate -- edge label (e.g. 'sub-feature-of', 'built-with')
        hops:         expand -- depth. 1 = root full + direct neighbors summary.
                      2 = root + 1st hop full + 2nd hop summary. Max 3.
        session_id:   Agent session ID for this save (pass current session ID).
                      Stored on node; visible on expand. Enables session resumption.
        session_name: Short human-readable description of this session (5-8 words).
                      e.g. 'DataSync auth JWT debugging' or 'onboarding setup walkthrough'
              Agent generates this. Stored on node; visible on expand.

    Edge label suggestions:
        sub-feature-of, built-with, owned-by, depends-on, member-of,
        blocks, part-of, related, uses, defined-by, influences
    """
    import json
    import hashlib
    import math
    from pathlib import Path
    from datetime import datetime
    import uuid

    MEMORY_DIR    = Path.home() / ".code_puppy" / "cyber_memory"
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_FILE = MEMORY_DIR / "memories.json"
    MODEL_CACHE   = MEMORY_DIR / "model_cache"

    nodes = []
    if MEMORIES_FILE.exists():
        try:
            with open(MEMORIES_FILE, "r", encoding="utf-8") as f:
                nodes = json.load(f)
        except Exception:
            nodes = []

    def persist():
        with open(MEMORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=2, ensure_ascii=False)

    def get_node(nid):
        return next((n for n in nodes if n["id"] == nid), None)

    def get_summary(n):
        """Return the summary field, falling back to first 150 chars of content for old nodes."""
        s = n.get("summary", "").strip()
        if s:
            return s
        return n.get("content", "")[:150].strip()

    def get_content(n):
        """Return full content, falling back to summary for old nodes."""
        c = n.get("content", "").strip()
        if c:
            return c
        return n.get("summary", "").strip()

    # ── Similarity engines (auto-upgrade path) ──────────────────────

    def _try_st(texts):
        try:
            from sentence_transformers import SentenceTransformer
            MODEL_CACHE.mkdir(exist_ok=True)
            m = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=str(MODEL_CACHE),
                local_files_only=True,
            )
            return m.encode(texts), "sentence-transformers"
        except Exception:
            return None, None

    def _tfidf(query_text, corpus):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            docs = corpus + [query_text]
            v = TfidfVectorizer(
                analyzer="word", stop_words="english", ngram_range=(1, 2), min_df=1
            )
            mat = v.fit_transform(docs)
            return cosine_similarity(mat[-1], mat[:-1])[0].tolist(), "tfidf"
        except Exception:
            return None, None

    def _hash(query_text, corpus):
        def vec(t):
            d = {}
            for w in t.lower().split():
                k = int(hashlib.md5(w.encode()).hexdigest(), 16) % 1024
                d[k] = d.get(k, 0) + 1
            return d
        def cos(a, b):
            keys = set(a) | set(b)
            dot  = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
            na   = math.sqrt(sum(v ** 2 for v in a.values()))
            nb   = math.sqrt(sum(v ** 2 for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0
        qv = vec(query_text)
        return [cos(qv, vec(d)) for d in corpus], "hash-fallback"

    def get_similarities(query_text, corpus):
        embs, mode = _try_st(corpus + [query_text])
        if embs is not None:
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                return (
                    cosine_similarity(embs[-1].reshape(1, -1), embs[:-1])[0].tolist(),
                    mode,
                )
            except Exception:
                pass
        s, mode = _tfidf(query_text, corpus)
        if s is not None:
            return s, mode
        return _hash(query_text, corpus)

    def node_as_summary(n):
        """Lightweight view: summary + edge map. Never includes full content."""
        rels = n.get("relations", [])
        return {
            "id":         n["id"],
            "summary":    get_summary(n),
            "importance": n["importance"],
            "tags":       n["tags"],
            "timestamp":  n["timestamp"],
            "edge_count": len(rels),
            "relations":  [
                {
                    "label":     r["label"],
                    "target_id": r["target_id"],
                    "preview":   r.get("preview", ""),
                }
                for r in rels
            ],
        }

    def node_as_full(n):
        """Full view: summary + content + session info + edge map. Used by expand on inner nodes."""
        rels = n.get("relations", [])
        sid  = n.get("session_id", "").strip()
        sname = n.get("session_name", "").strip()
        session = {}
        if sid or sname:
            session = {
                "session_id":   sid,
                "session_name": sname,
                "resume_hint":  f"Resume session '{sname}' with session_id: {sid}" if sid and sname
                                else f"Session: {sname or sid}",
            }
        result = {
            "id":         n["id"],
            "summary":    get_summary(n),
            "content":    get_content(n),
            "importance": n["importance"],
            "tags":       n["tags"],
            "timestamp":  n["timestamp"],
            "edge_count": len(rels),
            "relations":  [
                {
                    "label":     r["label"],
                    "target_id": r["target_id"],
                    "preview":   r.get("preview", ""),
                }
                for r in rels
            ],
        }
        if session:
            result["session"] = session
        return result

    # ═══════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════
    if action == "save":
        if not summary or not summary.strip():
            return {
                "success": False,
                "error":   "summary required (80-150 chars, searchable, self-identifying)",
            }
        node = {
            "id":           str(uuid.uuid4())[:8],
            "summary":      summary.strip(),
            "content":      content.strip() if content else "",
            "importance":   max(1, min(10, int(importance))),
            "tags":         tags or [],
            "relations":    [],
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "session_id":   session_id.strip() if session_id else "",
            "session_name": session_name.strip() if session_name else "",
        }
        nodes.append(node)
        persist()
        return {
            "success":     True,
            "id":          node["id"],
            "importance":  node["importance"],
            "tags":        node["tags"],
            "total_nodes": len(nodes),
            "hint":        "Call relate() to connect this node to others.",
        }

    # ═══════════════════════════════════════════════════════
    # QUERY -- searches summaries, returns summaries + edge previews only
    # ═══════════════════════════════════════════════════════
    elif action == "query":
        if not content and not summary:
            return {"success": False, "error": "provide content or summary as search text"}
        search_text = (summary or content).strip()

        pool = nodes
        if filter_tags:
            req  = set(filter_tags)
            pool = [n for n in nodes if req.issubset(set(n.get("tags", [])))]
        if not pool:
            return {"results": [], "total_nodes": len(nodes), "mode": "empty"}

        # Search against summary field only
        corpus = [get_summary(n) for n in pool]
        sims, mode = get_similarities(search_text, corpus)
        scored  = sorted(zip(sims, pool), key=lambda x: x[0], reverse=True)
        results = [
            node_as_summary(n) | {"similarity": round(s, 3)}
            for s, n in scored[:top_k]
            if s > 0.01
        ]
        return {
            "results":     results,
            "total_nodes": len(nodes),
            "pool_size":   len(pool),
            "query":       search_text,
            "mode":        mode,
            "note":        "Full content NOT included. Call expand(memory_id=<id>) to load it.",
            "hint":        "Read edge previews to decide which nodes warrant expansion.",
        }

    # ═══════════════════════════════════════════════════════
    # RELATE -- bidirectional labeled edge; preview uses summary
    # ═══════════════════════════════════════════════════════
    elif action == "relate":
        if not source_id or not target_id:
            return {"success": False, "error": "source_id and target_id required"}
        src = get_node(source_id)
        tgt = get_node(target_id)
        if not src:
            return {"success": False, "error": f"source_id '{source_id}' not found"}
        if not tgt:
            return {"success": False, "error": f"target_id '{target_id}' not found"}

        lbl     = label.strip() or "related"
        src_ids = {r["target_id"] for r in src.get("relations", [])}
        tgt_ids = {r["target_id"] for r in tgt.get("relations", [])}

        # Edge previews use summary (already the right length, no truncation needed)
        if target_id not in src_ids:
            src.setdefault("relations", []).append(
                {"target_id": target_id, "label": lbl, "preview": get_summary(tgt)[:80]}
            )
        if source_id not in tgt_ids:
            tgt.setdefault("relations", []).append(
                {"target_id": source_id, "label": lbl, "preview": get_summary(src)[:80]}
            )

        persist()
        return {
            "success":   True,
            "edge":      f"{source_id} --[{lbl}]--> {target_id}",
            "src_edges": len(src["relations"]),
            "tgt_edges": len(tgt["relations"]),
        }

    # ═══════════════════════════════════════════════════════
    # EXPAND -- depth-aware: inner nodes get full content, leaf layer gets summary
    #
    # hops=1: root (full) + direct neighbors (summary + edge map)
    # hops=2: root + 1st hop (full) + 2nd hop (summary + edge map)
    # hops=3: root + 1st + 2nd hop (full) + 3rd hop (summary + edge map)
    # ═══════════════════════════════════════════════════════
    elif action == "expand":
        if not memory_id:
            return {"success": False, "error": "memory_id required"}
        root = get_node(memory_id)
        if not root:
            return {"success": False, "error": f"'{memory_id}' not found"}

        h = max(1, min(3, int(hops)))

        # BFS with depth tracking
        visited = {memory_id: 0}           # id -> depth
        queue   = [(memory_id, 0)]

        while queue:
            nid, depth = queue.pop(0)
            if depth >= h:
                continue
            n = get_node(nid)
            if not n:
                continue
            for r in n.get("relations", []):
                tid = r["target_id"]
                if tid not in visited:
                    visited[tid] = depth + 1
                    queue.append((tid, depth + 1))

        # Build output: inner nodes (depth < h) get full content
        #               leaf nodes (depth == h) get summary only
        neighborhood = []
        for nid, depth in visited.items():
            n = get_node(nid)
            if not n:
                continue
            if depth < h:
                neighborhood.append(node_as_full(n) | {"depth": depth, "loaded": "full"})
            else:
                neighborhood.append(node_as_summary(n) | {"depth": depth, "loaded": "summary"})

        # Sort by depth so root is first
        neighborhood.sort(key=lambda x: x["depth"])

        full_count    = sum(1 for x in neighborhood if x["loaded"] == "full")
        summary_count = sum(1 for x in neighborhood if x["loaded"] == "summary")

        return {
            "root_id":       memory_id,
            "hops":          h,
            "nodes_loaded":  len(neighborhood),
            "full_content":  full_count,
            "summary_only":  summary_count,
            "neighborhood":  neighborhood,
            "hint": (
                f"{summary_count} leaf node(s) returned as summary only. "
                "Call expand(memory_id=<leaf_id>) to load any leaf's full content."
            ),
        }

    # ═══════════════════════════════════════════════════════
    # LIST -- returns summaries only, sorted by importance
    # ═══════════════════════════════════════════════════════
    elif action == "list":
        pool = nodes
        if filter_tags:
            req  = set(filter_tags)
            pool = [n for n in nodes if req.issubset(set(n.get("tags", [])))]
        limit       = max(1, int(top_k))
        sorted_pool = sorted(
            pool,
            key=lambda n: (n.get("importance", 0), n.get("timestamp", "")),
            reverse=True,
        )
        return {
            "nodes":       [node_as_summary(n) for n in sorted_pool[:limit]],
            "total":       len(nodes),
            "filtered_to": len(pool),
            "showing":     min(limit, len(pool)),
            "filter_tags": filter_tags or [],
            "note":        "Summaries only. Use expand(memory_id=<id>) to load full content.",
        }

    # ═══════════════════════════════════════════════════════
    # STATS
    # ═══════════════════════════════════════════════════════
    elif action == "stats":
        st_cached, tfidf_ok = False, False
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=str(MODEL_CACHE),
                local_files_only=True,
            )
            st_cached = True
        except Exception:
            pass
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf_ok = True
        except Exception:
            pass

        total_edges  = sum(len(n.get("relations", [])) for n in nodes) // 2
        rules        = [n for n in nodes if "scoring-rule" in n.get("tags", [])]
        isolated     = [n for n in nodes if not n.get("relations")]
        legacy_nodes = [n for n in nodes if not n.get("summary")]

        return {
            "total_nodes":    len(nodes),
            "total_edges":    total_edges,
            "scoring_rules":  len(rules),
            "isolated_nodes": len(isolated),
            "legacy_nodes":   len(legacy_nodes),
            "active_mode":    (
                "sentence-transformers" if st_cached
                else "tfidf"           if tfidf_ok
                else "hash-fallback"
            ),
            "storage_path":   str(MEMORY_DIR),
        }

    # ═══════════════════════════════════════════════════════
    # DELETE -- removes node and all edges pointing to it
    # ═══════════════════════════════════════════════════════
    elif action == "delete":
        if not memory_id:
            return {"success": False, "error": "memory_id required"}
        target = get_node(memory_id)
        if not target:
            return {"success": False, "error": f"'{memory_id}' not found"}
        for n in nodes:
            n["relations"] = [
                r for r in n.get("relations", []) if r["target_id"] != memory_id
            ]
        nodes[:] = [n for n in nodes if n["id"] != memory_id]
        persist()
        return {
            "success":   True,
            "deleted":   get_summary(target)[:80],
            "remaining": len(nodes),
        }

    else:
        return {
            "success": False,
            "error":   f"Unknown action '{action}'",
            "valid":   ["save", "query", "relate", "expand", "list", "stats", "delete"],
        }
