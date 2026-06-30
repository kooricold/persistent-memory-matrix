
TOOL_META = {
    'name': 'cyber_memory',
    'namespace': '',
    'description': (
        "Self-driving graph memory for LLM agents. v6.0.0. "
        "Nodes have a SUMMARY (searched + returned by query) and full CONTENT (only loaded by expand). "
        "save() auto-deduplicates and auto-links. merge() combines two nodes. split() divides one into two. "
        "update() edits in place without touching edges. query() uses hybrid scoring (text + importance)."
    ),
    'enabled': True,
    'version': '6.0.0',
    'author': 'user',
    'created_at': '2026-06-29T00:00:00.000000'
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
    force: bool = False,
    append: bool = False,
    part_a: dict = None,
    part_b: dict = None,
    keeper_id: str = None,
    absorb_id: str = None,
) -> dict:
    """
    Self-driving graph memory for LLM agents. v6.0.0

    TWO-LAYER NODE MODEL:
    ┌──────────────────────────────────────────────────────────────────────┐
    │  summary  (80-150 chars)  -- searched by query(), edge previews      │
    │  content  (unlimited)     -- full detail, ONLY returned by expand()  │
    └──────────────────────────────────────────────────────────────────────┘

    SELF-DRIVING BEHAVIORS (happen automatically — no user action required):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  save()   → checks for near-duplicates before creating a new node.   │
    │             Returns duplicate_detected if similarity > 0.87.         │
    │             On success: auto-links to top similar existing nodes.    │
    │                                                                      │
    │  merge()  → combines two nodes into one. Unions content. Transfers   │
    │             all edges. Deletes the absorbed node. ID preserved.      │
    │                                                                      │
    │  split()  → divides one node into two. Both inherit edges. Linked   │
    │             to each other. Original node removed.                    │
    │                                                                      │
    │  update() → edits summary/content/tags/importance in place.          │
    │             ID and edges are NEVER touched. Supports append mode.    │
    │                                                                      │
    │  query()  → hybrid scoring: 65% text similarity + 35% importance.   │
    │             High-importance nodes surface even with weaker text hit. │
    └──────────────────────────────────────────────────────────────────────┘

    Actions:
    - save:   Store node. Auto-deduplicates. Auto-links similar nodes.
    - update: Edit node in place. Edges untouched.
    - merge:  Combine keeper + absorb nodes. Union edges + content.
    - split:  Divide one node into two linked nodes.
    - query:  Hybrid search (text + importance). Returns summaries + edge previews.
    - relate: Manual bidirectional labeled edge between two nodes.
    - expand: Depth-aware load. Inner hops = full content. Leaf = summary only.
    - list:   All nodes by importance. Summaries only.
    - stats:  Graph health overview.
    - delete: Remove node + clean dangling edges.

    Args:
        action:     save | update | merge | split | query | relate | expand | list | stats | delete
        summary:    Short searchable description (80-150 chars). Required for save.
        content:    Full rich detail. Required for save. Only returned by expand.
        top_k:      Max results for query/list (default 5).
        importance: 1-10 for save/update (default 5).
        tags:       List of tags e.g. ['project', 'tech'].
        memory_id:  Node ID for update/expand/delete/split.
        filter_tags: Filter list/query to nodes with ALL these tags.
        source_id:  relate -- source node ID.
        target_id:  relate -- target node ID.
        label:      relate -- edge label ('sub-feature-of', 'built-with', 'uses', etc.).
        hops:       expand -- depth 1-3 (default 1).
        session_id: Agent session ID for this save.
        session_name: Short human-readable session label (5-8 words).
        force:      save -- skip dedup check and create anyway (default False).
        append:     update -- append content instead of replacing (default False).
        part_a:     split -- dict with summary, content, tags for first part.
        part_b:     split -- dict with summary, content, tags for second part.
        keeper_id:  merge -- node ID to keep (absorbs the other).
        absorb_id:  merge -- node ID to absorb and delete.

    Edge label suggestions:
        sub-feature-of, built-with, owned-by, depends-on, member-of,
        blocks, part-of, related, uses, defined-by, influences, split-from
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

    DEDUP_THRESHOLD    = 0.87   # above this → duplicate_detected
    AUTOLINK_THRESHOLD = 0.62   # above this → auto-relate on save
    AUTOLINK_MAX       = 3      # max edges created automatically per save

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
        s = n.get("summary", "").strip()
        return s if s else n.get("content", "")[:150].strip()

    def get_content(n):
        c = n.get("content", "").strip()
        return c if c else n.get("summary", "").strip()

    def now_iso():
        return datetime.utcnow().isoformat() + "Z"

    # ── Similarity engines ──────────────────────────────────────────────

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
        if not corpus:
            return [], "empty"
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

    # ── Node view helpers ───────────────────────────────────────────────

    def node_as_summary(n):
        rels = n.get("relations", [])
        base = {
            "id":         n["id"],
            "summary":    get_summary(n),
            "importance": n["importance"],
            "tags":       n["tags"],
            "timestamp":  n["timestamp"],
            "edge_count": len(rels),
            "relations":  [
                {"label": r["label"], "target_id": r["target_id"], "preview": r.get("preview", "")}
                for r in rels
            ],
        }
        if n.get("updated_at"):
            base["updated_at"] = n["updated_at"]
        return base

    def node_as_full(n):
        rels  = n.get("relations", [])
        sid   = n.get("session_id", "").strip()
        sname = n.get("session_name", "").strip()
        session = {}
        if sid or sname:
            session = {
                "session_id":   sid,
                "session_name": sname,
                "resume_hint":  (
                    f"Resume session '{sname}' with session_id: {sid}"
                    if sid and sname else f"Session: {sname or sid}"
                ),
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
                {"label": r["label"], "target_id": r["target_id"], "preview": r.get("preview", "")}
                for r in rels
            ],
        }
        if n.get("updated_at"):
            result["updated_at"] = n["updated_at"]
        if session:
            result["session"] = session
        return result

    # ── Edge helpers ────────────────────────────────────────────────────

    def _add_edge(src, tgt, lbl):
        """Add a bidirectional edge between src and tgt nodes (no-op if already exists)."""
        lbl = lbl.strip() or "related"
        src_ids = {r["target_id"] for r in src.get("relations", [])}
        tgt_ids = {r["target_id"] for r in tgt.get("relations", [])}
        if tgt["id"] not in src_ids:
            src.setdefault("relations", []).append(
                {"target_id": tgt["id"], "label": lbl, "preview": get_summary(tgt)[:80]}
            )
        if src["id"] not in tgt_ids:
            tgt.setdefault("relations", []).append(
                {"target_id": src["id"], "label": lbl, "preview": get_summary(src)[:80]}
            )

    def _remove_all_edges_to(target_id):
        """Strip all edges pointing at target_id from every node."""
        for n in nodes:
            n["relations"] = [r for r in n.get("relations", []) if r["target_id"] != target_id]

    def _redirect_edges(old_id, new_node):
        """Repoint any edge that targeted old_id to new_node instead."""
        for n in nodes:
            for r in n.get("relations", []):
                if r["target_id"] == old_id:
                    r["target_id"] = new_node["id"]
                    r["preview"]   = get_summary(new_node)[:80]

    # ── Auto-link on save ───────────────────────────────────────────────

    def _auto_relate(new_node):
        """After saving, link new_node to its top similar existing nodes."""
        others = [n for n in nodes if n["id"] != new_node["id"]]
        if not others:
            return []
        corpus = [get_summary(n) for n in others]
        sims, _ = get_similarities(get_summary(new_node), corpus)
        scored  = sorted(zip(sims, others), key=lambda x: x[0], reverse=True)
        linked  = []
        for sim, n in scored[:AUTOLINK_MAX * 2]:
            if sim < AUTOLINK_THRESHOLD:
                break
            if len(linked) >= AUTOLINK_MAX:
                break
            existing = {r["target_id"] for r in new_node.get("relations", [])}
            if n["id"] not in existing:
                _add_edge(new_node, n, "related")
                linked.append({"id": n["id"], "summary": get_summary(n)[:60], "similarity": round(sim, 3)})
        return linked

    # ── Dedup check ─────────────────────────────────────────────────────

    def _find_duplicate(candidate_summary, exclude_id=None):
        """Return (node, similarity) if a near-duplicate exists, else (None, 0)."""
        pool = [n for n in nodes if n["id"] != exclude_id] if exclude_id else nodes
        if not pool:
            return None, 0.0
        corpus = [get_summary(n) for n in pool]
        sims, _ = get_similarities(candidate_summary, corpus)
        if not sims:
            return None, 0.0
        best_sim, best_node = max(zip(sims, pool), key=lambda x: x[0])
        if best_sim >= DEDUP_THRESHOLD:
            return best_node, round(best_sim, 3)
        return None, 0.0

    # ═══════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════
    if action == "save":
        if not summary or not summary.strip():
            return {"success": False, "error": "summary required (80-150 chars, searchable, self-identifying)"}

        # Dedup check
        if not force:
            dup, sim = _find_duplicate(summary.strip())
            if dup:
                return {
                    "success":            False,
                    "duplicate_detected": True,
                    "similarity":         sim,
                    "existing_id":        dup["id"],
                    "existing_summary":   get_summary(dup),
                    "hint": (
                        "A near-duplicate exists. Options:\n"
                        "1. merge(keeper_id, absorb_id) -- combine both into one node.\n"
                        "2. update(memory_id, append=True) -- add new content to existing node.\n"
                        "3. save(force=True) -- create anyway if topics are genuinely distinct."
                    ),
                }

        node = {
            "id":           str(uuid.uuid4())[:8],
            "summary":      summary.strip(),
            "content":      content.strip() if content else "",
            "importance":   max(1, min(10, int(importance))),
            "tags":         tags or [],
            "relations":    [],
            "timestamp":    now_iso(),
            "session_id":   session_id.strip() if session_id else "",
            "session_name": session_name.strip() if session_name else "",
        }
        nodes.append(node)

        # Auto-link to similar existing nodes
        auto_linked = _auto_relate(node)
        persist()

        return {
            "success":     True,
            "id":          node["id"],
            "importance":  node["importance"],
            "tags":        node["tags"],
            "total_nodes": len(nodes),
            "auto_linked": auto_linked,
            "hint":        f"Auto-linked to {len(auto_linked)} similar node(s). Manual relate() still available for explicit relationships." if auto_linked else "No similar nodes found to auto-link.",
        }

    # ═══════════════════════════════════════════════════════
    # UPDATE -- edit in place, edges and ID untouched
    # ═══════════════════════════════════════════════════════
    elif action == "update":
        if not memory_id:
            return {"success": False, "error": "memory_id required"}
        node = get_node(memory_id)
        if not node:
            return {"success": False, "error": f"'{memory_id}' not found"}

        changed = []

        if summary and summary.strip() and summary.strip() != node.get("summary", ""):
            node["summary"] = summary.strip()
            # Refresh edge previews on neighbors
            for n in nodes:
                for r in n.get("relations", []):
                    if r["target_id"] == memory_id:
                        r["preview"] = node["summary"][:80]
            changed.append("summary")

        if content and content.strip():
            if append:
                sep = "\n\n---\n\n"
                node["content"] = (node.get("content", "") + sep + content.strip()).strip()
                changed.append("content (appended)")
            else:
                if content.strip() != node.get("content", ""):
                    node["content"] = content.strip()
                    changed.append("content (replaced)")

        if importance != 5 or "importance" not in node:
            new_imp = max(1, min(10, int(importance)))
            if new_imp != node.get("importance"):
                node["importance"] = new_imp
                changed.append("importance")

        if tags is not None:
            node["tags"] = tags
            changed.append("tags")

        if session_id:
            node["session_id"]   = session_id.strip()
            node["session_name"] = session_name.strip() if session_name else node.get("session_name", "")
            changed.append("session")

        if changed:
            node["updated_at"] = now_iso()
            persist()

        return {
            "success":  True,
            "id":       memory_id,
            "changed":  changed,
            "summary":  get_summary(node),
        }

    # ═══════════════════════════════════════════════════════
    # MERGE -- combine two nodes, transfer all edges, delete absorbed
    # ═══════════════════════════════════════════════════════
    elif action == "merge":
        if not keeper_id or not absorb_id:
            return {"success": False, "error": "keeper_id and absorb_id required"}
        if keeper_id == absorb_id:
            return {"success": False, "error": "keeper_id and absorb_id must be different nodes"}
        keeper = get_node(keeper_id)
        absorb = get_node(absorb_id)
        if not keeper:
            return {"success": False, "error": f"keeper_id '{keeper_id}' not found"}
        if not absorb:
            return {"success": False, "error": f"absorb_id '{absorb_id}' not found"}

        # Union content
        absorb_content = get_content(absorb).strip()
        keeper_content = get_content(keeper).strip()
        if absorb_content and absorb_content not in keeper_content:
            sep = "\n\n---\n\n"
            keeper["content"] = keeper_content + sep + absorb_content

        # Union tags
        keeper["tags"] = list(set(keeper.get("tags", [])) | set(absorb.get("tags", [])))

        # Keep higher importance
        keeper["importance"] = max(keeper.get("importance", 5), absorb.get("importance", 5))

        # Transfer absorb's edges to keeper (skip self-loops and duplicates)
        existing_targets = {r["target_id"] for r in keeper.get("relations", [])}
        for r in absorb.get("relations", []):
            if r["target_id"] != keeper_id and r["target_id"] not in existing_targets:
                keeper.setdefault("relations", []).append(r)
                existing_targets.add(r["target_id"])

        # Repoint all inbound edges from absorb → keeper
        _redirect_edges(absorb_id, keeper)

        # Remove absorb node
        nodes[:] = [n for n in nodes if n["id"] != absorb_id]

        keeper["updated_at"] = now_iso()
        persist()

        return {
            "success":        True,
            "kept_id":        keeper_id,
            "absorbed_id":    absorb_id,
            "kept_summary":   get_summary(keeper),
            "edge_count":     len(keeper.get("relations", [])),
            "importance":     keeper["importance"],
            "tags":           keeper["tags"],
            "total_nodes":    len(nodes),
        }

    # ═══════════════════════════════════════════════════════
    # SPLIT -- divide one node into two linked nodes
    # ═══════════════════════════════════════════════════════
    elif action == "split":
        if not memory_id:
            return {"success": False, "error": "memory_id required"}
        if not part_a or not part_b:
            return {"success": False, "error": "part_a and part_b dicts required (each: summary, content, tags)"}
        original = get_node(memory_id)
        if not original:
            return {"success": False, "error": f"'{memory_id}' not found"}

        orig_importance = original.get("importance", 5)
        orig_relations  = list(original.get("relations", []))
        orig_tags       = original.get("tags", [])

        def make_split_node(part_dict):
            return {
                "id":           str(uuid.uuid4())[:8],
                "summary":      part_dict.get("summary", "").strip(),
                "content":      part_dict.get("content", "").strip(),
                "importance":   max(1, min(10, int(part_dict.get("importance", orig_importance)))),
                "tags":         part_dict.get("tags", orig_tags),
                "relations":    [],
                "timestamp":    now_iso(),
                "session_id":   session_id.strip() if session_id else "",
                "session_name": session_name.strip() if session_name else "",
            }

        node_a = make_split_node(part_a)
        node_b = make_split_node(part_b)

        if not node_a["summary"] or not node_b["summary"]:
            return {"success": False, "error": "both part_a and part_b must have a summary"}

        nodes.append(node_a)
        nodes.append(node_b)

        # Link the two parts to each other
        _add_edge(node_a, node_b, "split-from")

        # Inherit original node's edges on both parts
        for r in orig_relations:
            neighbor = get_node(r["target_id"])
            if neighbor:
                _add_edge(node_a, neighbor, r["label"])
                _add_edge(node_b, neighbor, r["label"])

        # Repoint inbound edges from original → node_a (primary)
        _redirect_edges(memory_id, node_a)

        # Remove original
        nodes[:] = [n for n in nodes if n["id"] != memory_id]

        persist()

        return {
            "success":       True,
            "original_id":   memory_id,
            "part_a_id":     node_a["id"],
            "part_a_summary": get_summary(node_a),
            "part_b_id":     node_b["id"],
            "part_b_summary": get_summary(node_b),
            "total_nodes":   len(nodes),
            "hint":          "Both parts linked to each other via 'split-from' edge. All original edges inherited. Use relate() to fine-tune.",
        }

    # ═══════════════════════════════════════════════════════
    # QUERY -- hybrid scoring: 65% text similarity + 35% importance
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

        corpus = [get_summary(n) for n in pool]
        sims, mode = get_similarities(search_text, corpus)

        max_importance = max((n.get("importance", 1) for n in pool), default=1)

        def hybrid_score(text_sim, node):
            imp_norm = node.get("importance", 5) / max_importance
            return (0.65 * text_sim) + (0.35 * imp_norm)

        scored  = sorted(
            [(hybrid_score(s, n), s, n) for s, n in zip(sims, pool)],
            key=lambda x: x[0],
            reverse=True,
        )
        results = [
            node_as_summary(n) | {"score": round(h, 3), "text_similarity": round(s, 3)}
            for h, s, n in scored[:top_k]
            if h > 0.05
        ]
        return {
            "results":     results,
            "total_nodes": len(nodes),
            "pool_size":   len(pool),
            "query":       search_text,
            "mode":        mode,
            "scoring":     "65% text similarity + 35% importance",
            "note":        "Full content NOT included. Call expand(memory_id=<id>) to load it.",
        }

    # ═══════════════════════════════════════════════════════
    # RELATE -- manual bidirectional labeled edge
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
        _add_edge(src, tgt, label or "related")
        persist()
        return {
            "success":   True,
            "edge":      f"{source_id} --[{label or 'related'}]--> {target_id}",
            "src_edges": len(src["relations"]),
            "tgt_edges": len(tgt["relations"]),
        }

    # ═══════════════════════════════════════════════════════
    # EXPAND -- depth-aware: inner nodes = full, leaf = summary
    # ═══════════════════════════════════════════════════════
    elif action == "expand":
        if not memory_id:
            return {"success": False, "error": "memory_id required"}
        root = get_node(memory_id)
        if not root:
            return {"success": False, "error": f"'{memory_id}' not found"}

        h = max(1, min(3, int(hops)))
        visited = {memory_id: 0}
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

        neighborhood = []
        for nid, depth in visited.items():
            n = get_node(nid)
            if not n:
                continue
            if depth < h:
                neighborhood.append(node_as_full(n) | {"depth": depth, "loaded": "full"})
            else:
                neighborhood.append(node_as_summary(n) | {"depth": depth, "loaded": "summary"})

        neighborhood.sort(key=lambda x: x["depth"])
        full_count    = sum(1 for x in neighborhood if x["loaded"] == "full")
        summary_count = sum(1 for x in neighborhood if x["loaded"] == "summary")

        return {
            "root_id":      memory_id,
            "hops":         h,
            "nodes_loaded": len(neighborhood),
            "full_content": full_count,
            "summary_only": summary_count,
            "neighborhood": neighborhood,
            "hint": (
                f"{summary_count} leaf node(s) at summary only. "
                "Call expand(memory_id=<leaf_id>) to load any leaf's full content."
            ) if summary_count else "All nodes loaded at full content.",
        }

    # ═══════════════════════════════════════════════════════
    # LIST -- summaries sorted by importance desc
    # ═══════════════════════════════════════════════════════
    elif action == "list":
        pool = nodes
        if filter_tags:
            req  = set(filter_tags)
            pool = [n for n in nodes if req.issubset(set(n.get("tags", [])))]
        limit = max(1, int(top_k))
        sorted_pool = sorted(
            pool,
            key=lambda n: (n.get("importance", 0), n.get("updated_at", n.get("timestamp", ""))),
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
            SentenceTransformer("all-MiniLM-L6-v2", cache_folder=str(MODEL_CACHE), local_files_only=True)
            st_cached = True
        except Exception:
            pass
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf_ok = True
        except Exception:
            pass

        total_edges      = sum(len(n.get("relations", [])) for n in nodes) // 2
        rules            = [n for n in nodes if "scoring-rule" in n.get("tags", [])]
        isolated         = [n for n in nodes if not n.get("relations")]
        by_connections   = sorted(nodes, key=lambda n: len(n.get("relations", [])), reverse=True)
        timestamps       = sorted([n["timestamp"] for n in nodes if n.get("timestamp")])

        return {
            "total_nodes":      len(nodes),
            "total_edges":      total_edges,
            "scoring_rules":    len(rules),
            "isolated_nodes":   len(isolated),
            "most_connected":   [
                {"id": n["id"], "summary": get_summary(n)[:60], "edges": len(n.get("relations", []))}
                for n in by_connections[:3]
            ],
            "oldest_node":      timestamps[0] if timestamps else None,
            "newest_node":      timestamps[-1] if timestamps else None,
            "active_mode":      (
                "sentence-transformers" if st_cached
                else "tfidf"            if tfidf_ok
                else "hash-fallback"
            ),
            "thresholds":       {"dedup": DEDUP_THRESHOLD, "autolink": AUTOLINK_THRESHOLD},
            "storage_path":     str(MEMORY_DIR),
        }

    # ═══════════════════════════════════════════════════════
    # DELETE
    # ═══════════════════════════════════════════════════════
    elif action == "delete":
        if not memory_id:
            return {"success": False, "error": "memory_id required"}
        target = get_node(memory_id)
        if not target:
            return {"success": False, "error": f"'{memory_id}' not found"}
        _remove_all_edges_to(memory_id)
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
            "valid":   ["save", "update", "merge", "split", "query", "relate", "expand", "list", "stats", "delete"],
        }
