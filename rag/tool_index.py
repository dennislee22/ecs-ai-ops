from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Tools that must ALWAYS be included regardless of semantic score.
# These are your catch-all / primary-dispatch tools.
_ALWAYS_INCLUDE = {
    "kubectl_exec",    # catch-all kubectl fallback
    "find_resource",   # primary named-resource dispatcher
    "rag_search",      # knowledge base
}

# LanceDB table name — isolated from your doc/excel tables
_TABLE_NAME = "tool_index"

# Distance threshold for confidence check.
# sentence-transformers cosine distance: 0.0 = identical, 2.0 = opposite.
# Values below this are considered confident matches.
# Calibrate by printing distances for a few test queries — good matches
# are typically 0.2–0.5, poor matches 0.7+.
_CONFIDENCE_THRESHOLD = 0.65

# Minimum number of confident results before we trust semantic search.
# If fewer than this pass the threshold, fall back to all tools.
_MIN_CONFIDENT = 3


# ── Embedding helper ─────────────────────────────────────────────────────────

from sentence_transformers import SentenceTransformer
import os
import config.config as _cfg  # assume this always exists

def _get_embedder():
    # Read the model path from your config
    model_name = getattr(_cfg, "EMBED_MODEL")

    # Expand ~ if user provided a local path
    model_name = os.path.expanduser(model_name)

    # Load the embedder (works for local dirs or HF models)
    return SentenceTransformer(model_name)


_embedder = None  # module-level singleton — loaded once on first use

def _embed(text: str) -> list[float]:
    global _embedder
    if _embedder is None:
        _embedder = _get_embedder()
    vec = _embedder.encode(text, normalize_embeddings=True)
    return vec.tolist()


# ── Text to embed for each tool ───────────────────────────────────────────────

def _tool_text(name: str, cfg: dict) -> str:
    """
    Build the text that gets embedded for a tool.
    We embed name + first 300 chars of description only.
    Parameters are excluded — they add noise without helping semantic match.
    """
    desc = cfg.get("description", "")
    # First 300 chars captures the intent; verbose guardrails are not needed
    # for retrieval — they're needed at execution time (full schema).
    short_desc = desc[:300].rsplit(" ", 1)[0]  # trim at word boundary
    return f"{name}: {short_desc}"


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_tools(all_tools: dict[str, dict]) -> None:
    """
    Drop and recreate the tool_index table in LanceDB with fresh embeddings.
    Called once at startup (always-fresh strategy).

    Parameters
    ----------
    all_tools : dict
        Combined tool registry — {**K8S_TOOL_METADATA, **RAG_TOOLS}
        Same dict you pass to build_agent().
    """
    try:
        import lancedb
        import config.config as _cfg

        db = lancedb.connect(_cfg.LANCEDB_DIR)

        # Always-fresh: drop existing table if present
        existing = db.table_names()
        if _TABLE_NAME in existing:
            db.drop_table(_TABLE_NAME)
            logger.info(f"[tool_index] Dropped existing '{_TABLE_NAME}' table")

        rows = []
        for name, cfg in all_tools.items():
            text = _tool_text(name, cfg)
            vector = _embed(text)
            rows.append({
                "tool_name": name,
                "text":      text,
                "vector":    vector,
            })

        db.create_table(_TABLE_NAME, data=rows)
        logger.info(f"[tool_index] Ingested {len(rows)} tools into '{_TABLE_NAME}'")

    except Exception as exc:
        logger.error(f"[tool_index] ingest_tools failed: {exc}", exc_info=True)
        # Non-fatal — app will fall back to all tools if table is missing


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve_tools(
    user_query: str,
    all_schemas: list[dict],
    top_k: int = 8,
) -> list[dict]:
    """
    Return the most relevant tool schemas for user_query.

    Strategy:
    1. Embed user_query, ANN search tool_index table.
    2. Filter results by confidence threshold.
    3. If fewer than _MIN_CONFIDENT pass, fall back to all schemas.
    4. Always inject _ALWAYS_INCLUDE tools on top of semantic results.

    Parameters
    ----------
    user_query   : raw user message string
    all_schemas  : full list of OpenAI-format tool schemas (built at startup)
    top_k        : max tools to return from semantic search (before fallbacks)

    Returns
    -------
    List of OpenAI-format tool schema dicts to pass to apply_chat_template.
    """
    # Build a quick lookup: tool_name → full schema
    schema_map = {s["function"]["name"]: s for s in all_schemas}

    # ── Always-include tools (always present regardless of query) ─────────────
    always_schemas = [schema_map[n] for n in _ALWAYS_INCLUDE if n in schema_map]
    always_names   = {s["function"]["name"] for s in always_schemas}

    try:
        import lancedb
        import config.config as _cfg

        db    = lancedb.connect(_cfg.LANCEDB_DIR)

        # If the table wasn't created (ingest failed at startup), fall back
        if _TABLE_NAME not in db.table_names():
            logger.warning("[tool_index] tool_index table not found — falling back to all tools")
            return all_schemas

        table = db.open_table(_TABLE_NAME)

        query_vec = _embed(user_query)

        # Retrieve top_k + buffer so we have room after filtering
        results = (
            table.search(query_vec)
                 .limit(top_k + len(_ALWAYS_INCLUDE))
                 .to_pandas()
        )

        # ── Confidence check ──────────────────────────────────────────────────
        # _distance is cosine distance (0 = identical, 2 = opposite)
        confident = results[results["_distance"] < _CONFIDENCE_THRESHOLD]

        if len(confident) < _MIN_CONFIDENT:
            logger.info(
                f"[tool_index] Low confidence ({len(confident)} results under threshold "
                f"{_CONFIDENCE_THRESHOLD}) for query={user_query[:60]!r} — "
                f"falling back to all {len(all_schemas)} tools"
            )
            return all_schemas

        # ── Build final schema list ───────────────────────────────────────────
        selected_names  = set(confident["tool_name"].tolist())
        selected_names |= always_names   # union with always-include set

        selected_schemas = [schema_map[n] for n in selected_names if n in schema_map]

        logger.info(
            f"[tool_index] query={user_query[:60]!r} → "
            f"{len(selected_schemas)} tools selected "
            f"(semantic={len(confident)}, always={len(always_names)}): "
            f"{sorted(selected_names)}"
        )

        return selected_schemas

    except Exception as exc:
        logger.error(f"[tool_index] retrieve_tools failed: {exc} — falling back to all tools", exc_info=True)
        return all_schemas
