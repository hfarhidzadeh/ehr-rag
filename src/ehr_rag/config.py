"""Runtime settings. Every field can be overridden with an ``EHR_RAG_<FIELD>`` environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # paths
    data_dir: Path = PROJECT_ROOT / "data" / "synthetic"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    reports_dir: Path = PROJECT_ROOT / "reports"

    # synthetic data
    n_patients: int = 30
    seed: int = 42

    # chunking
    chunk_max_chars: int = 900
    chunk_overlap_sentences: int = 1

    # embeddings
    embedder: str = "minilm"                     # "minilm" | "tfidf"
    minilm_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tfidf_dim: int = 256

    # retrieval
    retrieval_mode: str = "hybrid"               # "dense" | "bm25" | "hybrid" | "random"
    top_k: int = 6
    rrf_k: int = 60
    candidate_pool: int = 30                     # per-retriever depth before fusion

    # generation
    llm: str = "auto"                            # "auto" | "claude" | "extractive"
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"                # low | medium | high | xhigh | max
    claude_max_tokens: int = 16000
    claude_fallbacks: bool = True                # server-side refusal fallback (beta)

    def with_overrides(self, **kwargs) -> "Settings":
        """Copy with the given non-None fields replaced; str values for path fields become ``Path``."""
        values = {k: v for k, v in kwargs.items() if v is not None}
        for k, v in values.items():
            if isinstance(getattr(self, k, None), Path) and isinstance(v, str):
                values[k] = Path(v).expanduser()
        return replace(self, **values)

    @property
    def collection_name(self) -> str:
        return f"ehr_chunks_{self.embedder}"

    @property
    def embedder_index_dir(self) -> Path:
        return self.index_dir / self.embedder


def _coerce(value: str, current):
    if isinstance(current, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, Path):
        return Path(value).expanduser()
    return value


def load_settings(**overrides) -> Settings:
    """Defaults <- EHR_RAG_* environment variables <- explicit keyword overrides."""
    base = Settings()
    env: dict = {}
    for f in fields(Settings):
        raw = os.environ.get(f"EHR_RAG_{f.name.upper()}")
        if raw is not None:
            env[f.name] = _coerce(raw, getattr(base, f.name))
    return base.with_overrides(**env).with_overrides(**overrides)
