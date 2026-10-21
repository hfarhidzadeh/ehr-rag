"""Pluggable text embedders.

* ``tfidf``: TF-IDF (word 1-2 grams, sublinear tf) -> TruncatedSVD -> L2 normalisation. Fitted on the
  corpus, no downloads, deterministic for a given seed. Never imports torch.
* ``minilm``: sentence-transformers ``all-MiniLM-L6-v2`` (384-d). torch / sentence-transformers are imported
  lazily inside :class:`MiniLMEmbedder` so the tfidf path stays torch-free.

All embedders return float32 arrays with L2-normalised rows (a text sharing no vocabulary with a fitted
tfidf model maps to the zero vector, which cannot be normalised and is returned as-is).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ehr_rag.config import Settings

logger = logging.getLogger(__name__)

TFIDF_FILENAME = "tfidf_svd.joblib"
MINILM_CONFIG_FILENAME = "minilm.json"
DEFAULT_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keeps decimals ("7.2"), codes ("4548-4", "e11.9"), ratios ("132/84") and short tokens ("bp", "k").
_TOKEN_PATTERN = r"(?u)\b\w+(?:[./-]\w+)*\b"


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, texts: list[str]) -> None: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...

    def save(self, directory: Path) -> None: ...


def _l2_normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


# --------------------------------------------------------------------------- TF-IDF + SVD


class TfidfSvdEmbedder:
    """Corpus-fitted LSA embedder: TF-IDF (1-2 grams, sublinear tf, min_df=1) -> TruncatedSVD.

    ``n_components = min(dim, n_features - 1, n_samples - 1)``, so ``self.dim`` may shrink after
    :meth:`fit` on a small corpus.
    """

    name = "tfidf"

    def __init__(self, dim: int = 256, seed: int = 42):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.requested_dim = int(dim)
        self.dim = int(dim)
        self.seed = int(seed)
        self._vectorizer: Any = None
        self._svd: Any = None

    @property
    def is_fitted(self) -> bool:
        return self._svd is not None

    def fit(self, texts: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = list(texts)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=True, token_pattern=_TOKEN_PATTERN
        )
        x = vectorizer.fit_transform(texts)
        n_samples, n_features = x.shape
        n_components = min(self.requested_dim, n_features - 1, n_samples - 1)
        if n_components < 1:
            raise ValueError(
                f"TfidfSvdEmbedder needs at least 2 distinct texts with >= 2 terms to fit "
                f"(got {n_samples} texts, {n_features} features)"
            )
        svd = TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=self.seed)
        svd.fit(x)
        self._vectorizer, self._svd, self.dim = vectorizer, svd, n_components
        logger.info(
            "Fitted tfidf embedder: %d texts, %d features, dim=%d (requested %d), explained variance %.3f",
            n_samples, n_features, n_components, self.requested_dim, float(svd.explained_variance_ratio_.sum()),
        )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "TfidfSvdEmbedder is not fitted: call fit(texts) on the corpus first, or load a fitted "
                "model with TfidfSvdEmbedder.load(directory) / load_embedder(settings)."
            )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self._require_fitted()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        z = self._svd.transform(self._vectorizer.transform(texts))
        return _l2_normalise(z)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def save(self, directory: Path) -> None:
        self._require_fitted()
        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "name": self.name,
            "requested_dim": self.requested_dim,
            "dim": self.dim,
            "seed": self.seed,
            "vectorizer": self._vectorizer,
            "svd": self._svd,
        }
        joblib.dump(state, directory / TFIDF_FILENAME)
        logger.info("Saved tfidf embedder to %s", directory / TFIDF_FILENAME)

    @classmethod
    def load(cls, directory: Path) -> "TfidfSvdEmbedder":
        import joblib

        path = Path(directory) / TFIDF_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No fitted tfidf embedder at {path}. Build the index first: "
                "`python -m ehr_rag ingest --embedder tfidf`."
            )
        state = joblib.load(path)
        obj = cls(dim=state["requested_dim"], seed=state["seed"])
        obj._vectorizer, obj._svd, obj.dim = state["vectorizer"], state["svd"], int(state["dim"])
        return obj


# --------------------------------------------------------------------------- MiniLM


class MiniLMEmbedder:
    """sentence-transformers encoder (default ``all-MiniLM-L6-v2``); torch is imported on first use."""

    name = "minilm"

    def __init__(self, model_name: str = DEFAULT_MINILM_MODEL, batch_size: int = 32, device: str = "cpu"):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: pulls in torch

            try:
                self._model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
            except Exception as exc:  # not in the local HF cache yet
                logger.info("Model %s not cached locally (%s); downloading", self.model_name, exc)
                self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:  # type: ignore[override]
        return int(self._load_model().get_sentence_embedding_dimension())

    def fit(self, texts: list[str]) -> None:
        """No-op: the model is pretrained."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = list(texts)
        model = self._load_model()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def save(self, directory: Path) -> None:
        """Write a small JSON config (the weights live in the HF cache)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config = {"name": self.name, "model_name": self.model_name, "batch_size": self.batch_size,
                  "device": self.device}
        (directory / MINILM_CONFIG_FILENAME).write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "MiniLMEmbedder":
        path = Path(directory) / MINILM_CONFIG_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No MiniLM config at {path}. Build the index first: `python -m ehr_rag ingest --embedder minilm`."
            )
        config = json.loads(path.read_text())
        return cls(
            model_name=config.get("model_name", DEFAULT_MINILM_MODEL),
            batch_size=config.get("batch_size", 32),
            device=config.get("device", "cpu"),
        )


# --------------------------------------------------------------------------- factories


def get_embedder(settings: Settings) -> Embedder:
    """Fresh, unfitted embedder named by ``settings.embedder``."""
    if settings.embedder == "minilm":
        return MiniLMEmbedder(model_name=settings.minilm_model)
    if settings.embedder == "tfidf":
        return TfidfSvdEmbedder(dim=settings.tfidf_dim, seed=settings.seed)
    raise ValueError(f"Unknown embedder {settings.embedder!r}; expected 'minilm' or 'tfidf'")


def load_embedder(settings: Settings) -> Embedder:
    """Ready-to-use embedder. tfidf is loaded from ``settings.embedder_index_dir``.

    For minilm, the saved config (if any) wins over ``settings.minilm_model`` so query vectors match the
    index; without a saved config a fresh pretrained model is returned.
    """
    directory = settings.embedder_index_dir
    if settings.embedder == "tfidf":
        emb = TfidfSvdEmbedder.load(directory)
        if emb.requested_dim != settings.tfidf_dim:
            logger.warning("Loaded tfidf embedder with dim=%d (settings.tfidf_dim=%d)", emb.requested_dim,
                           settings.tfidf_dim)
        return emb
    if settings.embedder == "minilm":
        if (directory / MINILM_CONFIG_FILENAME).exists():
            emb_m = MiniLMEmbedder.load(directory)
            if emb_m.model_name != settings.minilm_model:
                logger.warning("Index was built with %s; using it instead of settings.minilm_model=%s",
                               emb_m.model_name, settings.minilm_model)
            return emb_m
        return MiniLMEmbedder(model_name=settings.minilm_model)
    raise ValueError(f"Unknown embedder {settings.embedder!r}; expected 'minilm' or 'tfidf'")
