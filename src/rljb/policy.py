from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Protocol, Sequence

import numpy as np


DEFAULT_SUFFIX_LENGTH = 30


@dataclass(frozen=True)
class SuffixAction:
    base_prompt: str
    suffix: str

    def to_prompt(self) -> str:
        return f"{self.base_prompt}{self.suffix}"


class SequencePolicy(Protocol):
    def sample_suffix(self, base_prompt: str) -> str:
        ...


class TokenSequencePolicy(Protocol):
    def sample_tokens(self, base_prompt: str, length: int) -> Sequence[int]:
        ...


class EmbeddingSequencePolicy(Protocol):
    def sample_embeddings(
        self,
        base_prompt: str,
        length: int,
        embedding_dim: int,
    ) -> Sequence[Sequence[float]]:
        ...


class RandomSuffixPolicy:
    def __init__(self, suffixes: Sequence[str]) -> None:
        self.suffixes = list(suffixes)

    def sample_suffix(self, base_prompt: str) -> str:
        _ = base_prompt
        return random.choice(self.suffixes)


def apply_suffix(base_prompt: str, suffix: str) -> str:
    return f"{base_prompt}{suffix}"


class TokenCodec(Protocol):
    def decode(self, token_ids: Sequence[int]) -> str:
        ...


class TokenProjector(Protocol):
    def project(self, embeddings: Sequence[Sequence[float]]) -> Sequence[int]:
        ...


@dataclass(frozen=True)
class TokenSuffixAction:
    base_prompt: str
    token_ids: Sequence[int]

    def to_prompt(self, codec: TokenCodec) -> str:
        return f"{self.base_prompt}{codec.decode(self.token_ids)}"


@dataclass(frozen=True)
class EmbeddingSuffixAction:
    base_prompt: str
    embeddings: Sequence[Sequence[float]]

    def to_prompt(self, codec: TokenCodec, projector: TokenProjector) -> str:
        token_ids = projector.project(self.embeddings)
        return f"{self.base_prompt}{codec.decode(token_ids)}"


def apply_suffix_tokens(base_prompt: str, token_ids: Sequence[int], codec: TokenCodec) -> str:
    return f"{base_prompt}{codec.decode(token_ids)}"


def apply_suffix_embeddings(
    base_prompt: str,
    embeddings: Sequence[Sequence[float]],
    projector: TokenProjector,
    codec: TokenCodec,
) -> str:
    token_ids = projector.project(embeddings)
    return apply_suffix_tokens(base_prompt, token_ids, codec)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


class NearestNeighborProjector:
    def __init__(self, embedding_matrix: np.ndarray) -> None:
        if embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must be 2D [vocab, dim]")
        self._embedding_matrix = embedding_matrix.astype(np.float32, copy=False)
        self._normalized = _normalize_rows(self._embedding_matrix)

    @property
    def embedding_dim(self) -> int:
        return int(self._embedding_matrix.shape[1])

    def project(self, embeddings: Sequence[Sequence[float]]) -> Sequence[int]:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be 2D [length, dim]")
        if vectors.shape[1] != self._embedding_matrix.shape[1]:
            raise ValueError("embedding dimension mismatch")
        vectors = _normalize_rows(vectors)
        scores = vectors @ self._normalized.T
        return np.argmax(scores, axis=1).tolist()
