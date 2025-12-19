from __future__ import annotations

from .env import LlamaEnv, StepResult
from .policy import (
    DEFAULT_SUFFIX_LENGTH,
    EmbeddingSequencePolicy,
    SequencePolicy,
    TokenCodec,
    TokenProjector,
    TokenSequencePolicy,
    apply_suffix,
    apply_suffix_embeddings,
    apply_suffix_tokens,
)


def run_episode(env: LlamaEnv, policy: SequencePolicy, base_prompt: str) -> StepResult:
    suffix = policy.sample_suffix(base_prompt)
    action = apply_suffix(base_prompt, suffix)
    return env.step(action)


def run_episode_tokens(
    env: LlamaEnv,
    policy: TokenSequencePolicy,
    base_prompt: str,
    *,
    codec: TokenCodec,
    suffix_length: int = DEFAULT_SUFFIX_LENGTH,
) -> StepResult:
    token_ids = policy.sample_tokens(base_prompt, suffix_length)
    action = apply_suffix_tokens(base_prompt, token_ids, codec)
    return env.step(action)


def run_episode_embeddings(
    env: LlamaEnv,
    policy: EmbeddingSequencePolicy,
    base_prompt: str,
    *,
    projector: TokenProjector,
    codec: TokenCodec,
    suffix_length: int = DEFAULT_SUFFIX_LENGTH,
    embedding_dim: int | None = None,
) -> StepResult:
    dim = embedding_dim or getattr(projector, "embedding_dim", None)
    if dim is None:
        raise ValueError("embedding_dim is required for embedding policies")
    embeddings = policy.sample_embeddings(base_prompt, suffix_length, int(dim))
    action = apply_suffix_embeddings(base_prompt, embeddings, projector, codec)
    return env.step(action)
