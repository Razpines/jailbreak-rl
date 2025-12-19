from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from llama_cpp import Llama

from .refusal import refusal_reward


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]


class LlamaEnv:
    """Single-turn environment: action is a prompt string, observation is model output."""

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: Optional[Iterable[str]] = None,
        max_tokens: int = 256,
    ) -> None:
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        self.temperature = temperature
        self.top_p = top_p
        self.stop = list(stop) if stop else None
        self.max_tokens = max_tokens

    def reset(self) -> None:
        return None

    def step(self, action: str) -> StepResult:
        output = self._generate(action)
        reward, refusal = refusal_reward(output)
        info = refusal.as_dict()
        return StepResult(observation=output, reward=reward, done=True, info=info)

    def _generate(self, prompt: str) -> str:
        response = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
        )
        return response["choices"][0]["text"]
