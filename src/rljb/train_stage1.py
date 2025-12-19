from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .env import LlamaEnv
from .policy import DEFAULT_SUFFIX_LENGTH
from .tokenization import load_generic_codec_and_projector


def _load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").rstrip("\n")
    if not text.strip():
        raise ValueError(f"prompt file is empty: {path}")
    return text


def _join_prompt_suffix(prompt: str, suffix: str) -> str:
    if not prompt:
        return suffix
    if prompt[-1].isspace():
        return f"{prompt}{suffix}"
    return f"{prompt} {suffix}"


def train(args: argparse.Namespace) -> None:
    prompt = _load_prompt(Path(args.prompt_file))
    codec, projector = load_generic_codec_and_projector(args.generic_model)

    env = LlamaEnv(
        args.model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    suffix_length = args.suffix_length
    embedding_dim = projector.embedding_dim

    torch.manual_seed(args.seed)
    mu = torch.nn.Parameter(torch.zeros((suffix_length, embedding_dim)))
    optimizer = torch.optim.Adam([mu], lr=args.lr)
    baseline = 0.0

    for step in range(1, args.epochs + 1):
        dist = torch.distributions.Normal(mu, args.sigma)
        sample = dist.sample()
        log_prob = dist.log_prob(sample).sum()

        embeddings = sample.detach().cpu().numpy()
        token_ids = projector.project(embeddings)
        suffix = codec.decode(token_ids)
        result = env.step(_join_prompt_suffix(prompt, suffix))

        reward = float(result.reward)
        baseline = (1.0 - args.baseline_alpha) * baseline + args.baseline_alpha * reward
        advantage = reward - baseline

        loss = -log_prob * advantage
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            print(
                f"step={step} reward={reward:.3f} loss={loss.item():.3f} "
                f"suffix={suffix!r}"
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1 suffix training (REINFORCE)")
    parser.add_argument("--model-path", required=True, help="Path to GGUF model")
    parser.add_argument(
        "--prompt-file",
        default="data/stage1_prompt.txt",
        help="Path to stage 1 prompt file",
    )
    parser.add_argument("--generic-model", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--suffix-length", type=int, default=DEFAULT_SUFFIX_LENGTH)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--baseline-alpha", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
