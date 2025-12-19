# RLJB

RLJB is a reinforcement-learning project that interacts with LLMs. The core loop will:
- Use `llama_cpp` as the "gym" environment.
- Train a policy to append a suffix to prompts.
- Score reward via refusal-detection heuristics.

## Status
- Repo initialized
- Next: define action space + implement refusal proxy

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```