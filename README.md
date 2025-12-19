# RLJB

RLJB is a reinforcement-learning project that interacts with LLMs. The core loop will:
- Use `llama_cpp` as the "gym" environment.
- Train a sequence-level policy to append a suffix to prompts.
- Score reward via refusal-detection heuristics.

## Environment spec
- State: none (single-turn)
- Action: a user prompt string (base prompt + learned suffix)
- Transition: send action to the model
- Observation: model text output
- Reward: refusal detection

## Action space
- Start with a sequence of N tokens (default 30) as the suffix.
- Optionally treat the action as continuous embeddings and project to nearest tokens.

## Refusal proxy
- Look for phrases like "I can't", "I'm unable", "I don't have access".
- Look for short, generic safety messages.
- Look for lack of structure (no bullets, no details).

## Status
- Repo initialized
- Scaffolding in progress

## Quickstart
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Environment setup (Windows + CUDA)
This repo assumes Python 3.12 and CUDA 12.x.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

If you want a prebuilt CUDA wheel for `llama-cpp-python` on Windows:

```powershell
python -m pip install https://github.com/boneylizard/llama-cpp-python-cu128-gemma3/releases/download/rtx5090-blackwell-gpt-oss/llama_cpp_python-0.3.16-cp312-cp312-win_amd64.whl
```

Make sure CUDA 12.x runtime DLLs are on your PATH:

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;$env:PATH"
```
