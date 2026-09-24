"""API key loading.

Looked for in this order, so the common case needs no setup and an override
needs no file edit:

1. an exported ``VLMRUN_API_KEY`` in the environment
2. the nearest `.env` at or above the project directory

The upward walk is what lets this project sit inside a larger tree that keeps
one shared `.env` at its root, while a standalone checkout just puts `.env`
next to `main.py`. See `.env.example`.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_env_file(start: Path) -> Path | None:
    """Nearest `.env` at or above *start*, or None."""
    for directory in [start, *start.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_api_key(start: Path) -> tuple[str, str]:
    """Return ``(api_key, source)``.

    An already-exported ``VLMRUN_API_KEY`` wins over the `.env` file, so a
    one-off override does not require editing anything.
    """
    preset = os.getenv("VLMRUN_API_KEY")
    if preset:
        return preset, "environment"

    env_file = find_env_file(start)
    if env_file is None:
        raise RuntimeError(
            "No VLMRUN_API_KEY in the environment, and no .env found in "
            f"{start} or any parent directory. Run `cp .env.example .env` and "
            "add your key. Get one at https://app.vlm.run."
        )

    from dotenv import dotenv_values

    key = (dotenv_values(env_file) or {}).get("VLMRUN_API_KEY")
    if not key:
        raise RuntimeError(
            f"{env_file} has no VLMRUN_API_KEY entry. Add "
            "VLMRUN_API_KEY=... to it. Get a key at https://app.vlm.run."
        )

    os.environ["VLMRUN_API_KEY"] = key
    return key, str(env_file)
