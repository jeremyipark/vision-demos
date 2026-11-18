"""The API key: an exported ``VLMRUN_API_KEY``, else the nearest `.env` at or above the project."""

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
    """``(api_key, where it came from)``."""
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
