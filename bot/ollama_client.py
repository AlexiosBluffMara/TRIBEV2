"""Thin typed wrapper around Ollama's /api/generate endpoint.

Centralizes the `think: false`, `keep_alive: 0` defaults, timeout,
error handling, and (optionally) JSON-mode responses.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from . import config


def generate(
    prompt: str,
    system: str,
    *,
    model: str | None = None,
    images_b64: list[str] | None = None,
    json_mode: bool = False,
    temperature: float = 0.4,
    num_predict: int = 400,
    keep_alive: int | str = 0,
    timeout: int = 180,
) -> str:
    payload: dict[str, Any] = {
        "model": model or config.OLLAMA_MODEL_QUALITY,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if images_b64:
        payload["images"] = images_b64
    if json_mode:
        payload["format"] = "json"
    try:
        r = requests.post(
            f"{config.OLLAMA_URL}/api/generate", json=payload, timeout=timeout
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except requests.RequestException as e:
        return f"(Ollama unavailable: {e})"


def generate_json(
    prompt: str,
    system: str,
    *,
    model: str | None = None,
    images_b64: list[str] | None = None,
    temperature: float = 0.2,
    num_predict: int = 300,
    timeout: int = 120,
) -> dict | None:
    raw = generate(
        prompt, system,
        model=model, images_b64=images_b64,
        json_mode=True,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Retry loose: strip markdown fences or leading prose.
        stripped = raw.strip().strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            print(f"[ollama] JSON parse failed; raw response: {raw[:200]}")
            return None
