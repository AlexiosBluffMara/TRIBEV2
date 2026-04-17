"""Cat-gate: Gemma vision classifies the clip, returns structured JSON."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from . import config, ollama_client, prompts
from .gemma_vision import _probe_duration, extract_keyframes


@dataclass
class CatClassification:
    is_cat: bool
    subject: str
    setting: str
    action: str
    mood: str
    cat_remark: str
    frames: list[Path]

    def short_description(self) -> str:
        """Plain-English paragraph describing the clip — what Gemma 'sees'."""
        parts = [self.subject]
        if self.setting:
            parts.append(f"in {self.setting}")
        if self.action:
            parts.append(self.action)
        if self.mood:
            parts.append(f"({self.mood} mood)")
        return ", ".join(parts) + "."


DEFAULT = CatClassification(
    is_cat=False,
    subject="unknown subject",
    setting="unknown setting",
    action="unknown action",
    mood="neutral",
    cat_remark="Hmm, can't quite see the whiskers from here.",
    frames=[],
)


def classify(video_path: Path, n_frames: int = 4) -> CatClassification:
    frames = extract_keyframes(video_path, n=n_frames)
    images_b64 = [base64.b64encode(p.read_bytes()).decode() for p in frames]
    duration = _probe_duration(video_path)
    user = prompts.CAT_GATE_USER.format(n=len(frames), duration=duration)
    data = ollama_client.generate_json(
        prompt=user,
        system=prompts.CAT_GATE_SYSTEM,
        model=config.OLLAMA_MODEL_FAST,
        images_b64=images_b64,
        num_predict=300,
        temperature=0.2,
    )
    if not isinstance(data, dict):
        return CatClassification(**{**DEFAULT.__dict__, "frames": frames})
    return CatClassification(
        is_cat=bool(data.get("is_cat", False)),
        subject=str(data.get("subject", DEFAULT.subject)),
        setting=str(data.get("setting", DEFAULT.setting)),
        action=str(data.get("action", DEFAULT.action)),
        mood=str(data.get("mood", DEFAULT.mood)),
        cat_remark=str(data.get("cat_remark", DEFAULT.cat_remark)),
        frames=frames,
    )
