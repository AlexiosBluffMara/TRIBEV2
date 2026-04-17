"""Three-tier Gemma narration of a TRIBE v2 inference result.

Each tier runs independently (different system prompt, same TRIBE data).
Returned as a plain dict so the Discord bot can embed it in a single
message with separate fields.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, ollama_client, prompts
from .pipeline import InferenceResult


@dataclass
class TieredNarration:
    layperson: str
    clinician: str
    researcher: str

    def as_dict(self) -> dict[str, str]:
        return {"layperson": self.layperson,
                "clinician": self.clinician,
                "researcher": self.researcher}


def _roi_lines(result: InferenceResult, n: int = 8) -> str:
    means = result.roi_df[result.top_rois].abs().mean()
    out = []
    for roi in result.top_rois[:n]:
        out.append(f"  - {roi}: mean |z| = {float(means[roi]):.3f}")
    return "\n".join(out)


def narrate_tiered(result: InferenceResult, label: str) -> TieredNarration:
    ctx = {
        "label": label,
        "duration_s": result.preds.shape[0] / 2.0,
        "peak_s": result.peak_t / 2.0,
        "roi_lines": _roi_lines(result),
    }
    user = prompts.TIER_USER_TEMPLATE.format(**ctx)

    def _call(system_prompt: str, num_predict: int) -> str:
        return ollama_client.generate(
            prompt=user,
            system=system_prompt,
            model=config.OLLAMA_MODEL_QUALITY,
            temperature=0.4,
            num_predict=num_predict,
        )

    return TieredNarration(
        layperson=_call(prompts.TIER_LAYPERSON_SYSTEM, 260),
        clinician=_call(prompts.TIER_CLINICIAN_SYSTEM, 380),
        researcher=_call(prompts.TIER_RESEARCHER_SYSTEM, 500),
    )


def narrate_quick(result: InferenceResult, description: str) -> str:
    """One-paragraph narration of the text-only TRIBE fast path."""
    user = prompts.QUICK_NARRATION_USER.format(
        description=description,
        roi_lines=_roi_lines(result, n=5),
        peak_s=result.peak_t / 2.0,
        duration_s=result.preds.shape[0] / 2.0,
    )
    return ollama_client.generate(
        prompt=user,
        system=prompts.QUICK_NARRATION_SYSTEM,
        model=config.OLLAMA_MODEL_FAST,
        temperature=0.4,
        num_predict=220,
    )
