"""Seven-tier Gemma narration of a TRIBE v2 inference result.

Expertise levels 0-6 from toddler to neuroscience researcher.
Each tier uses the full BrainAnalysis context block so Gemma has every metric.

Backwards-compatible: narrate_tiered() still returns TieredNarration(layperson/clinician/researcher).
New: narrate_tier(result, brain_analysis, tier) returns a single tier as a string.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, ollama_client, prompts
from .pipeline import InferenceResult


@dataclass
class TieredNarration:
    layperson:  str
    clinician:  str
    researcher: str

    def as_dict(self) -> dict[str, str]:
        return {
            "layperson":  self.layperson,
            "clinician":  self.clinician,
            "researcher": self.researcher,
        }


def _roi_lines(result: InferenceResult, n: int = 8) -> str:
    means = result.roi_df[result.top_rois].abs().mean()
    return "\n".join(
        f"  - {roi}: mean |z| = {float(means[roi]):.3f}"
        for roi in result.top_rois[:n]
    )


def _build_user_prompt(label: str, brain_context: str) -> str:
    return prompts.TIER_USER_TEMPLATE.format(
        label=label,
        brain_context=brain_context,
    )


def _legacy_user_prompt(result: InferenceResult, label: str) -> str:
    """Fallback user prompt when BrainAnalysis is not available (old call sites)."""
    return (
        f"Stimulus: {label}\n\n"
        f"Full brain-response data:\n"
        f"Clip duration: {result.preds.shape[0] / 2.0:.1f}s. "
        f"Peak predicted activity at t={result.peak_t / 2.0:.1f}s.\n"
        f"Top Schaefer-400 regions by mean |z|:\n{_roi_lines(result)}\n\n"
        "Write the narration for your assigned audience. "
        "Stick strictly to your tier's register and rules."
    )


def narrate_tier(
    result: InferenceResult,
    label: str,
    tier: int,
    brain_context: str | None = None,
) -> str:
    """Generate narration for a single expertise tier (0-6).

    Args:
        result:        InferenceResult from pipeline
        label:         Human-readable stimulus description
        tier:          0 (toddler) … 6 (researcher)
        brain_context: Output of BrainAnalysis.gemma_context() — if None, uses legacy format
    """
    tier = max(0, min(6, tier))
    system_prompt = prompts.ALL_TIER_SYSTEMS[tier]
    user_prompt = (
        _build_user_prompt(label, brain_context)
        if brain_context else
        _legacy_user_prompt(result, label)
    )
    num_predict = [120, 180, 260, 300, 340, 420, 520][tier]
    temperature = [0.5, 0.5, 0.4, 0.4, 0.4, 0.35, 0.3][tier]

    return ollama_client.generate(
        prompt=user_prompt,
        system=system_prompt,
        model=config.OLLAMA_MODEL_QUALITY,
        temperature=temperature,
        num_predict=num_predict,
    )


def narrate_tiered(
    result: InferenceResult,
    label: str,
    brain_context: str | None = None,
) -> TieredNarration:
    """Three-tier narration (tiers 2, 5, 6) for backwards compatibility."""
    user = (
        _build_user_prompt(label, brain_context)
        if brain_context else
        _legacy_user_prompt(result, label)
    )

    def _call(system_prompt: str, num_predict: int, temperature: float) -> str:
        return ollama_client.generate(
            prompt=user,
            system=system_prompt,
            model=config.OLLAMA_MODEL_QUALITY,
            temperature=temperature,
            num_predict=num_predict,
        )

    return TieredNarration(
        layperson  = _call(prompts.TIER_LAYPERSON_SYSTEM,  260, 0.4),
        clinician  = _call(prompts.TIER_CLINICIAN_SYSTEM,  420, 0.35),
        researcher = _call(prompts.TIER_RESEARCHER_SYSTEM, 520, 0.3),
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


def narrate_all_tiers(
    result: InferenceResult,
    label: str,
    brain_context: str | None = None,
) -> dict[int, str]:
    """Generate all 7 tiers. Slow — use only in batch / research mode."""
    return {
        tier: narrate_tier(result, label, tier, brain_context)
        for tier in range(7)
    }
