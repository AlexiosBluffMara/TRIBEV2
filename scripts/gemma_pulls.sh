#!/usr/bin/env bash
# Pull Gemma 3 27B variants for RTX 5090 (32 GB VRAM).
# See docs/KIMI_VS_GEMMA.md for the decision rationale.
#
# Usage:
#   bash scripts/gemma_pulls.sh            # pull all recommended variants
#   bash scripts/gemma_pulls.sh quality    # just the narration variant
#   bash scripts/gemma_pulls.sh speed      # just the agent-loop variant
#
# These are Unsloth's GGUF releases, pulled via Ollama's HuggingFace pass-through.

set -euo pipefail

MODE="${1:-all}"

pull_quality() {
    echo "[gemma] Pulling Q8_0 (~29GB) — max quality for tier 5-6 narration"
    ollama pull hf.co/unsloth/gemma-3-27b-it-GGUF:Q8_0
}

pull_speed() {
    echo "[gemma] Pulling Q5_K_M (~19GB) — fast agent loop for Hermes bot"
    ollama pull hf.co/unsloth/gemma-3-27b-it-GGUF:Q5_K_M
}

pull_balanced() {
    echo "[gemma] Pulling Q6_K (~22GB) — good quality + long context room"
    ollama pull hf.co/unsloth/gemma-3-27b-it-GGUF:Q6_K
}

case "$MODE" in
    all)
        pull_quality
        pull_speed
        pull_balanced
        ;;
    quality)
        pull_quality
        ;;
    speed)
        pull_speed
        ;;
    balanced)
        pull_balanced
        ;;
    *)
        echo "Unknown mode: $MODE. Use all|quality|speed|balanced"
        exit 1
        ;;
esac

echo ""
echo "[gemma] Verifying models present..."
ollama list | grep -i "gemma-3-27b" || echo "WARNING: no gemma-3-27b models listed"

echo ""
echo "[gemma] Done. Model tags to use in ollama_client.py:"
echo "  Quality (narration):   hf.co/unsloth/gemma-3-27b-it-GGUF:Q8_0"
echo "  Speed (Hermes agent):  hf.co/unsloth/gemma-3-27b-it-GGUF:Q5_K_M"
echo "  Balanced (default):    hf.co/unsloth/gemma-3-27b-it-GGUF:Q6_K"
