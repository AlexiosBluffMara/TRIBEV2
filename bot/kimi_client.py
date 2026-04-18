"""
Kimi K2.5 API client (Moonshot AI, OpenAI-compatible)
Used for:
  - Generating custom Three.js GLSL shader variations
  - Designing colour palette suggestions for the BOLD heatmap
  - Vision analysis of cortical activation screenshots
  - Creative cross-modal descriptions (brain state → art)

API: https://api.moonshot.cn/v1  (or api.moonshot.ai/v1 outside China)
Model: moonshot-v1-128k  (Kimi K2.5 — 256K context, vision)

Requires: KIMI_API_KEY in .env
"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger('jemma.kimi')

KIMI_BASE  = os.getenv('KIMI_BASE_URL', 'https://api.moonshot.cn/v1')
KIMI_MODEL = os.getenv('KIMI_MODEL', 'moonshot-v1-128k')

# ── Kimi-specific creative prompts ────────────────────────────────────────────

SHADER_DESIGN_PROMPT = """\
You are a creative GLSL shader designer specializing in neuroscience visualization.

The system renders a 3D cortical brain surface (fsaverage5, 20,484 vertices) with
per-vertex BOLD z-score values. The current colour map uses an inferno gradient.

Your task: propose an enhanced GLSL fragment shader that:
1. Uses a richer colour map tailored for neuroscience aesthetics
2. Adds subtle animated pulsing at high-activation vertices (use u_time uniform)
3. Creates a chromatic aberration rim effect at the brain silhouette
4. Keeps the shader under 60 lines and WebGL2-compatible

Reply with ONLY the GLSL fragment shader code, no explanation.
"""

BRAIN_ART_PROMPT = """\
You are an interdisciplinary artist and neuroscientist.

A participant just watched "{stimulus}". Their cortical BOLD response showed:
- Peak activation at {peak_t:.1f}s
- Dominant network: {dominant_network}
- {cortex_pct:.0f}% of cortex active above threshold
- Strongest ROIs: {top_rois}

Compose a single evocative paragraph (≤120 words) that translates this brain state
into vivid sensory language — as if describing the neural firing as music, colour,
or texture. Be precise, poetic, and scientifically grounded.
"""

COLOUR_PALETTE_PROMPT = """\
You are a data visualization expert designing for a neuroscience dashboard.

The dashboard uses:
  - Background: #080c14 (deep midnight blue)
  - Accent: #5865f2 (Discord purple)
  - Brain BOLD heatmap: currently inferno colourmap

Propose 3 alternative GLSL vec3[7] colour gradient stop arrays for the BOLD heatmap.
Each should:
  1. Work well on a dark background
  2. Be perceptually uniform
  3. Be appropriate for scientific publication

Format as JSON: { "palettes": [ { "name": str, "stops": [[r,g,b]×7] }, ...] }
"""

# ── Client ────────────────────────────────────────────────────────────────────

class KimiClient:
    """Thin synchronous wrapper around the Kimi K2.5 / Moonshot OpenAI-compatible API."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key  = api_key or os.getenv('KIMI_API_KEY', '')
        self.base_url = (base_url or KIMI_BASE).rstrip('/')
        self.model    = KIMI_MODEL
        if not self.api_key:
            log.warning('KIMI_API_KEY not set — Kimi features will be skipped')

    @property
    def _headers(self):
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type':  'application/json',
        }

    # ── Text generation ───────────────────────────────────────────────────────

    def chat(self, messages: list[dict], max_tokens: int = 1024,
             temperature: float = 0.7) -> str:
        """Call Kimi chat completions, return content string."""
        if not self.api_key:
            return '[Kimi API key not configured]'
        try:
            resp = requests.post(
                f'{self.base_url}/chat/completions',
                headers=self._headers,
                json={
                    'model':       self.model,
                    'messages':    messages,
                    'max_tokens':  max_tokens,
                    'temperature': temperature,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']
        except Exception as e:
            log.error('Kimi chat error: %s', e)
            return f'[Kimi error: {e}]'

    def design_shader(self) -> str:
        """Ask Kimi to generate an enhanced BOLD fragment shader."""
        return self.chat([
            {'role': 'user', 'content': SHADER_DESIGN_PROMPT}
        ], max_tokens=600, temperature=0.8)

    def brain_to_art(self, stimulus: str, peak_t: float, dominant_network: str,
                     cortex_pct: float, top_rois: list) -> str:
        """Creative translation of brain state → poetic text."""
        prompt = BRAIN_ART_PROMPT.format(
            stimulus=stimulus,
            peak_t=peak_t,
            dominant_network=dominant_network,
            cortex_pct=cortex_pct,
            top_rois=', '.join(top_rois[:3]) if top_rois else 'unknown',
        )
        return self.chat([
            {'role': 'user', 'content': prompt}
        ], max_tokens=200, temperature=0.9)

    def design_colour_palettes(self) -> dict:
        """Ask Kimi for alternative colour palettes; returns parsed JSON or {}."""
        raw = self.chat([
            {'role': 'user', 'content': COLOUR_PALETTE_PROMPT}
        ], max_tokens=800, temperature=0.6)
        import json as _json
        try:
            start = raw.index('{')
            end   = raw.rindex('}') + 1
            return _json.loads(raw[start:end])
        except Exception:
            return {}

    # ── Vision (image) ────────────────────────────────────────────────────────

    def analyse_cortex_image(self, image_path: Path, question: str = '') -> str:
        """
        Send a PNG/JPG of the cortical activation map to Kimi for vision analysis.
        Kimi K2.5 supports image inputs via base64.
        """
        if not self.api_key:
            return '[Kimi API key not configured]'
        try:
            img_b64 = base64.b64encode(image_path.read_bytes()).decode()
            suffix  = image_path.suffix.lower().lstrip('.')
            mime    = 'image/png' if suffix == 'png' else 'image/jpeg'
            prompt  = question or (
                'Describe the cortical activation pattern in this brain visualization. '
                'What regions appear most active? What does this pattern suggest cognitively?'
            )
            return self.chat([
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{img_b64}'}},
                        {'type': 'text',      'text': prompt},
                    ],
                }
            ], max_tokens=300, temperature=0.4)
        except Exception as e:
            log.error('Kimi vision error: %s', e)
            return f'[Kimi vision error: {e}]'

    # ── Async wrappers ────────────────────────────────────────────────────────

    async def async_brain_to_art(self, *args, **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.brain_to_art(*args, **kwargs))

    async def async_design_shader(self) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.design_shader)

    async def async_colour_palettes(self) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.design_colour_palettes)


# ── Singleton ─────────────────────────────────────────────────────────────────
_client: Optional[KimiClient] = None

def get_kimi() -> KimiClient:
    global _client
    if _client is None:
        _client = KimiClient()
    return _client


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    kimi = get_kimi()
    if not kimi.api_key:
        print('Set KIMI_API_KEY in your .env file')
        sys.exit(1)

    print('Testing Kimi K2.5 connection…')
    result = kimi.chat([{'role': 'user', 'content': 'Reply with exactly: KIMI_OK'}], max_tokens=10)
    print('Result:', result)

    print('\nDesigning colour palettes…')
    palettes = kimi.design_colour_palettes()
    import json
    print(json.dumps(palettes, indent=2))
