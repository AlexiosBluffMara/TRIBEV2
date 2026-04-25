"""Compute paper-ready statistics on v2 brain-LoRA eval outputs.

Loads the base vs adapter eval dirs under C:/Users/soumi/AppData/Local/Temp/eval_brain_*
and computes:

- Template adherence: fraction opening with "The stimulus"
- Disclaimer presence: fraction mentioning TRIBE-v2 + not-diagnostic tail
- Yeo-7 network mention count (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default)
- Mean word count + char count
- Type-token ratio (lexical diversity)
- Paired per-sample deltas

Writes CSV + a markdown summary to D:/TRIBEV2/outputs/paper/eval_stats/.

Usage:
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe scripts/compute_eval_stats.py
"""
from __future__ import annotations

import io
import json
import random
import re
import sys
from pathlib import Path
from statistics import mean, stdev

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

EVAL_DIRS = [
    Path('C:/Users/soumi/AppData/Local/Temp/eval_brain_1776633841'),
    Path('C:/Users/soumi/AppData/Local/Temp/eval_brain_1776637687'),
]
OUT_DIR = Path('D:/TRIBEV2/outputs/paper/eval_stats')

YEO7 = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

YEO7_ALIASES = {
    'Vis'        : [r'\bVis\b', r'\bvisual\b'],
    'SomMot'     : [r'\bSomMot\b', r'\bsomatomotor\b', r'\bsensorimotor\b'],
    'DorsAttn'   : [r'\bDorsAttn\b', r'\bdorsal\s+attention\b'],
    'SalVentAttn': [r'\bSalVentAttn\b', r'\bsalience\b', r'\bventral\s+attention\b'],
    'Limbic'     : [r'\bLimbic\b', r'\blimbic\b'],
    'Cont'       : [r'\bCont\b', r'\bcontrol\b'],
    'Default'    : [r'\bDefault\b', r'\bdefault\s+mode\b', r'\bDMN\b'],
}

_TEMPLATE_OPENER  = re.compile(r'^\s*The stimulus', re.IGNORECASE)
_DISCLAIMER_RE    = re.compile(r'group[- ]averaged\s+TRIBE[- ]v2\s+prediction', re.IGNORECASE)
_NOT_DIAG_RE      = re.compile(r'not\s+a\s+diagnostic', re.IGNORECASE)
_ROI_VERBATIM_RE  = re.compile(r'\b7Networks_[A-Z]H_[A-Za-z]+_[A-Za-z0-9_]+\b')
_PEAK_TIME_MENTION = re.compile(r'\b\d+\.?\d*\s*s(?:econds?)?\b')


def _words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9'_\-]*", s)


def _ttr(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(w.lower() for w in words)) / len(words)


def _networks_mentioned_strict(s: str) -> int:
    return sum(1 for n in YEO7 if re.search(rf'\b{n}\b', s))


def _networks_mentioned_any(s: str) -> int:
    return sum(1 for pats in YEO7_ALIASES.values() if any(re.search(p, s, re.IGNORECASE) for p in pats))


def _score(text: str) -> dict:
    ws = _words(text)
    return {
        'opens_with_template': bool(_TEMPLATE_OPENER.search(text)),
        'has_tribe_disclaimer': bool(_DISCLAIMER_RE.search(text)),
        'has_not_diagnostic': bool(_NOT_DIAG_RE.search(text)),
        'yeo7_networks_mentioned': _networks_mentioned_strict(text),
        'yeo7_any_alias': _networks_mentioned_any(text),
        'roi_verbatim_count': len(_ROI_VERBATIM_RE.findall(text)),
        'mentions_peak_time': bool(_PEAK_TIME_MENTION.search(text)),
        'n_words': len(ws),
        'n_chars': len(text),
        'ttr': _ttr(ws),
    }


def _agg(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for k in keys:
        vals = [float(r[k]) for r in rows]
        out[k + '_mean'] = mean(vals) if vals else 0.0
        out[k + '_std']  = stdev(vals) if len(vals) >= 2 else 0.0
    return out


def _bootstrap_ci(values: list[float], *, iters: int = 5000, alpha: float = 0.05,
                  seed: int = 13) -> tuple[float, float]:
    if len(values) < 2:
        return (float('nan'), float('nan'))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iters):
        samp = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(samp) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return lo, hi


def _paired_sign_test(deltas: list[int | float]) -> dict:
    """Two-sided sign test on paired deltas. Returns exact p-value via binomial."""
    nz = [d for d in deltas if d != 0]
    if not nz:
        return {'n_nonzero': 0, 'n_positive': 0, 'p_two_sided': 1.0}
    n = len(nz)
    k = sum(1 for d in nz if d > 0)
    # exact binomial two-sided via symmetry
    from math import comb
    def _p_ge(kk, nn):
        return sum(comb(nn, i) for i in range(kk, nn + 1)) / (2 ** nn)
    p_lo = _p_ge(k, n)
    p_hi = _p_ge(n - k, n)
    p = min(1.0, 2.0 * min(p_lo, p_hi))
    return {'n_nonzero': n, 'n_positive': k, 'p_two_sided': p}


def _load_dir(d: Path) -> tuple[list[dict], list[dict]] | None:
    base_p = d / 'base_outputs.json'
    adap_p = d / 'adapter_outputs.json'
    if not (base_p.exists() and adap_p.exists()):
        return None
    base = json.loads(base_p.read_text(encoding='utf-8'))
    adap = json.loads(adap_p.read_text(encoding='utf-8'))
    return base, adap


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_base: list[dict] = []
    all_adap: list[dict] = []
    pairs: list[dict] = []

    for d in EVAL_DIRS:
        loaded = _load_dir(d)
        if loaded is None:
            print(f'[stats] skip (incomplete): {d}')
            continue
        base_rows, adap_rows = loaded
        for b, a in zip(base_rows, adap_rows):
            if b['prompt'] != a['prompt']:
                print(f'[stats] WARN prompt mismatch in {d}')
                continue
            b_s = _score(b['completion'])
            a_s = _score(a['completion'])
            all_base.append(b_s)
            all_adap.append(a_s)
            pairs.append({
                'dir'    : d.name,
                'prompt' : b['prompt'][:80],
                'base'   : b_s,
                'adapter': a_s,
            })

    n = len(pairs)
    if n == 0:
        print('[stats] no paired samples found — abort')
        return

    keys_bin = ['opens_with_template', 'has_tribe_disclaimer', 'has_not_diagnostic', 'mentions_peak_time']
    keys_num = ['yeo7_networks_mentioned', 'yeo7_any_alias', 'roi_verbatim_count', 'n_words', 'n_chars', 'ttr']

    base_summary: dict = {}
    adap_summary: dict = {}
    for k in keys_bin:
        base_summary[k + '_rate'] = mean(float(r[k]) for r in all_base)
        adap_summary[k + '_rate'] = mean(float(r[k]) for r in all_adap)
    base_summary.update(_agg(all_base, keys_num))
    adap_summary.update(_agg(all_adap, keys_num))

    deltas = {
        **{f'{k}_delta': adap_summary[k + '_rate'] - base_summary[k + '_rate'] for k in keys_bin},
        **{f'{k}_mean_delta': adap_summary[k + '_mean'] - base_summary[k + '_mean'] for k in keys_num},
    }

    # Per-sample paired deltas for paired-test style observation
    paired = []
    for p in pairs:
        b = p['base']; a = p['adapter']
        paired.append({
            'opens_with_template_delta': int(a['opens_with_template']) - int(b['opens_with_template']),
            'has_tribe_disclaimer_delta': int(a['has_tribe_disclaimer']) - int(b['has_tribe_disclaimer']),
            'yeo7_strict_delta': a['yeo7_networks_mentioned'] - b['yeo7_networks_mentioned'],
            'yeo7_alias_delta': a['yeo7_any_alias'] - b['yeo7_any_alias'],
            'roi_verbatim_delta': a['roi_verbatim_count'] - b['roi_verbatim_count'],
            'n_words_delta': a['n_words'] - b['n_words'],
            'ttr_delta': a['ttr'] - b['ttr'],
        })

    paired_mean = {k: mean(p[k] for p in paired) for k in paired[0].keys()}
    paired_std  = {k: (stdev(p[k] for p in paired) if len(paired) >= 2 else 0.0) for k in paired[0].keys()}

    # 95% bootstrap CI on paired mean deltas
    paired_ci = {k: _bootstrap_ci([p[k] for p in paired]) for k in paired[0].keys()}

    # Sign-test p-values per paired delta
    paired_sign = {k: _paired_sign_test([p[k] for p in paired]) for k in paired[0].keys()}

    # 95% bootstrap CI for per-output rates (binary) and per-output means (numeric)
    summary_ci_base: dict = {}
    summary_ci_adap: dict = {}
    for k in keys_bin:
        summary_ci_base[k + '_rate'] = _bootstrap_ci([float(r[k]) for r in all_base])
        summary_ci_adap[k + '_rate'] = _bootstrap_ci([float(r[k]) for r in all_adap])
    for k in keys_num:
        summary_ci_base[k + '_mean'] = _bootstrap_ci([float(r[k]) for r in all_base])
        summary_ci_adap[k + '_mean'] = _bootstrap_ci([float(r[k]) for r in all_adap])

    # Build markdown summary
    lines = []
    lines.append(f'# v2 brain-LoRA eval statistics (n={n} paired samples)\n')
    lines.append('## Style-transfer rates (binary)\n')
    lines.append('| metric | base | adapter | delta |')
    lines.append('|---|---:|---:|---:|')
    for k in keys_bin:
        b_r = base_summary[k + '_rate']
        a_r = adap_summary[k + '_rate']
        lines.append(f'| {k} | {b_r:.3f} | {a_r:.3f} | {a_r - b_r:+.3f} |')
    lines.append('\n## Continuous metrics (mean ± std)\n')
    lines.append('| metric | base | adapter | mean delta |')
    lines.append('|---|---:|---:|---:|')
    for k in keys_num:
        b_m = base_summary[k + '_mean']; b_s = base_summary[k + '_std']
        a_m = adap_summary[k + '_mean']; a_s = adap_summary[k + '_std']
        lines.append(f'| {k} | {b_m:.3f} ± {b_s:.3f} | {a_m:.3f} ± {a_s:.3f} | {a_m - b_m:+.3f} |')
    lines.append('\n## Paired within-sample deltas (adapter - base), 95% bootstrap CI + sign test\n')
    lines.append('| metric | mean ± std | 95% CI | sign-test p (two-sided) |')
    lines.append('|---|---:|---:|---:|')
    for k, v in paired_mean.items():
        lo, hi = paired_ci[k]
        s = paired_sign[k]
        p_str = f"{s['p_two_sided']:.3g}" if isinstance(s['p_two_sided'], float) else str(s['p_two_sided'])
        lines.append(f'| {k} | {v:+.3f} ± {paired_std[k]:.3f} | [{lo:+.3f}, {hi:+.3f}] | {p_str} |')

    md = '\n'.join(lines) + '\n'
    (OUT_DIR / 'eval_stats_v2.md').write_text(md, encoding='utf-8')

    combined = {
        'n_pairs': n,
        'eval_dirs': [str(d) for d in EVAL_DIRS if (d / 'adapter_outputs.json').exists()],
        'base_summary': base_summary,
        'adapter_summary': adap_summary,
        'summary_ci_base': {k: list(v) for k, v in summary_ci_base.items()},
        'summary_ci_adapter': {k: list(v) for k, v in summary_ci_adap.items()},
        'aggregate_deltas': deltas,
        'paired_mean_deltas': paired_mean,
        'paired_std_deltas': paired_std,
        'paired_ci_deltas': {k: list(v) for k, v in paired_ci.items()},
        'paired_sign_tests': paired_sign,
    }
    (OUT_DIR / 'eval_stats_v2.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')

    # Per-sample CSV
    csv_lines = ['dir,prompt80,base_template,adap_template,base_disclaimer,adap_disclaimer,'
                 'base_yeo7,adap_yeo7,base_words,adap_words,base_ttr,adap_ttr']
    for p in pairs:
        b = p['base']; a = p['adapter']
        row = [
            p['dir'],
            '"' + p['prompt'].replace('"', "'") + '"',
            int(b['opens_with_template']), int(a['opens_with_template']),
            int(b['has_tribe_disclaimer']), int(a['has_tribe_disclaimer']),
            b['yeo7_networks_mentioned'], a['yeo7_networks_mentioned'],
            b['n_words'], a['n_words'],
            f"{b['ttr']:.4f}", f"{a['ttr']:.4f}",
        ]
        csv_lines.append(','.join(str(x) for x in row))
    (OUT_DIR / 'eval_stats_v2.csv').write_text('\n'.join(csv_lines) + '\n', encoding='utf-8')

    print(md)
    print(f'\n[stats] wrote {OUT_DIR}/eval_stats_v2.{{md,json,csv}}')


if __name__ == '__main__':
    main()
