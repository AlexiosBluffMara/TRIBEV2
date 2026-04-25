#!/usr/bin/env bash
# Quick dashboard of active research jobs + recent artifacts.
# Usage: bash scripts/status.sh

set -u

echo "=== $(date) ==="
echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>/dev/null \
  | awk -F', ' '{ printf "used=%sMB  free=%sMB  total=%sMB  util=%s%%  temp=%sC\n", $1, $2, $3, $4, $5 }' \
  || echo "(nvidia-smi unavailable)"
echo ""

echo "--- Running research python procs ---"
powershell -NoProfile -Command "Get-WmiObject Win32_Process | Where-Object { \$_.CommandLine -like '*python*' -and (\$_.CommandLine -like '*TRIBEV2*' -or \$_.CommandLine -like '*autoresearch*' -or \$_.CommandLine -like '*lm_eval*' -or \$_.CommandLine -like '*run_genuine*' -or \$_.CommandLine -like '*finetune_*' -or \$_.CommandLine -like '*pull_kaggle*') } | Select-Object ProcessId, @{Name='Start';Expression={\$_.ConvertToDateTime(\$_.CreationDate).ToString('HH:mm:ss')}}, @{Name='Cmd';Expression={(\$_.CommandLine -split '\\\\')[-1].Substring(0,[math]::Min(110,(\$_.CommandLine -split '\\\\')[-1].Length))}} | Format-Table -Auto" 2>/dev/null \
  | head -20
echo ""

echo "--- Recent benchmarks ---"
for d in $(ls -t D:/research/benchmarks/ 2>/dev/null | head -5); do
  summary="D:/research/benchmarks/$d/summary.csv"
  if [[ -f "$summary" ]]; then
    rows=$(wc -l < "$summary")
    echo "  [OK] $d  summary.csv ($rows lines)"
  else
    echo "  [..] $d  no summary.csv"
  fi
done
echo ""

echo "--- Pipeline state (new scheduler) ---"
if [[ -d D:/research/pipeline ]]; then
  latest_run=$(ls -td D:/research/pipeline/*/ 2>/dev/null | head -1)
  if [[ -n "$latest_run" && -f "$latest_run/heartbeat.json" ]]; then
    C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe -m pipeline \
      --root D:/research/pipeline status 2>/dev/null | sed 's/^/  /'
  else
    echo "  (no active runs — start with: python -m pipeline escalate|sweep)"
  fi
else
  echo "  (pipeline/ dir absent — new scheduler not yet used)"
fi
echo ""

echo "--- Autoresearch state ---"
# Regenerate LEADERBOARD.md with the current writer so the Δ-vs-base column
# reflects the latest code even if the running loop imported an older version.
C:/Users/soumi/TRIBEV2/.venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'D:/TRIBEV2/scripts')
from pathlib import Path
from autoresearch_loop import _update_leaderboard
_update_leaderboard(Path('D:/research/autoresearch'))
" 2>/dev/null
if [[ -f D:/research/autoresearch/LEADERBOARD.md ]]; then
  n_results=$(ls D:/research/autoresearch/results/*.json 2>/dev/null | wc -l)
  echo "  results: $n_results"
  echo "  top 5:"
  sed -n '7,11p' D:/research/autoresearch/LEADERBOARD.md | head -6 | sed 's/^/    /'
fi
latest_sess=$(ls -t D:/research/autoresearch/sessions/ 2>/dev/null | head -1)
if [[ -n "$latest_sess" ]]; then
  echo "  latest session: $latest_sess"
  if [[ -f "D:/research/autoresearch/sessions/$latest_sess/try_1.log" ]]; then
    tail -3 "D:/research/autoresearch/sessions/$latest_sess/try_1.log" | sed 's/^/    /'
  fi
fi
echo ""

echo "--- Pull logs ---"
for f in $(ls -t D:/research/logs/neuro_pulls_*.log 2>/dev/null | head -3); do
  ok=$(grep -c "\bOK\b" "$f" 2>/dev/null)
  fail=$(grep -c "\bFAIL\b" "$f" 2>/dev/null)
  printf "  %s  ok=%s  fail=%s\n" "$(basename "$f")" "${ok:-0}" "${fail:-0}"
done
echo ""

echo "--- Corpora on disk ---"
for d in $(ls D:/research/corpora/ 2>/dev/null); do
  sz=$(du -sh "D:/research/corpora/$d" 2>/dev/null | awk '{print $1}')
  printf "  %-32s  %s\n" "$d" "${sz:-?}"
done
echo ""

echo "--- Latest curricula ---"
ls -lt D:/research/datasets/curriculum_v*.jsonl 2>/dev/null | head -5 \
  | awk '{ printf "  %s  %s  %s  %s\n", $6, $7, $8, $NF }'
echo ""

echo "--- HF cache (gemma-4) ---"
HF_HUB="${HF_HOME:-D:/unsloth/hf_cache}/hub"
for m in gemma-4-e4b-it-unsloth-bnb-4bit gemma-4-31B-it-unsloth-bnb-4bit gemma-4-E4B-it; do
  d="$HF_HUB/models--unsloth--$m"
  if [[ -d "$d" ]]; then
    incomplete=$(find "$d/blobs" -name '*.incomplete' 2>/dev/null | wc -l)
    total=$(du -sh "$d" 2>/dev/null | awk '{print $1}')
    if (( incomplete > 0 )); then
      printf "  %-40s  %s  [%d blob(s) downloading]\n" "$m" "$total" "$incomplete"
    else
      printf "  %-40s  %s  [complete]\n" "$m" "$total"
    fi
  else
    printf "  %-40s  [not cached]\n" "$m"
  fi
done
echo ""

echo "--- Latest adapters ---"
ls -lt D:/research/weights/ 2>/dev/null | head -6 | awk 'NR>1 { printf "  %s  %s  %s\n", $6, $7, $NF }'
echo ""

echo "--- Bench matrix ---"
if [[ -f D:/research/BENCH_MATRIX.md ]]; then
  rows=$(wc -l < D:/research/BENCH_MATRIX.md)
  echo "  BENCH_MATRIX.md: $rows lines"
  sed -n '1,8p' D:/research/BENCH_MATRIX.md | sed 's/^/    /'
else
  echo "  (BENCH_MATRIX.md not generated yet — run compile_bench_table.py)"
fi
echo ""

echo "--- Bench deltas (top 5) ---"
if [[ -f D:/research/BENCH_DELTAS.md ]]; then
  sed -n '1,12p' D:/research/BENCH_DELTAS.md | sed 's/^/    /'
else
  echo "  (no deltas yet)"
fi
echo ""

echo "--- Best adapter ---"
if [[ -f D:/research/BEST_ADAPTER.md ]]; then
  # Top row + verdict
  sed -n '/^| 1 /p' D:/research/BEST_ADAPTER.md | head -1 | sed 's/^/    /'
  grep -E '^\*\*(ESCALATE|CONDITIONAL|REJECT|INCONCLUSIVE)\*\*' D:/research/BEST_ADAPTER.md \
    | head -1 | sed 's/^/    /'
else
  echo "  (not computed — run scripts/best_adapter.py)"
fi
