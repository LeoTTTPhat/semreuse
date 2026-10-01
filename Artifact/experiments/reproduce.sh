#!/usr/bin/env bash
# Regenerate the result files of reference_results/ into results/.
#
# Usage:  bash experiments/reproduce.sh [cpu|llm|matrices|lotus] ...
#
#   cpu       (default) every result that needs no live LLM: the simulated-
#             oracle experiments, replays of the shipped LLM answer matrices,
#             the analyses and the figures.  About 1.5-2 h on one machine.
#   llm       the steps that query a local Ollama server: oracle determinism
#             (exp13) and the LLM arbiter (exp10).  Needs the models pulled.
#   matrices  re-assemble the LLM answer matrices in data/llm_matrix/ through
#             data/llm_cache/ (0 live calls while the caches are complete; to
#             re-query a model, point --cache at a new file, see README).
#   lotus     SemReuse under the released LOTUS package (exp12, exp18).  Needs
#             .venv-lotus and Ollama; exp18 runs for many hours.
#
# Per-step logs go to results/logs/<step>.log.  Afterwards compare with
#   .venv/bin/python experiments/compare_results.py
#
# Environment overrides: PYTHON (main interpreter, default .venv/bin/python),
# LOTUS_PYTHON (default .venv-lotus/bin/python), OLLAMA_URL (default
# http://localhost:11434).
set -u
cd "$(dirname "$0")/.."
V=${PYTHON:-.venv/bin/python}
VL=${LOTUS_PYTHON:-.venv-lotus/bin/python}
HOST=${OLLAMA_URL:-http://localhost:11434}
mkdir -p results/logs
FAILED=0

run () {                      # run <step> <script> <args...>
  local name="$1"; shift
  echo "=== $name  ($(date +%H:%M:%S)) ==="
  "$V" -u "$@" > "results/logs/${name}.log" 2>&1 \
    && grep -E "^(\[|  \[)" "results/logs/${name}.log" | tail -12 \
    || { echo "FAILED: $name (see results/logs/${name}.log)"; \
         tail -5 "results/logs/${name}.log"; FAILED=$((FAILED + 1)); }
}

M=data/llm_matrix
L2000=$M/frozen_analyst_20ng_2000_llama3.1-8b.npz      # Llama-3.1-8B, 2,000 rows (frozen)
L4000=$M/analyst_20ng_4000_llama3.1-8b-instruct-q4_K_M.npz
QWEN=$M/analyst_20ng_4000_qwen2.5-14b-instruct-q4_K_M.npz
MISTRAL=$M/analyst_20ng_4000_mistral-small3.1-24b.npz
GEMMA=$M/analyst_20ng_4000_gemma3-27b.npz

tier_cpu () {
  # The analyses at the end read outputs of the live-LLM steps (exp10, exp12,
  # exp13, exp18).  Unless those tiers have been run, use the recorded ones.
  local seeded=0
  for f in reference_results/exp10_* reference_results/exp12_* \
           reference_results/exp13_* reference_results/exp18_lotus_*; do
    [ -e "results/$(basename "$f")" ] || { cp "$f" results/; seeded=$((seeded + 1)); }
  done
  [ $seeded -gt 0 ] && echo "(copied $seeded recorded live-LLM outputs into results/)"

  # -- simulated oracle: 20 Newsgroups, AG News, RCV1 ------------------------
  run exp1_20ng       experiments/exp1_e2e.py --scale full --corpus 20ng   --overlap 0.8
  run exp1_agnews     experiments/exp1_e2e.py --scale full --corpus agnews --overlap 0.8
  run exp1_noise05    experiments/exp1_e2e.py --scale full --corpus 20ng   --overlap 0.8 \
                      --oracle-noise 0.05 --out-tag noise05
  run exp1_calib900   experiments/exp1_e2e.py --scale full --corpus 20ng   --overlap 0.8 \
                      --calib-pairs 900 --out-tag calib900 \
                      --methods cold,semreuse,semreuse-noaudit
  run exp2            experiments/exp2_entailment.py --scale medium --corpus 20ng
  run exp3            experiments/exp3_audit.py --scale medium --corpus 20ng
  run exp4            experiments/exp4_overlap.py --scale full --corpus 20ng
  run exp4_stats      experiments/exp4_workload_stats.py --scale full --corpus 20ng
  run exp5            experiments/exp5_ablation.py --scale full --corpus 20ng
  run exp6            experiments/exp6_scaling.py --corpus 20ng --sizes 500,1000,2000,4000,8000
  run exp7_20ng       experiments/exp7_tau.py --scale full --corpus 20ng
  run exp7_agnews     experiments/exp7_tau.py --scale full --corpus agnews
  run exp9            experiments/exp9_proxy.py --scale full --corpus 20ng \
                      --pilot 400 --min-kept-recall 0.6
  run exp9_sweep      experiments/exp9_proxy.py --scale full --corpus 20ng --sweep
  local P="proxy,semreuse+proxy,semreuse-gt+proxy"
  run exp9_novalidate experiments/exp9_proxy.py --scale full --corpus 20ng \
                      --pilot 400 --min-kept-recall 0.0 --methods $P --out-tag novalidate
  run exp9_bands1     experiments/exp9_proxy.py --scale full --corpus 20ng \
                      --pilot 400 --min-kept-recall 0.6 --bands 1 --methods $P --out-tag bands1
  run exp9_bands2     experiments/exp9_proxy.py --scale full --corpus 20ng \
                      --pilot 400 --min-kept-recall 0.6 --bands 2 --methods $P --out-tag bands2
  run exp11_agnews    experiments/exp11_scale.py --corpus agnews --scale full \
                      --sizes 8000,30000,120000 --n-queries 120
  run exp11_rcv1      experiments/exp11_scale.py --corpus rcv1 --scale full \
                      --sizes 25000,100000,400000,804414 --n-queries 120
  run exp14_sim_nli   experiments/exp14_slack.py --oracle sim --corpus 20ng --scale full \
                      --out-tag sim_nli --budgets none,1.0,0.5,0.25
  run exp14_sim_err30 experiments/exp14_slack.py --oracle sim --corpus 20ng --scale full \
                      --entailment-error 0.3 --out-tag sim_err30 --budgets none,1.0,0.5,0.25
  run exp15_20ng      experiments/exp15_audit_design.py --setting 20ng --reasoners nli,gt
  run exp15_agnews    experiments/exp15_audit_design.py --setting agnews \
                      --sizes 8000,120000 --reasoners nli,gt
  run exp16_primitives experiments/exp16_coverage.py --part primitives
  run exp16_synthetic experiments/exp16_coverage.py --part synthetic --reps 20000
  run exp16_replay_20ng experiments/exp16_coverage.py --part replay --reps 1000 --out-tag 20ng

  # -- real LLM oracles, replayed from the recorded answer matrices -----------
  run exp8_llama      experiments/exp8_realllm.py --matrix $L2000 --scale full \
                      --sizes 500,1000,2000
  run exp8_eps10      experiments/exp8_realllm.py --matrix $L2000 --scale full \
                      --eps 0.10 --out-tag eps0.10
  run exp8_n4000      experiments/exp8_realllm.py --matrix $L4000 \
                      --sizes 500,1000,2000,3000,4000 --out-tag n4000
  run exp8_qwen14b    experiments/exp8_realllm.py --matrix $QWEN --scale full \
                      --sizes 500,1000,2000 --out-tag qwen14b
  run exp8_mistral24b experiments/exp8_realllm.py --matrix $MISTRAL --scale full \
                      --sizes 500,1000,2000 --out-tag mistral24b
  run exp8_gemma27b   experiments/exp8_realllm.py --matrix $GEMMA --scale full \
                      --out-tag gemma27b_n500
  run exp14_real      experiments/exp14_slack.py --oracle real --scale full --matrix $L2000 \
                      --out-tag real --budgets none,1.0,0.5,0.25,0.1
  run exp14_feat      experiments/exp14_slack.py --oracle real --scale full --matrix $L2000 \
                      --out-tag feat --dump-slack --budgets none
  run exp15_llama     experiments/exp15_audit_design.py --setting real --reasoners nli,gt
  run exp15_qwen14b   experiments/exp15_audit_design.py --setting real --reasoners nli,gt \
                      --matrix $QWEN --out-tag qwen14b
  run exp15_mistral24b experiments/exp15_audit_design.py --setting real --reasoners nli,gt \
                      --matrix $MISTRAL --out-tag mistral24b
  run exp16_replay_llama   experiments/exp16_coverage.py --part replay --reps 1000 \
                           --matrix $L2000 --out-tag llama
  run exp16_replay_qwen14b experiments/exp16_coverage.py --part replay --reps 1000 \
                           --matrix $QWEN --out-tag qwen14b
  run exp16_replay_mistral24b experiments/exp16_coverage.py --part replay --reps 1000 \
                           --matrix $MISTRAL --out-tag mistral24b

  # -- analyses and figures ---------------------------------------------------
  run exp17           experiments/exp17_multi_oracle.py
  run exp18_analyze   experiments/exp18_analyze.py --n 1000 --queries 47
  run summary         experiments/summary_numbers.py
  run figures         experiments/make_figures.py
  run figures_cost    experiments/make_cost_figures.py
}

tier_llm () {
  run exp13_llama      experiments/exp13_determinism.py --host $HOST --n-pairs 200
  run exp13_qwen14b    experiments/exp13_determinism.py --host $HOST \
                       --model qwen2.5:14b-instruct-q4_K_M \
                       --cache data/llm_cache/20ng_4000_0_qwen14b.sqlite \
                       --n-pairs 200 --concurrency 1 --rows 2000 --out-tag qwen14b
  run exp13_mistral24b experiments/exp13_determinism.py --host $HOST \
                       --model mistral-small3.1:24b \
                       --cache data/llm_cache/20ng_4000_0_mistral24b.sqlite \
                       --n-pairs 200 --concurrency 1 --rows 2000 --out-tag mistral24b
  run exp13_gemma27b   experiments/exp13_determinism.py --host $HOST --model gemma3:27b \
                       --n-pairs 200 --concurrency 1 --out-tag gemma27b
  # The arbiter talks to the default Ollama endpoint (localhost:11434).
  run exp10            experiments/exp10_arbiter.py --scale full --corpus 20ng \
                       --arbiter-model llama3.1:8b-instruct-q4_K_M --parts pairs,e2e
}

tier_matrices () {
  # The builds rewrite data/llm_matrix/ in place; keep the shipped matrices
  # for comparison (compare_results.py --results data/llm_matrix
  # --reference data/llm_matrix.orig).
  [ -d data/llm_matrix.orig ] || cp -R data/llm_matrix data/llm_matrix.orig
  local B="experiments/build_llm_matrix.py --workload analyst --corpus 20ng --n 4000 --seed 0 --host $HOST"
  run matrix_llama   $B --rows 4000 --model llama3.1:8b-instruct-q4_K_M --concurrency 6
  run matrix_qwen    $B --rows 2000 --model qwen2.5:14b-instruct-q4_K_M --order doc \
                     --cache data/llm_cache/20ng_4000_0_qwen14b.sqlite
  run matrix_mistral $B --rows 2000 --model mistral-small3.1:24b --order doc \
                     --cache data/llm_cache/20ng_4000_0_mistral24b.sqlite
  run matrix_gemma   $B --rows 500 --model gemma3:27b
}

tier_lotus () {
  [ -x "$VL" ] || { echo "LOTUS interpreter $VL not found (see README)"; FAILED=$((FAILED + 1)); return; }
  local V="$VL"
  run exp12 experiments/exp12_lotus.py --n 300 --queries 6 --batch 8 --api-base $HOST
  V=${PYTHON:-.venv/bin/python}
  # Drop recorded exp18 outputs copied in by the cpu tier, or the supervisor
  # would consider both repetitions finished.
  for f in results/exp18_lotus_*; do
    r="reference_results/$(basename "$f")"
    [ -e "$r" ] && cmp -s "$f" "$r" && rm "$f"
  done
  if [ "$(uname)" = Darwin ]; then
    # Supervisor: four Ollama servers behind a round-robin proxy, both
    # repetitions, checkpoints after every query pair, memory guard.
    run exp18 experiments/exp18_memguard.py
    run exp18_analyze experiments/exp18_analyze.py --n 1000 --queries 47
  else
    echo "exp18: the memory-guarded supervisor is macOS-only; run the two"
    echo "repetitions by hand as described in README.md (section exp18)."
  fi
}

[ $# -eq 0 ] && set -- cpu
for t in "$@"; do
  case "$t" in
    cpu|llm|matrices|lotus) tier_$t ;;
    *) echo "unknown tier: $t (expected cpu, llm, matrices or lotus)"; exit 2 ;;
  esac
done
echo "=== done ($(date +%H:%M:%S)), $FAILED step(s) failed ==="
[ $FAILED -eq 0 ]
