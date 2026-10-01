# SemReuse — reproduction artifact

SemReuse answers semantic-operator queries (natural-language predicates such as
"the document is about sports", normally evaluated with one LLM call per row)
by reusing the per-row results of earlier predicates. Every evaluated predicate
is cached as a natural-language *view*. For a new predicate, a two-tier
entailment reasoner (a calibrated NLI cross-encoder, plus an optional LLM
arbiter for the pairs it is unsure about) classifies its relation to the cached
ones (equivalence, subsumption, disjointness, overlap), and a rewrite evaluates
the predicate only on the reduced candidate set. A pooled stratified audit with
exact hypergeometric bounds then certifies per-query **recall and precision**
lower bounds at level 1−α, whatever mistakes the reasoner made. When it cannot
certify, it escalates toward cold evaluation.

This folder contains the code, the recorded LLM answers, the scripts that
regenerate every result file, and the reference results to compare against.
Everything runs locally and needs no paid API.

---

## 1. Contents

```
README.md                 this file
pyproject.toml            the semreuse package (pip install -e .)
requirements.txt          pinned versions of the main environment (Python 3.12)
requirements-lotus.txt    pinned versions of the LOTUS environment (exp12, exp18)
src/semreuse/             the system (module map in Section 9)
src/tests/                pytest suite (87 tests)
experiments/
  reproduce.sh            regenerates every result file, tier by tier
  compare_results.py      diffs results/ against reference_results/
  prepare_data.py         downloads the public datasets and models into data/
  common.py               shared harness (scales, seeds, NLI calibration, output naming)
  exp1_e2e.py ... exp18_*.py   the experiments (Section 6)
  build_llm_matrix.py     queries an Ollama model for every (predicate, row) pair
  rr_proxy.py             round-robin proxy over several Ollama servers (exp18)
  summary_numbers.py      headline numbers and certificate tally over all results
  make_figures.py, make_cost_figures.py    PDF plots from the result CSVs
data/
  llm_matrix/*.npz        recorded LLM answer matrices (predicate x row), replayed by the experiments
  llm_cache/*.sqlite      every recorded LLM answer with its raw output and token counts
reference_results/        the reference outputs (CSV / JSON / NPZ) that reproduce.sh regenerates
```

`results/` (created by the scripts) receives regenerated outputs and per-step
logs. `data/` also receives the public datasets and models when you run
`prepare_data.py`.

### Recorded LLM answers (`data/`)

Rebuilding these takes days of GPU time, so the artifact ships them. Every
experiment over a real LLM oracle replays them under unit-cost accounting.

| file | model (Ollama tag) | predicates × rows |
|---|---|---|
| `llm_matrix/frozen_analyst_20ng_2000_llama3.1-8b.npz` | `llama3.1:8b-instruct-q4_K_M` | 46 × 2,000 (frozen copy, the primary real-oracle matrix) |
| `llm_matrix/analyst_20ng_4000_llama3.1-8b-instruct-q4_K_M.npz` | `llama3.1:8b-instruct-q4_K_M` | 46 × 4,000 (columns 0–1,999 equal the frozen matrix) |
| `llm_matrix/analyst_20ng_4000_qwen2.5-14b-instruct-q4_K_M.npz` | `qwen2.5:14b-instruct-q4_K_M` | 46 × 2,000 |
| `llm_matrix/analyst_20ng_4000_mistral-small3.1-24b.npz` | `mistral-small3.1:24b` | 46 × 2,000 |
| `llm_matrix/analyst_20ng_4000_gemma3-27b.npz` | `gemma3:27b` | 46 × 500 |
| `llm_cache/20ng_4000_0.sqlite` | Llama-3.1-8B (184,000 answers) and Gemma-3-27B (23,000) | |
| `llm_cache/20ng_4000_0_qwen14b.sqlite` | Qwen2.5-14B (92,000 answers) | |
| `llm_cache/20ng_4000_0_mistral24b.sqlite` | Mistral-Small-3.1-24B (92,000 answers) | |

The rows are the first N documents of the seed-0, 4,000-document subsample of
20 Newsgroups. The predicates are the 46 distinct filters of the hand-written,
47-query analyst log in `src/semreuse/predicate_log.py`. The caches have one
table, `ans(model, pver, claim, row, answer, ptok, otok, raw)`, keyed by
(model, prompt version, predicate, row). For example, the token totals of the
2,000-row Llama matrix:

```bash
sqlite3 data/llm_cache/20ng_4000_0.sqlite \
  "SELECT COUNT(*), SUM(ptok), SUM(otok) FROM ans
   WHERE model='llama3.1:8b-instruct-q4_K_M' AND row < 2000;"
```

---

## 2. Requirements

| what | needed for | reference setup |
|---|---|---|
| Python 3.12 + `requirements.txt` | everything | Python 3.12.13, torch 2.13.0, sentence-transformers 6.0.0, numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.0 |
| ~1.5 GB disk for `data/` | the public datasets and models | — |
| a multi-core CPU (no GPU required) | the `cpu` tier | Apple M3 Ultra, 32 cores, 512 GB RAM, macOS 26.5; NLI model on MPS |
| [Ollama](https://ollama.com) + the four models above | the `llm` and `matrices` tiers | Ollama 0.34.4 / 0.35.0, all models Q4_K_M |
| Python 3.12 + `requirements-lotus.txt` | the `lotus` tier (exp12, exp18) | `lotus-ai` 1.2.4, `litellm` 1.98.0 |
| macOS (`vm_stat`, `footprint`) | only the exp18 memory-guarded supervisor | — |

The NLI cross-encoder runs on whatever device sentence-transformers selects
(CUDA, MPS or CPU).

---

## 3. Setup

```bash
cd Artifact
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/python experiments/prepare_data.py      # ~410 MB download
```

`prepare_data.py` caches, under `data/`:

- 20 Newsgroups through scikit-learn (`data/sklearn_20ng/`, ~14 MB);
- the AG News train split (`data/ag_news_train.csv`, ~29 MB);
- `cross-encoder/nli-deberta-v3-xsmall`, the NLI reasoner (`data/hf/`, ~275 MB);
- `sentence-transformers/all-MiniLM-L6-v2`, used by the embedding-cache baseline and the proxy cascade (`data/hf/`, ~90 MB).

RCV1-v2 (~0.5 GB) is downloaded through scikit-learn into `data/sklearn_rcv1/`
the first time `exp11_scale.py --corpus rcv1` runs. All scripts set
`HF_HOME=data/hf`, so nothing is written outside the folder.

**Optional, for the live-LLM tiers:**

```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull mistral-small3.1:24b
ollama pull gemma3:27b
```

**Optional, for the LOTUS tier.** `lotus-ai` 1.2.4 pins older numpy, pandas and
transformers than the rest of the project, so it gets its own environment. The
LOTUS scripts import `semreuse` from `src/` directly, so no install is needed
there.

```bash
python3.12 -m venv .venv-lotus
.venv-lotus/bin/pip install -r requirements-lotus.txt
```

---

## 4. Quick check (about 2 minutes)

```bash
.venv/bin/pytest src/tests -q                                     # 87 passed, ~40 s
.venv/bin/python experiments/exp1_e2e.py --scale smoke --corpus 20ng
```

The unit tests run offline. The NLI integration tests skip themselves if the
model is not yet in `data/hf`. The smoke run (400 rows, 12 queries) only checks
that the pipeline works: at that size savings are small by design, because the
audit's sample size does not shrink with N.

---

## 5. Reproducing the results

```bash
bash experiments/reproduce.sh                  # = the cpu tier
.venv/bin/python experiments/compare_results.py
```

`reproduce.sh` takes one or more tiers:

| tier | what it runs | needs | time (reference machine) |
|---|---|---|---|
| `cpu` (default) | exp1–9, 11, 14–17 (simulated oracles and replays of the recorded LLM answers), the exp18 analysis, the summary and the figures | main env + `prepare_data.py` | 1 h 47 min (measured at low priority on a loaded machine) |
| `llm` | exp13 (oracle determinism, four models) and exp10 (LLM arbiter) | Ollama + models | ~30 min |
| `matrices` | re-assembles `data/llm_matrix/` from `data/llm_cache/` | Ollama + models | minutes while the caches are complete (0 live calls) |
| `lotus` | exp12 (300 rows) and exp18 (1,000 rows × 47 queries × 2 repetitions) | LOTUS env + Ollama | exp12 ~25 min; exp18 ~15 h |

Every step writes `results/logs/<step>.log` and prints its summary lines. A
failed step is reported and the run continues. Set `PYTHON`, `LOTUS_PYTHON` or
`OLLAMA_URL` to override the interpreters or the Ollama endpoint
(default `http://localhost:11434`).

The `cpu` tier's analyses (exp17, the exp18 analysis, the summary) also read
outputs of the live-LLM steps. Unless those tiers have been run first, it copies
the recorded ones from `reference_results/` into `results/` (exp10, exp12, exp13
and the raw exp18 runs). For those files `compare_results.py` therefore reports
a trivial match.

`compare_results.py` compares every reference file that has been regenerated:
CSVs cell by cell (numbers to a relative tolerance of 1e-6), JSON value by
value, NPZ array by array. It lists wall-clock fields (any name containing
`wall`, `second`, `latency` or `clock`, or ending in `_s`) but never counts
them as differences. It exits non-zero if anything else differs. Use
`--verbose` to also see the timing fields, and `--results` / `--reference` to
compare other directories.

### What to expect

- **Simulated oracles** (20 Newsgroups, AG News, RCV1) and **replays of the
  recorded LLM answers** are deterministic given the seed (0 everywhere). The
  oracle is deterministic per (predicate, row, seed), including its noise
  model, and audit randomness is seeded. A full `cpu` run on the reference
  machine with the pinned versions reproduces 101 of the 109 reference files
  exactly, apart from wall-clock fields. The remaining 8 are listed under
  *Known differences* below.
- The NLI reasoner's scores can differ in the last digits across devices
  (CUDA / MPS / CPU) or library versions. A predicate pair that sits exactly on
  a decision threshold can then change class, which moves that query's call
  count slightly. Certificates are unaffected: they hold whatever the
  reasoner decides.
- **Live LLM calls** (`llm`, `lotus` tiers) are made at temperature 0, `top_k`
  1, fixed seed. exp13 re-issued 200 stored (predicate, row) pairs per model
  through a fresh, cache-less client and got 200/200 identical answers for
  each of the four models. LOTUS behind a batched endpoint is not fully
  deterministic: its own answers agree on 99.97% of tuples across the two
  exp18 repetitions. So exp12/exp18 numbers vary slightly between runs, and
  `results/exp13_*.csv` samples depend on the cache contents at the time.
- Wall-clock numbers depend on the machine and its load. The reference runs
  shared their machine with unrelated jobs (load average 20–180), so treat
  their timings as upper bounds. Call counts, token counts, accuracy and
  certificate counts are unaffected.

### Known differences

A few reference files were written before the last edits to `audit.py`,
`engine.py` and `slack.py`, and the current code does not reproduce them
exactly. `compare_results.py` reports these 8 files as `DIFF`:

| file(s) | what differs | current code | reference |
|---|---|---|---|
| `exp9_summary_20ng_full.csv`, `exp9_summary_20ng_full_novalidate.csv` and their `exp9_perquery_*` files | the `semreuse-gt+proxy` method (perfect entailment + proxy, 4 score bands) only; the other methods and the 1- and 2-band variants match | 327,501 calls (2.91×) | 336,119 calls (2.84×) |
| `exp9_sweep_summary_20ng_full.csv`, `exp9_sweep_perquery_20ng_full.csv` | the first of the 12 proxy configurations (pilot 200, gate 0.6) only | 693,733 calls, 94 certified queries | 684,770 calls, 98 certified queries |
| `exp14_summary_20ng_full_feat.csv` | one diagnostic column; all call counts and accuracies match | `slack_rate` (0.597 / 0.0033) | `slack_global_mean` (0.448 / 0.0016) |
| `summary_numbers.json` | follows from the exp9 rows above | 29,726 recall / 15,076 precision certificates; gt+proxy 2.91× | 29,730 / 15,080; 2.84× |

In every case the certificates still hold: no new violations, and the
qualitative conclusions are unchanged. Composing reuse with the proxy still
costs more than reuse alone (2.91× vs. 4.46× with perfect entailment).

### Plots

```bash
mkdir -p results && cp -n reference_results/* results/   # only if you have not regenerated results/
.venv/bin/python experiments/make_figures.py         # results/figures/*.pdf
.venv/bin/python experiments/make_cost_figures.py    # cost decomposition and scaling plots
```

The `cpu` tier already runs both on the regenerated results.

---

## 6. Experiments

Every source of randomness is seeded (`--seed`, default 0). The
simulated-oracle scripts also accept `--scale {smoke,small,medium,full}`
(corpus 400 / 1,500 / 4,000 / 8,000 rows; 12 / 30 / 60 / 120 queries) and
`--corpus {20ng,agnews,synthetic,rcv1}`.
They write `results/<exp>_<corpus>_<scale>[_<tag>].csv`. The exact command
behind every reference file is in `experiments/reproduce.sh`. Times are from
the reference machine.

| exp | question | reference outputs | tier | time |
|---|---|---|---|---|
| exp1 | End-to-end comparison of all methods (listed below), on 20NG and AG News; with 5% oracle noise; with 900 calibration pairs | `exp1_{summary,perquery}_*` | cpu | 1–3 min each |
| exp2 | Quality and calibration of the NLI reasoner: threshold vs. calibrated head, per-relation F1, reliability / ECE, pair throughput | `exp2_*_20ng_medium.csv` | cpu | ~1 min |
| exp3 | Audit under controlled entailment error: budget × error rate × target recall → empirical bound coverage | `exp3_audit_20ng_medium.csv` | cpu | ~1 min |
| exp4 | Savings vs. workload predicate overlap (0–0.95), and what the overlap knob generates | `exp4_overlap_*`, `exp4_workload_stats_*` | cpu | 3–12 min |
| exp5 | Rewrite-rule ablation (each rule cumulative and alone), with NLI and perfect entailment | `exp5_ablation_20ng_full.csv` | cpu | ~1 min |
| exp6 | Corpus-size scaling of the audit cost, N = 500 … 8,000 (the `_small` tag is the harness default, not the sweep size) | `exp6_scaling_20ng_small.csv` | cpu | ~1 min |
| exp7 | Sensitivity to the rewrite thresholds τ | `exp7_tau_{20ng,agnews}_full.csv` | cpu | ~1 min each |
| exp8 | End-to-end over real LLM oracles on the analyst log, replayed from `data/llm_matrix/`: Llama at N = 500/1,000/2,000 (and up to 4,000: tag `n4000`), relation slack ε = 0.10, Qwen, Mistral, Gemma | `exp8_{summary,perquery}_*`, `exp8_workload_*.json` | cpu | 1–10 min each |
| exp9 | The proxy-cascade competitor (SUPG-style) and its composition with reuse; a 12-configuration sweep; stratum-band and no-validation variants | `exp9_*` | cpu | 2–4 min each |
| exp10 | Tier-two LLM arbiter: pair accuracy and end-to-end effect | `exp10_{pairs,e2e}_20ng_full.csv` | llm | ~10 min |
| exp11 | Scale: AG News up to 120,000 rows, RCV1-v2 up to 804,414 documents (multi-label, 103 topics) | `exp11_{summary,perquery}_*` | cpu | ~1 min each, plus the RCV1 download |
| exp12 | SemReuse under the released LOTUS package, measured by LOTUS's own counters (300 rows, 6 queries) | `exp12_lotus_300_6.json` | lotus | ~25 min |
| exp13 | Is the temperature-0 oracle deterministic? 200 cached pairs re-issued without a cache | `exp13_determinism*` | llm | 3–7 min per model |
| exp14 | Measured agreement slack (`slack.py`): does declining rewrites the recall allowance cannot absorb pay off? Real and simulated oracles; plan-time features | `exp14_*` | cpu | 1–5 min each |
| exp15 | Audit-design ablation: pooled vs. per-stratum bounds, adaptive vs. fixed sample sizes, joint vs. recall-only certification | `exp15_*` | cpu | ~2 min each |
| exp16 | Certificate coverage and tightness: exact non-coverage of the bound primitives; 10.8M audits of controlled populations at the certification boundary; 1,000 re-audits of every real rewrite plan | `exp16_*` | cpu | primitives 15–40 min, synthetic 5–20 min, replays 1–4 min |
| exp17 | The analyst log on four LLM oracles side by side: self-consistency of declared implications, economics, cross-model agreement | `exp17_*` | cpu | <1 min |
| exp18 | The whole 47-query analyst log under LOTUS, N = 1,000, two repetitions, paired per-query latency; then economics, run-to-run agreement and certificates (`exp18_analyze.py`) | `exp18_*` | lotus (analysis: cpu) | ~15 h |
| summary | Headline numbers, plus every recall/precision certificate in every per-query file checked against realized values | `summary_numbers.json` | cpu | seconds |

### Methods in the output files

| method | meaning |
|---|---|
| `cold` | no reuse: the oracle on every row (LOTUS-style) |
| `exact` | exact (normalized) predicate-text cache |
| `embed@θ` | embedding-similarity cache (GPTCache-style), reused wholesale at cosine ≥ θ |
| `proxy` | per-query proxy cascade trained on the query's own pilot labels, audited with the same machinery |
| `semreuse` | entailment + rewrites + audit (the system) |
| `semreuse+proxy` | both, certified by a single audit |
| `semreuse-2tier` | with the LLM arbiter tier (exp10) |
| `semreuse-noaudit` | rewrites without the audit (cheap, no guarantee) |
| `semreuse-gt` | perfect entailment: an upper bound that isolates reasoner quality |

### Main columns

- `total_oracle_calls`: the cost unit, split into `candidate_calls`,
  `audit_calls` and `escalation_calls` where reported. NLI pair scorings are
  counted separately (`total_nli_scores`).
- `macro_precision`, `macro_recall`: realized accuracy against the oracle's
  answers.
- `recall_bound`, `precision_bound`: the published per-query certificates.
  `bound_violated` and `precision_bound_violated` flag a realized value below
  its bound. Under a noisy simulated oracle, `bound_violated_oracle` checks
  against the oracle's own semantics, which is what the certificate promises.
- `prompt_tokens`, `output_tokens`, `projected_dollars` (exp8): token
  accounting, priced at a reference hosted small model ($0.15 / $0.60 per
  million input / output tokens). This is a projection, not a bill.

---

## 7. Reference results

All values are in `reference_results/`. Reduction = cold calls ÷ method calls.
Recall target t = 0.9, α = 0.05.

| setting | cold calls | SemReuse | reduction | perfect entailment |
|---|---|---|---|---|
| 20 Newsgroups, N = 7,951, 120 queries (exp1) | 954,120 | 421,507 | 2.26× | 4.46× |
| AG News, N = 8,000 (exp1) | 960,000 | 65,467 | 14.66× | 29.95× |
| AG News, N = 120,000 (exp11) | 14,400,000 | 933,467 | 15.43× | 33.96× |
| RCV1-v2, N = 804,414 (exp11) | 96,529,680 | 40,101,457 | 2.41× | 3.03× |
| Llama-3.1-8B oracle, analyst log, N = 2,000 (exp8) | 94,000 | 88,538 | 1.06× | 1.12× |
| proxy cascade, best configuration, 20NG (exp9) | 954,120 | 589,705 (proxy) | 1.62× | — |
| under LOTUS, N = 300, 6 queries (exp12) | — | — | 1.00× | — |
| under LOTUS, N = 1,000, 47 queries, 2 runs (exp18) | 47,000 | 45,375 / 45,370 | 1.036× | — |

The real-oracle results:

- Of the 24 implications the analyst log declares between its predicates,
  exactly one holds exactly on Llama-3.1-8B's own answers at N = 2,000. The
  median one misses by 9.3% (`exp8_workload_20ng_full.json`, `exp17_*`). The
  binding constraint is the oracle's inconsistency with itself, so the audit
  escalates. Without the audit the same plans would reach 1.97×, at a
  realized recall of only 0.69.
- Other oracles at N = 2,000: Qwen2.5-14B 1.065×, Mistral-Small-3.1-24B 1.053×.
  Gemma-3-27B at N = 500: 1.026×.
- Measured agreement slack (exp14, real oracle) cuts escalation by 22% but
  total cost by under 1%.

Certificates over every per-query file (`summary_numbers.json`, `coverage`):
29,730 recall certificates with 3 violations, and 15,080 precision certificates
with 0 violations, against a nominal failure rate of 5%. Two of the violations
are in the proxy-configuration sweep and one is in the `recall-trust` design of
the audit ablation. Under LOTUS (exp18), 28 of 180 certificates fall short of
the same run's LOTUS-alone answers, each by at most 3 tuples. 26 of these are
bounds of exactly 1.0. The other 2 are the recall certificate of the same
query in both runs, one tuple short each (`exp18_analysis_1000_47.json`).

---

## 8. Live-LLM steps in detail

### Oracle configuration

`src/semreuse/llm_oracle.py` sends a LOTUS-style `sem_filter` prompt (version
`semfilter-v1`) to Ollama's `/api/generate`, one (predicate, row) pair per
call. Settings: `temperature 0`, `top_k 1`, `top_p 1`, `seed 0`,
`num_predict 3`, with documents truncated to 1,200 characters. Every answer is
memoized in the SQLite cache, so re-runs and audits see the identical answer.

### Rebuilding or re-querying the answer matrices

```bash
bash experiments/reproduce.sh matrices
.venv/bin/python experiments/compare_results.py \
    --results data/llm_matrix --reference data/llm_matrix.orig
```

This tier first copies the shipped matrices to `data/llm_matrix.orig/`. It
then re-assembles each matrix through its cache, which costs no live calls
while the cache is complete. To query a model afresh, point `--cache` at a new
file, for example:

```bash
.venv/bin/python experiments/build_llm_matrix.py --workload analyst --corpus 20ng \
    --n 4000 --rows 2000 --seed 0 --model llama3.1:8b-instruct-q4_K_M \
    --concurrency 12 --cache data/llm_cache/fresh_llama.sqlite
```

Builds are resumable: after an interruption only the missing pairs are
re-issued. `--order doc` asks all predicates of one document back to back, so
the server keeps the document's prompt prefix cached (same answers, faster).
`--row-start` and `--no-assemble` split one build across several servers that
share a cache. Recorded cost: 92,000 pairs per model at N = 2,000. Llama-3.1-8B
ran at 5.6 rows/s on an idle M3 Ultra (about 4 h per 2,000 rows). Mistral-24B
ran as two builders of about 9 h each on two 8-slot servers, and Gemma-27B took
5.4 h for 500 rows.
The frozen `frozen_analyst_20ng_2000_llama3.1-8b.npz` is a copy of the Llama
build at 2,000 rows. Later appends to the working file cannot change it.

### exp10 and exp13

```bash
bash experiments/reproduce.sh llm
```

exp10 sends its arbiter prompts to `http://localhost:11434` (the
`LLMArbiter` default). exp13 takes `--host`.

### exp18: LOTUS at scale

Reference configuration: `lotus-ai` 1.2.4 with model
`ollama/llama3.1:8b-instruct-q4_K_M` (LOTUS settings untouched,
`max_batch_size` 64), served by Ollama 0.35.0 as four instances on ports
11440–11443. Each instance runs with

```
OLLAMA_NUM_PARALLEL=8 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=600m OLLAMA_CONTEXT_LENGTH=4096
```

behind `experiments/rr_proxy.py --port 11500` (plain round robin). The clients
run with `OLLAMA_API_BASE=http://127.0.0.1:11500`, so LiteLLM's model-metadata
lookups go to the same stack, and with `HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`. The tuples are the first 1,000 of the seed-0
4,000-document 20NG subsample. The queries are the 47 filters of the analyst
log in issue order. SemReuse uses the calibrated NLI reasoner, τ = 0.5,
t = 0.9, α = 0.05, and a fresh store per repetition. Within a repetition, the
two arms answer each query back to back, in alternating order. The two
repetitions run as separate processes side by side. exp18 fetches Ollama's
model metadata once and serves it from memory, which avoids a LiteLLM stall on
repeated `/api/show` requests. LOTUS's prompts, batching and parsing are
untouched.

**macOS:** `experiments/exp18_memguard.py`, called by the `lotus` tier, owns
the whole stack: servers, proxy and both repetitions. It checkpoints after
every query pair (`results/exp18_state_*.pkl`, resumed with `--resume`). It
stops everything when the project's processes or the machine's used memory
cross a limit, and restarts slowly once memory has recovered. A repetition
whose log is silent for 20 minutes is restarted from its checkpoint. Its
defaults were set for a 512 GB machine, so lower them on smaller ones, e.g.
`--project-limit-gb 40 --system-limit-gb 56 --resume-below-gb 48`. Set
`OLLAMA_BIN` if the `ollama` binary is not at
`/Applications/Ollama.app/Contents/Resources/ollama`. Add `--detach` to run it
in the background (log: `results/logs/exp18_memguard.log`).

**Other platforms:** start the Ollama servers and the proxy yourself, then run
the two repetitions and the analysis:

```bash
for p in 11440 11441 11442 11443; do
  OLLAMA_HOST=127.0.0.1:$p OLLAMA_NUM_PARALLEL=8 OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=600m OLLAMA_CONTEXT_LENGTH=4096 ollama serve > results/logs/ollama_$p.log 2>&1 &
done
.venv/bin/python experiments/rr_proxy.py --port 11500 \
    --backends http://127.0.0.1:11440,http://127.0.0.1:11441,http://127.0.0.1:11442,http://127.0.0.1:11443 &
export OLLAMA_API_BASE=http://127.0.0.1:11500 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv-lotus/bin/python -u experiments/exp18_lotus_scale.py --n 1000 --repeats 2 --only-rep 0 \
    --api-base http://127.0.0.1:11500 > results/logs/exp18_lotus_1000.log 2>&1 & R0=$!
.venv-lotus/bin/python -u experiments/exp18_lotus_scale.py --n 1000 --repeats 2 --only-rep 1 \
    --out-tag _rep1 --api-base http://127.0.0.1:11500 > results/logs/exp18_lotus_1000_rep1.log 2>&1 & R1=$!
wait $R0 $R1
.venv/bin/python experiments/exp18_analyze.py --n 1000 --queries 47
```

Add `--resume` to continue an interrupted repetition from its checkpoint.
Delete any `results/exp18_lotus_*` files copied in by the `cpu` tier first.
On an idle GPU the four servers sustain about 7.2 calls/s. In the reference
run, three of the 94 query pairs were retried after transient errors.
`exp18_analyze.py` reports them, and reports tokens and latency over the
clean pairs.

---

## 9. Code map (`src/semreuse/`)

| module | role |
|---|---|
| `corpus.py` | corpora: 20 Newsgroups, AG News, RCV1-v2, synthetic taxonomies |
| `predicates.py` | NL predicates with ground-truth extensions; workload generation |
| `predicate_log.py` | the hand-written analyst predicate log; extensional (oracle-measured) relations with slack ε |
| `rcv1_predicates.py` | NL predicates over the RCV1 topic hierarchy; drill-down workloads |
| `oracle.py` | simulated unit-cost oracle, deterministic per (predicate, row, seed), with optional flip noise |
| `llm_oracle.py` | real LLM oracle over Ollama; answer matrices; token, $ and latency accounting |
| `store.py` | the predicate store: cached NL views with lineage |
| `entailment.py` | entailment reasoners: ground truth, noisy, NLI cross-encoder (calibrated on a synthetic taxonomy, never on evaluation data) |
| `arbiter.py` | tier two: LLM arbiter by binary decomposition, calibrated per class |
| `rewriter.py` | rewrite planning: superset pruning, positive union, disjoint elimination, multi-view combination |
| `audit.py` | pooled stratified audit; exact hypergeometric recall/precision bounds; escalation; `AuditConfig` switches used by exp15 |
| `slack.py` | measured agreement slack and the rewrite budget (off by default: `slack_budget=None`) |
| `engine.py` | the SemReuse engine: store + reasoner + rewriter + audit |
| `proxy.py` | proxy cascade, and its composition with reuse under one certificate |
| `baselines.py` | cold, exact-match and embedding-similarity cache engines |
| `integration.py` | `SemanticFilterService`: the seam for running SemReuse under a host system such as LOTUS |
| `metrics.py` | accuracy, certificate and cost accounting |

The bounds come from exact hypergeometric CDF inversion, with no normal
approximation. α is union-bounded across pools, assumed-positive strata and
the escalation chain.
