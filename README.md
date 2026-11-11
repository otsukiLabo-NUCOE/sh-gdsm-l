# SH-GD/SM-L — code and results (v2.0.0, revised manuscript)

Reference implementation, experiment harnesses and result logs for

> A. Otsuki, "SH-GD/SM-L: Dynamic Alternative-Path Selection for AI Agents Based on Learned
> Schema Compatibility and State Monitoring," revised manuscript resubmitted to *IEEE Access*.

Every number, table and figure of the article is generated from the files in `results/`; nothing is
transcribed by hand. Version 2.0.0 accompanies the revised manuscript. Version 1.0.0 accompanied the
first submission and remains available under the same concept DOI.

## What is new in v2.0.0

* **Model.** A learned retry posterior and the hand-over to the language model are now part of the transition rule (Eqs. 18–21), plus a noise-aware posterior for imperfect validators and soft attribution for missing verdicts (`code/shgdsm.py`; the defaults reproduce v1).
* **Frontier language models.** Experiments 5, 6 and 8 rerun with Claude Sonnet 5.5 and Claude Opus 5.5 (`results/EL/`).
* **New experiments.** Retry under call budgets, baselines and ablation, degraded supervision, context-dependent compatibility, correlated failures, non-stationarity with alarm-triggered forgetting, look-ahead depth, scalability up to 483 tools, and decision time and reproducibility of the rule.

## Contents

| Path | What it is |
|---|---|
| `code/shgdsm.py` | The model: schema-pair relations, execution and retry reliability, noise-aware posterior, LCB planning, transition rule; exact DP and Monte Carlo evaluation |
| `code/envgen.py` | Randomly generated task universes (Experiment 7) |
| `code/llm_experiment.py` | Real JSON-Schema validators with language-model arms A/B/D (Experiment 5); backends llama.cpp, OpenAI-compatible, Anthropic |
| `code/recovery_experiment.py` | Fault scenarios S1–S6 with proxies or a real model, including C+H (hand-over) (Experiment 6) |
| `code/toolmaze_pilot.py` | Adapter for the external ToolMaze benchmark, arms `llm`, `hybrid`, `hybrid-h` (Experiment 8; benchmark not included) |
| `code/revision_experiments.py` | Degraded supervision and ablation (Experiment 9, Table 8) |
| `code/e_baselines.py` | Tool-level Bayesian bandit, similarity and exact-match baselines (Table 7) |
| `code/e_retry.py` | Retry under call budgets (Table 16) |
| `code/e_exp4_cr.py`, `code/e_exp4_cr_time.py` | C and C+R in the environment of Experiment 5; decision time and replay reproducibility (Tables 12, 23) |
| `code/e_context.py`, `code/e_corr.py` | Context-dependent compatibility and correlated provider outages (Experiment 9) |
| `code/e_drift.py` | Non-stationarity: drift, withdrawal, addition, compatibility change; forgetting and the over-claim alarm (Experiment 10) |
| `code/e_lookahead.py`, `code/e_scale.py` | Look-ahead depth and scalability (Experiment 11) |
| `code/make_rev_numbers.py`, `code/make_tex_numbers.py` | Generate the manuscript's numbers (`numbers_rev.tex`, `numbers.tex`) and tables |
| `code/make_figures_v4.py`, `code/make_fig_block.py` | Generate the figures |
| `code/make_highlight.py`, `code/reorder_refs.py`, `code/check_tex.py`, `code/tex2docx.py` | Highlighted-changes PDF, reference ordering, static LaTeX checks, LaTeX-to-Word conversion |
| `results/*.json` | Raw results of all experiments with seeds (`results_rev_*.json`: experiments added in the revision) |
| `results/EL/` | Runs with Claude Sonnet 5.5 and Opus 5.5 (per-seed and per-task results, API usage and cost) |
| `scripts/` | Launch scripts of the real-model runs (`resume_EL.sh`, `run_extra_EL.sh`; PowerShell scripts of v1) |

## Reproducing

Install the dependencies:

```bash
python -m pip install scipy matplotlib jsonschema anthropic
```

Experiments run from `code/` and write their result files into the current directory. Examples:

```bash
python revision_experiments.py --part EN      # degraded supervision
python e_drift.py                             # non-stationarity
python e_exp4_cr_time.py                      # decision time and reproducibility of C+R
```

The numbers of the manuscript are generated from `results/`:

```bash
cd results
python ../code/make_rev_numbers.py --out <manuscript directory>
```

### Real-model runs

These runs need an Anthropic API key in `ANTHROPIC_API_KEY` and are billed by the provider. `scripts/run_extra_EL.sh` and `scripts/resume_EL.sh` contain the exact command lines. For example:

```bash
python llm_experiment.py --backend anthropic --model claude-sonnet-5-5 --arms A,B,D --seed-list 1,2 --episodes 30 --out EL/exp4_sonnet_s1_2.json
python recovery_experiment.py --llm --backend anthropic --model claude-sonnet-5-5 --scenarios S4 --llm-arms C,A,B,C+H --llm-seeds 3 --episodes 12 --out EL/exp6_sonnet_S4.json
```

The local 9B model of v1 is still supported (`--backend llama`).

## ToolMaze (Experiment 8)

The ToolMaze benchmark (Zhu et al., arXiv:2606.05806) is **not redistributed** here.

* Code: https://github.com/Zhudongsheng75/ToolMaze
* Data: https://huggingface.co/datasets/dongsheng/ToolMaze

Place the code at `../IEEE_Access/external/ToolMaze_code/ToolMaze-main` and the data at `../IEEE_Access/external/ToolMaze_data`, relative to `code/`. Alternatively, edit `TM_ROOT` and `TM_DATA` at the top of `toolmaze_pilot.py`.

`results/EL/toolmaze_*.json` contain only task identifiers, outcomes, call counts and short judge messages.

## License

MIT (see `LICENSE`).
