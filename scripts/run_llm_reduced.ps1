# Reduced-scale real-LLM runs (CPU-only llama.cpp, Nemotron-Nano-9B-v2-Japanese Q8_0).
# Prerequisite: llama-server listening on 127.0.0.1:8080 with --jinja.
# Full scale for the manuscript: --seeds 5 --episodes 500 (Exp.4), --llm-seeds 5 --episodes 200 (Exp.6).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
python llm_experiment.py --arms A,B,D --seeds 2 --episodes 15 --lang en --out llm_results_real_ABD.json *> llm_run_ABD.log
python recovery_experiment.py --llm --backend llama --scenarios S1,S3,S6 --llm-seeds 1 --episodes 8 --lang en --out results_recovery_llm.json *> recovery_llm_run.log
