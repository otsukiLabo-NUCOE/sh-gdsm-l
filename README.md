# SH-GD/SM-L — code and results

Reference implementation, experiment harnesses and result logs for

> A. Otsuki, "SH-GD/SM-L: Dynamic Alternative-Path Selection for AI Agents Based on Learned
> Schema Compatibility and State Monitoring," submitted to *IEEE Access*.

Every number, table and figure of the article is generated from the files in `results/` by
`code/make_tex_numbers.py` and `code/make_figures_v4.py`; nothing is transcribed by hand.

## Contents

| Path | What it is |
|---|---|
| `code/shgdsm.py` | The model (learned schema-pair relation strengths, learned execution reliability, LCB planning, Eq. 1–19), exact DP evaluation with ECE/Brier, Monte Carlo evaluation, Experiments 1–3 and 5, sensitivity analyses |
| `code/envgen.py` | Experiment 7: randomly generated task universes (21–108 tasks) and the tool-withdrawal transfer test |
| `code/llm_experiment.py` | Experiment 4: real JSON-Schema validation, stub services inducing ρ*, arms A/B/C/D; backends llama.cpp, OpenAI-compatible, Claude |
| `code/recovery_experiment.py` | Experiment 6: six fault scenarios; decision-rule proxies or a real LLM (`--llm`) |
| `code/toolmaze_pilot.py` | Experiment 8: adapter for the external ToolMaze benchmark (not included, see below) |
| `code/make_tex_numbers.py`, `code/make_figures_v4.py` | Generate the manuscript's numbers, tables and figures from `results/` |
| `code/check_tex.py`, `code/tex2docx.py` | Static checks of the LaTeX manuscript; LaTeX-to-Word conversion |
| `results/*.json` | Raw results of all experiments (seeds included) |
| `scripts/*.ps1` | The reduced-scale real-LLM runs reported in the article (Windows PowerShell) |

## Reproducing (run from `code/`, results are written to the current directory)

```bash
python -m pip install scipy matplotlib jsonschema
python shgdsm.py --out results_v4.json
python envgen.py --sizes 4,8,16,24 --universes 8 --train-episodes 800 --eval-episodes 600 --workers 4 --out results_envgen.json
python llm_experiment.py --measure-rho 40000
python llm_experiment.py --arms C --seeds 5 --episodes 500 --out llm_results_armC_v4.json
python recovery_experiment.py --seeds 10 --episodes 400 --out results_recovery_v4.json
```

Real-LLM arms (local model served by llama.cpp with `--jinja`):

```bash
llama-server -m NVIDIA-Nemotron-Nano-9B-v2-Japanese-Q8_0.gguf --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 --jinja
python llm_experiment.py --arms A,B,D --seeds 2 --episodes 15 --out llm_results_real_ABD.json
python recovery_experiment.py --llm --scenarios S1,S3,S6 --llm-seeds 1 --episodes 8 --out results_recovery_llm.json
```

`--backend anthropic` (Claude) and `--backend openai --endpoint <url> --model <name>` run the same arms with other models.

## ToolMaze (Experiment 8)

The ToolMaze benchmark (Zhu et al., arXiv:2606.05806) is **not redistributed** here. Obtain the code from
https://github.com/Zhudongsheng75/ToolMaze and the data from https://huggingface.co/datasets/dongsheng/ToolMaze,
and place them at `../IEEE_Access/external/ToolMaze_code/ToolMaze-main` and `../IEEE_Access/external/ToolMaze_data`
relative to `code/` (or edit `TM_ROOT` / `TM_DATA` at the top of `toolmaze_pilot.py`).

## Licence

MIT (see `LICENSE`).
