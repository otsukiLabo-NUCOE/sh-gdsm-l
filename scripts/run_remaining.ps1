# Remaining real-LLM runs after the out-of-memory stop of 2026-10-10:
# Experiment 4 arm D, Experiment 6 (real LLM), then the ToolMaze pilot.
# Prerequisite: llama-server listening on 127.0.0.1:8080 with --jinja.
Set-Location $PSScriptRoot
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 http://127.0.0.1:8080/health | Out-Null; $ok = $true; break }
    catch { Start-Sleep -Seconds 10 }
}
if (-not $ok) { "llama-server not reachable" | Out-File -Encoding utf8 run_remaining.log; exit 1 }
python llm_experiment.py --arms D --seeds 2 --episodes 15 --lang en --out llm_results_real_D.json *> llm_run_D.log
python recovery_experiment.py --llm --backend llama --scenarios S1,S3,S6 --llm-seeds 1 --episodes 8 --lang en --out results_recovery_llm.json *> recovery_llm_run.log
python -u toolmaze_pilot.py --categories c3,c4 --modes P2 --tasks-per-cell 12 --out results_toolmaze_pilot.json *> toolmaze_pilot.log
