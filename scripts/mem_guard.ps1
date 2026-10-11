# Memory guard: if free RAM drops below the limit, stop the envgen.py job (and
# its pool workers) so that llama-server and the real-LLM runs survive.
# envgen.py --resume keeps finished cells, so the stopped job can be restarted.
param([double]$LimitGB = 2.5)
Set-Location $PSScriptRoot
for ($i = 0; $i -lt 40; $i++) {         # wait up to 10 min for envgen.py to start
    if (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'envgen\.py' }) { break }
    Start-Sleep -Seconds 15
}
while ($true) {
    $envgen = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'envgen\.py' }
    if (-not $envgen) { break }
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($free -lt $LimitGB) {
        $ids = $envgen.ProcessId
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
            Where-Object { $ids -contains $_.ParentProcessId -or $ids -contains $_.ProcessId } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        "[{0}] free RAM {1:N1} GB < {2} GB: envgen stopped" -f (Get-Date), $free, $LimitGB |
            Out-File -Append -Encoding utf8 mem_guard.log
        break
    }
    Start-Sleep -Seconds 15
}
