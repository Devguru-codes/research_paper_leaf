# Start ONE extra detached queue on the second Kaggle account, without touching queues already running.
# Usage, from the project folder:
#   powershell -ExecutionPolicy Bypass -File kaggle\run_alt_queue.ps1 main_norot_s1 main_norot_s2
# Kaggle runs at most 2 GPU notebooks at once, so start this only when at most one other queue is busy.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
if ($args.Count -eq 0) { Write-Host "give the batch names, e.g. main_norot_s1 main_norot_s2"; exit 1 }
$env:PYTHONUTF8 = "1"
$env:MEDLEAF_KAGGLE_USER = "devgurucodes"
$env:KAGGLE_API_TOKEN = (Get-Content "$env:USERPROFILE\.kaggle_alt\access_token" -Raw).Trim()
$tag = $args[0]
$p = Start-Process -FilePath python -ArgumentList (@("kaggle\kqueue.py") + $args) -WorkingDirectory $PWD -WindowStyle Hidden `
    -RedirectStandardOutput "results\kaggle\kqueue_$tag`_stdout.txt" -RedirectStandardError "results\kaggle\kqueue_$tag`_stderr.txt" -PassThru
$p.Id | Out-File "results\kaggle\kqueue_$tag.pid" -Encoding ascii
Write-Host "queue started (pid $($p.Id)): $($args -join ', '); progress: results\kaggle\queue_log_devgurucodes.txt"
