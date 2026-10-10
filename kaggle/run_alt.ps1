# Run the remaining MedLeaf experiments on the second Kaggle account (devgurucodes).
# Usage, from the project folder in your own PowerShell:
#   powershell -ExecutionPolicy Bypass -File kaggle\run_alt.ps1
# Step 1 uploads two PRIVATE datasets (once); step 2 starts a detached queue that survives closing the window.
# Continue (not Stop): Windows PowerShell turns any stderr line of a native command into a terminating
# error under Stop, and `kaggle datasets status` on a not-yet-existing dataset answers 403 on stderr.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONUTF8 = "1"
$env:MEDLEAF_KAGGLE_USER = "devgurucodes"
$env:KAGGLE_API_TOKEN = (Get-Content "$env:USERPROFILE\.kaggle_alt\access_token" -Raw).Trim()

foreach ($ds in "medleaf-esa-cache", "medleaf-esa-maskhead") {
    $s = python -m kaggle datasets status "devgurucodes/$ds" 2>&1 | Out-String
    if ($s -match "ready") { Write-Host "$ds already uploaded"; continue }
    $cmd = if ($ds -eq "medleaf-esa-cache") { "data-create" } else { "dimpsar-create" }
    Write-Host "uploading $ds (403 above just means it does not exist yet) ..."
    python kaggle\launch.py $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "upload of $ds failed - send this output to Claude"; exit 1 }
    do { Start-Sleep 30; $s = python -m kaggle datasets status "devgurucodes/$ds" 2>&1 | Out-String; Write-Host "  $ds : $($s.Trim())" } until ($s -match "ready")
}

# Two queues = two concurrent GPU notebooks (Kaggle allows 2). Each queue runs its kernels one after another;
# a queue adopts a kernel that is already running instead of re-pushing it, and skips finished+pulled ones.
foreach ($f in Get-ChildItem results\kaggle\kqueue_alt*.pid -ErrorAction SilentlyContinue) {
    Stop-Process -Id (Get-Content $f) -ErrorAction SilentlyContinue  # stop older queue processes (not the kernels)
}
$queues = @(("main_vit", "main_ablations", "main_p1"), ("main_bioclip_a", "main_bioclip_b", "main_norot"))
for ($i = 0; $i -lt $queues.Count; $i++) {
    $p = Start-Process -FilePath python -ArgumentList (@("kaggle\kqueue.py") + $queues[$i]) -WorkingDirectory $PWD -WindowStyle Hidden `
        -RedirectStandardOutput "results\kaggle\kqueue_alt$i`_stdout.txt" -RedirectStandardError "results\kaggle\kqueue_alt$i`_stderr.txt" -PassThru
    $p.Id | Out-File "results\kaggle\kqueue_alt$i.pid" -Encoding ascii
    Write-Host "alt queue $i started (pid $($p.Id)): $($queues[$i] -join ', ')"
    Start-Sleep 120  # let the first queue adopt/push before the second pushes
}
Write-Host "progress: results\kaggle\queue_log_devgurucodes.txt"
