# overnight_training.ps1 — ad-hoc overnight runner.
#
# Runs all three multi-label trainings sequentially with the optimal
# per-dataset hyperparameters documented in multilabel_training_runs.md
# (section 5.4). Captures stdout to a timestamped log file at repo root
# so partial progress survives a reboot.
#
# Usage (from any directory, with the API venv active):
#   .\scripts\overnight_training.ps1
#
# Notes:
#   - Run with the deep-tagger-api/.venv activated before launching.
#   - Disable Windows sleep/hibernate before leaving overnight, otherwise
#     the host will pause the Python process. (Settings → System →
#     Power → Sleep: Never, while plugged in.)
#   - $ErrorActionPreference is left at default; if one model errors the
#     remaining ones still run.

$ErrorActionPreference = "Continue"

# ---- paths ---------------------------------------------------------------
$repoRoot = Split-Path -Parent $PSScriptRoot
$apiDir   = Join-Path $repoRoot 'deep-tagger-api'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile  = Join-Path $repoRoot "overnight_training_$timestamp.log"

# ---- per-dataset optimal hyperparameters --------------------------------
# (epochs only differ; everything else stays at the script defaults)
$runs = @(
    @{ Type = 'tops';  Epochs = 10 },
    @{ Type = 'shoes'; Epochs = 15 },
    @{ Type = 'pants'; Epochs = 7  }
)

# ---- header -------------------------------------------------------------
$pythonVersion = & python --version 2>&1
$torchInfo     = & python -c "import torch; print(f'{torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" 2>&1

@"
================================================================
  Overnight multi-label training run
  Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
  Host:  $env:COMPUTERNAME
  User:  $env:USERNAME
  Python: $pythonVersion
  Torch:  $torchInfo
  Log:   $logFile
================================================================

"@ | Tee-Object -FilePath $logFile -Append | Out-Host

Set-Location $apiDir

# ---- runner -------------------------------------------------------------
$summary = @()

foreach ($run in $runs) {
    $type   = $run.Type
    $epochs = $run.Epochs

    $banner = @"
----------------------------------------------------------------
  $type — $epochs epochs
  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
----------------------------------------------------------------
"@
    $banner | Tee-Object -FilePath $logFile -Append | Out-Host

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    # `python -u` = unbuffered stdout so the log is written line-by-line
    # and survives a hard reboot mid-run.
    python -u deep_learning/train_multilabel.py `
        --product-type $type `
        --device cuda `
        --batch-size 128 `
        --num-workers 4 `
        --max-samples 0 `
        --epochs $epochs 2>&1 | Tee-Object -FilePath $logFile -Append

    $sw.Stop()
    $duration = $sw.Elapsed
    $hms = "{0:00}:{1:00}:{2:00}" -f $duration.Hours, $duration.Minutes, $duration.Seconds

    $footer = @"

[$type] Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
[$type] Duration: $hms ($([int]$duration.TotalSeconds) seconds)

"@
    $footer | Tee-Object -FilePath $logFile -Append | Out-Host

    $summary += [pscustomobject]@{
        Type     = $type
        Epochs   = $epochs
        Duration = $hms
        Seconds  = [int]$duration.TotalSeconds
    }
}

# ---- summary section ----------------------------------------------------
@"
================================================================
  SUMMARY
  End: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
================================================================

"@ | Tee-Object -FilePath $logFile -Append | Out-Host

# Pretty timing table
$summary | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $logFile -Append | Out-Host

# Extract the final per-model lines for quick grep when updating the doc.
"--- Per-dataset summary (grepped from log) ---" | Tee-Object -FilePath $logFile -Append | Out-Host
Select-String -Path $logFile -Pattern '^\[done\]|^\[ckpt\]|^\[epoch \d+\]|^\[split\]|^\[data\]' `
    | ForEach-Object { $_.Line } `
    | Tee-Object -FilePath $logFile -Append | Out-Host

"Log saved to: $logFile" | Tee-Object -FilePath $logFile -Append | Out-Host
