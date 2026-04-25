# overnight_training.ps1 — ad-hoc overnight runner.
#
# Runs the multi-label trainings sequentially with the optimal per-dataset
# hyperparameters (see deep-tagger-api/notebook/multilabel_training_runs.ipynb
# for rationale). Captures stdout to a timestamped log file at repo root so
# partial progress survives a reboot.
#
# Usage (from any directory, with the API venv active):
#   .\scripts\overnight_training.ps1                        # los 3 (default)
#   .\scripts\overnight_training.ps1 -Types tops,shoes      # sólo esos dos
#   .\scripts\overnight_training.ps1 -Types pants           # sólo pants
#
# Notes:
#   - Run with the deep-tagger-api/.venv activated before launching.
#   - Disable Windows sleep/hibernate before leaving overnight, otherwise
#     the host will pause the Python process. (Settings → System →
#     Power → Sleep: Never, while plugged in.)
#   - $ErrorActionPreference is left at default; if one model errors the
#     remaining ones still run.

param(
    [ValidateSet('tops', 'shoes', 'pants')]
    [string[]]$Types = @('tops', 'shoes', 'pants'),
    [switch]$PosWeight
)

$ErrorActionPreference = "Continue"

# ---- paths ---------------------------------------------------------------
$repoRoot = Split-Path -Parent $PSScriptRoot
$apiDir   = Join-Path $repoRoot 'deep-tagger-api'
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile  = Join-Path $repoRoot "overnight_training_$timestamp.log"

# ---- per-dataset optimal hyperparameters --------------------------------
# (epochs only differ; everything else stays at the script defaults)
$allRuns = @(
    @{ Type = 'tops';  Epochs = 10 },
    @{ Type = 'shoes'; Epochs = 15 },
    @{ Type = 'pants'; Epochs = 15 }
)

# Filter to only the requested types, preserving the canonical order above.
$runs = $allRuns | Where-Object { $Types -contains $_.Type }
if ($runs.Count -eq 0) {
    Write-Error "Ningún tipo válido en -Types. Permitidos: tops, shoes, pants"
    exit 1
}

# ---- header -------------------------------------------------------------
$pythonVersion = & python --version 2>&1
$torchProbe = @'
import torch
dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
print(f'{torch.__version__}  cuda={torch.cuda.is_available()}  device={dev}')
'@
$torchInfo = $torchProbe | & python - 2>&1

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
    $extraArgs = @()
    if ($PosWeight) { $extraArgs += '--pos-weight' }

    python -u deep_learning/train_multilabel.py `
        --product-type $type `
        --device cuda `
        --batch-size 128 `
        --num-workers 4 `
        --max-samples 0 `
        --epochs $epochs `
        @extraArgs 2>&1 | Tee-Object -FilePath $logFile -Append

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
# Read the file fully into memory FIRST and filter from the in-memory copy
# instead of using Select-String -Path. The previous version piped
# Select-String -Path back into Tee-Object -Append on the same file, which
# created a feedback loop: each line written by Tee got re-read by
# Select-String, re-emitted, re-appended... ad infinitum.
$logSnapshot = Get-Content -LiteralPath $logFile
$summaryLines = @(
    "--- Per-dataset summary (grepped from log) ---"
    $logSnapshot | Where-Object { $_ -match '^\[done\]|^\[ckpt\]|^\[epoch \d+\]|^\[split\]|^\[data\]' }
    "Log saved to: $logFile"
)
$summaryLines | ForEach-Object {
    $_ | Out-Host
    Add-Content -LiteralPath $logFile -Value $_
}
