$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "_OrphanWatchdog.ps1")
. (Join-Path $PSScriptRoot "utils\opf_run_defaults.ps1")

$Python = Join-Path $PSScriptRoot ".venv_opf_adp\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$ResultsDir = if ($env:RESULTS_DIR) { $env:RESULTS_DIR } else { "Results/massive_opf_suite" }
$RunRoot = if ($env:RUN_ROOT) { $env:RUN_ROOT } else { Join-Path $ResultsDir "run_root" }
$SourceRunRoot = if ($env:SOURCE_RUN_ROOT) { $env:SOURCE_RUN_ROOT } else { Join-Path $ResultsDir "source_run_root" }
$HostRamStopAvailMib = [double]$env:HOST_RAM_STOP_AVAIL_MIB
$HostRamResumeAvailMib = [double]$env:HOST_RAM_RESUME_AVAIL_MIB
$GpuStopAvailMib = [double]$env:GPU_STOP_AVAIL_MIB
$GpuResumeAvailMib = [double]$env:GPU_RESUME_AVAIL_MIB

function Test-Truthy {
  param([string]$Value)
  if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
  $normalized = $Value.Trim().ToLowerInvariant()
  return @('1','true','yes','on','y') -contains $normalized
}

$EnableThermalGuard = if ($env:ENABLE_THERMAL_GUARD) { Test-Truthy $env:ENABLE_THERMAL_GUARD } else { $true }
$StopFile = Join-Path $PSScriptRoot "thermal_guard.stop"
$ThermalGuard = Join-Path $PSScriptRoot "thermal_guard_embedded.ps1"
$StrongAwakeGuard = Join-Path $PSScriptRoot "strong_awake_guard.ps1"
$PreviousScheme = $null
$Surface = if ($env:SURFACE) { $env:SURFACE } else { "all" }
$Tasks = if ($env:TASKS) { $env:TASKS } else { "all" }
$ParamBand = $env:PARAM_BAND
$LegacyArchitectureGrid = Test-Truthy $env:LEGACY_ARCHITECTURE_GRID
$IncludeSmallStl = Test-Truthy $env:INCLUDE_SMALL_STL
$IncludeVanilla = if ($null -eq $env:INCLUDE_VANILLA -or [string]::IsNullOrWhiteSpace($env:INCLUDE_VANILLA)) { $true } else { Test-Truthy $env:INCLUDE_VANILLA }
$MissingPresentTaskRepeats = if ($env:MISSING_PRESENT_TASK_REPEATS) { [int]$env:MISSING_PRESENT_TASK_REPEATS } else { 1 }
$PredictionRepeats = if ($env:PREDICTION_REPEATS) { [int]$env:PREDICTION_REPEATS } else { 1 }

if (Test-Path $StopFile) {
  Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
}

if (Get-Command powercfg -ErrorAction SilentlyContinue) {
  try {
    $previousSchemeLine = powercfg /getactivescheme
    $PreviousScheme = [regex]::Match($previousSchemeLine, "[0-9a-fA-F-]{36}").Value
  } catch {}
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " [OPF_ADP] MASSIVE EXPERIMENT SUITE INITIATED " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " [OPF_ADP] launch concurrency=unbounded-by-default host_stop=$HostRamStopAvailMib host_resume=$HostRamResumeAvailMib gpu_stop=$GpuStopAvailMib gpu_resume=$GpuResumeAvailMib" -ForegroundColor Cyan

$_awakeJob = $null
$_thermalProc = $null
$_watchdogJob = $null
$_launcherProc = $null

try {
  if (Test-Path $StrongAwakeGuard) {
    $_awakeJob = Start-Process powershell -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Minimized -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", "`"$StrongAwakeGuard`""
    )
  }
  if ($EnableThermalGuard -and (Test-Path $ThermalGuard)) {
    $_thermalProc = Start-Process powershell -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Minimized -ArgumentList @(
      "-NoProfile",
      "-ExecutionPolicy", "Bypass",
      "-File", "`"$ThermalGuard`"",
      "-StopFile", "`"$StopFile`""
    )
  }
  $LauncherArgs = @(
    "-u", "utils\opf_suite_launcher.py",
    "--mode", "massive",
    "--results-dir", $ResultsDir,
    "--run-root", $RunRoot,
    "--source-run-root", $SourceRunRoot,
    "--surface", $Surface,
    "--tasks", $Tasks
  )
  if (-not [string]::IsNullOrWhiteSpace($ParamBand)) {
    $LauncherArgs += "--param-band"
    $LauncherArgs += $ParamBand
  }
  if ($LegacyArchitectureGrid) { $LauncherArgs += "--legacy-architecture-grid" }
  if ($IncludeSmallStl) { $LauncherArgs += "--include-small-stl" }
  if ($IncludeVanilla) { $LauncherArgs += "--include-vanilla" } else { $LauncherArgs += "--no-include-vanilla" }
  $LauncherArgs += "--missing-present-task-repeats"
  $LauncherArgs += [string]$MissingPresentTaskRepeats
  $LauncherArgs += "--prediction-repeats"
  $LauncherArgs += [string]$PredictionRepeats
  $_launcherProc = Start-Process -FilePath $Python -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow -ArgumentList (@($LauncherArgs) + @args)
  $_watchdogJob = Start-MasterWatchdog -MasterPid $PID -PythonPid $_launcherProc.Id
  Wait-Process -Id $_launcherProc.Id
  if ($_launcherProc.ExitCode -ne 0) {
    throw "OPF launcher exited with code $($_launcherProc.ExitCode)"
  }
}
finally {
  if ($EnableThermalGuard -and $StopFile) {
    New-Item -ItemType File -Path $StopFile -Force | Out-Null
  }
  if ($_thermalProc -and -not $_thermalProc.HasExited) {
    Stop-Process -Id $_thermalProc.Id -Force -ErrorAction SilentlyContinue
  }
  if ($_awakeJob -and -not $_awakeJob.HasExited) {
    Stop-Process -Id $_awakeJob.Id -Force -ErrorAction SilentlyContinue
  }
  if ($_watchdogJob) {
    Stop-MasterWatchdog -WatchdogJob $_watchdogJob
  }
  if ($_launcherProc -and -not $_launcherProc.HasExited) {
    Stop-Process -Id $_launcherProc.Id -Force -ErrorAction SilentlyContinue
  }
  if ($PreviousScheme -and (Get-Command powercfg -ErrorAction SilentlyContinue)) {
    try { powercfg /s $PreviousScheme | Out-Null } catch {}
  }
  if (Test-Path $StopFile) {
    Remove-Item $StopFile -Force -ErrorAction SilentlyContinue
  }
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host " [OPF_ADP] MASSIVE EXPERIMENT SUITE COMPLETED " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
