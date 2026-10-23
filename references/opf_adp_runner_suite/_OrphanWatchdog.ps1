# _OrphanWatchdog.ps1
# Dot-source this file in PowerShell launchers, then call:
#   $watchdogJob = Start-MasterWatchdog -MasterPid $PID -PythonPid $pythonPid
# before launching the Python orchestrator.
#
# Behavior:
#   - Ctrl+C / PowerShell exit: stop the known Python orchestrator and its
#     descendants only.
#   - Terminal close / parent process death: watchdog job notices the master
#     PID vanished and performs the same cleanup.

function Start-MasterWatchdog {
    param(
        [Parameter(Mandatory)][int] $MasterPid,
        [Parameter(Mandatory)][int] $PythonPid,
        [int] $GraceSeconds = 15,
        [int] $PollSeconds = 3
    )

    $jobScript = {
        param($masterPid, $pythonPid, $graceSec, $pollSec)

        function Get-DescendantIds {
            param([int] $RootPid)
            $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
            $pending = [System.Collections.Generic.Queue[int]]::new()
            $seen = [System.Collections.Generic.HashSet[int]]::new()
            $pending.Enqueue($RootPid)
            while ($pending.Count -gt 0) {
                $parent = $pending.Dequeue()
                foreach ($process in $processes) {
                    if ([int]$process.ParentProcessId -ne $parent) { continue }
                    $child = [int]$process.ProcessId
                    if ($seen.Add($child)) { $pending.Enqueue($child) }
                }
            }
            return @($seen)
        }

        function Stop-ProcessTree {
            param([int] $RootPid)
            $descendants = @(Get-DescendantIds -RootPid $RootPid)
            foreach ($childPid in ($descendants | Sort-Object -Descending)) {
                Stop-Process -Id $childPid -Force -ErrorAction SilentlyContinue
            }
            Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
        }

        Start-Sleep -Seconds 3

        while ($true) {
            Start-Sleep -Seconds $pollSec

            $masterAlive = $false
            try {
                $p = Get-Process -Id $masterPid -ErrorAction Stop
                $masterAlive = (-not $p.HasExited)
            } catch {
                $masterAlive = $false
            }

            if (-not $masterAlive) {
                $stamp = Get-Date -Format 'HH:mm:ss'
                Write-Host "[$stamp][WATCHDOG] Master PID $masterPid gone. Stopping owned Python PID $pythonPid..."
                Stop-Process -Id $pythonPid -ErrorAction SilentlyContinue
                Start-Sleep -Seconds $graceSec
                Stop-ProcessTree -RootPid $pythonPid

                $stamp = Get-Date -Format 'HH:mm:ss'
                Write-Host "[$stamp][WATCHDOG] Orphan cleanup complete."
                break
            }
        }
    }

    $job = Start-Job -ScriptBlock $jobScript -ArgumentList $MasterPid, $PythonPid, $GraceSeconds, $PollSeconds

    $sourceIdentifier = "OPF_ADP.PowerShell.Exiting.$PythonPid"
    $null = Register-EngineEvent -SourceIdentifier $sourceIdentifier -MessageData $PythonPid -Action {
        Stop-Process -Id ([int]$event.MessageData) -Force -ErrorAction SilentlyContinue
    }
    $job | Add-Member -NotePropertyName SourceIdentifier -NotePropertyValue $sourceIdentifier -Force

    return $job
}

function Stop-MasterWatchdog {
    param(
        [Parameter(Mandatory)] $WatchdogJob
    )

    try { Stop-Job -Job $WatchdogJob -ErrorAction SilentlyContinue } catch {}
    try { Remove-Job -Job $WatchdogJob -Force -ErrorAction SilentlyContinue } catch {}
    try {
        if ($WatchdogJob.SourceIdentifier) {
            Unregister-Event -SourceIdentifier $WatchdogJob.SourceIdentifier -ErrorAction SilentlyContinue
        }
    } catch {}
}
