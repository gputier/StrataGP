# Helpers shared by measure_partition.ps1 and bench71_server.ps1 (Windows PowerShell 5.1). Dot-sourced by each script:
#   . (Join-Path $PSScriptRoot 'ornith_server_common.ps1')
# No param block and no top-level side effect: it only defines functions. Assert-PreFlight and Assert-PortFree read $Port,
# $MinFreeGiB and $MaxIdleGpuMiB from the calling script's scope (they are parameters of both scripts).

function Assert-File([string]$Path, [string]$What) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$What not found: $Path" }
}

function Assert-NoDoubleQuote($Params) {
    # A path parameter that holds a double quote is refused: the server's arguments are quoted for Start-Process, and a
    # double quote inside one would end the quoting and add arguments. $Params: name -> value of each path parameter.
    foreach ($name in @($Params.Keys)) {
        if ([string]$Params[$name] -match '"') { throw "-$name holds a double quote, which is refused: $($Params[$name])" }
    }
}

function ConvertTo-ProcessArg([string]$Value) {
    # One argument of a Windows command line for Start-Process: Windows PowerShell 5.1 joins -ArgumentList with spaces and
    # quotes nothing, so the value is wrapped in double quotes here, a trailing run of backslashes doubled (it would escape
    # the closing quote). A value that holds a double quote is refused (the path parameters are checked for it at the start).
    if ($Value.Contains('"')) { throw "a process argument holds a double quote: $Value" }
    return '"' + ($Value -replace '(\\+)$', '$1$1') + '"'
}

function Get-GpuComputeApps {
    # The compute processes nvidia-smi lists (the lines, for the log, and their PIDs). Under WDDM it lists desktop programs
    # (dwm.exe, explorer.exe, ...) and reads [N/A] for the used_memory of every process, so used_memory decides nothing.
    $ErrorActionPreference = 'Continue'   # (stderr of a native command must not throw here; the exit code is checked)
    $out = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed (exit $LASTEXITCODE): $($out -join ' ')" }
    $lines = New-Object System.Collections.Generic.List[string]
    $pids = New-Object System.Collections.Generic.List[int]
    foreach ($o in $out) {
        $line = "$o".Trim()
        if ($line -eq '' -or $line -match 'No running') { continue }
        $lines.Add($line)
        $f = $line -split ','
        if ($f[0].Trim() -match '^\d+$') { $pids.Add([int]$f[0].Trim()) }
    }
    return [pscustomobject]@{ Lines = $lines.ToArray(); Pids = $pids.ToArray() }
}

function Get-GpuUsedMiB {
    # The memory in use on the whole GPU (MiB, nvidia-smi memory.used; the largest figure when there are several GPUs). Under
    # WDDM it counts the desktop's share too (963 MiB at idle on the RTX 5090). Throws when there is no number.
    $ErrorActionPreference = 'Continue'   # (stderr of a native command must not throw here; the exit code is checked)
    $out = @(& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "nvidia-smi failed (exit $LASTEXITCODE): $($out -join ' ')" }
    $max = [long]-1
    foreach ($o in $out) {
        $line = "$o".Trim()
        if ($line -eq '') { continue }
        if ($line -notmatch '^\d+$') { throw "nvidia-smi memory.used is not a number: $line" }
        if ([long]$line -gt $max) { $max = [long]$line }
    }
    if ($max -lt 0) { throw 'nvidia-smi gave no memory.used figure' }
    return $max
}

function Find-EngineProcs([string]$ExePath) {
    # One Win32_Process snapshot, filtered in memory. An engine is a process named strata.exe, strata-q36.exe or
    # llama-server.exe, whatever its path. Named: those processes ('name PID n'). ByPath: the PIDs of the processes running
    # from $ExePath (ExecutablePath, case-insensitive; none when $ExePath is empty), whatever their name.
    $engineNames = @('strata.exe', 'strata-q36.exe', 'llama-server.exe')   # (-contains is case-insensitive)
    $rows = @(Get-CimInstance -ClassName Win32_Process -Property ProcessId, Name, ExecutablePath)
    $named = New-Object System.Collections.Generic.List[string]
    $byPath = New-Object System.Collections.Generic.List[int]
    foreach ($r in $rows) {
        if ($engineNames -contains [string]$r.Name) { $named.Add("$($r.Name) PID $($r.ProcessId)") }
        if ($ExePath -ne '' -and $r.ExecutablePath -and [string]::Equals([string]$r.ExecutablePath, $ExePath, [StringComparison]::OrdinalIgnoreCase)) {
            $byPath.Add([int]$r.ProcessId)
        }
    }
    return [pscustomobject]@{ Named = $named.ToArray(); ByPath = $byPath.ToArray() }
}

function Get-GpuEngineProcs {
    # Every engine process, whatever its path ('name PID n'); see Find-EngineProcs.
    return , @((Find-EngineProcs '').Named)
}

function Assert-PortFree {
    # A server that already answers on the port (an earlier run's, say) would be measured instead of the new one.
    $busy = $false
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null; $busy = $true } catch { $busy = $false }
    if ($busy) { throw "port $Port already answers /health: another server runs; stop it first" }
}

function Assert-PreFlight([string]$ExePath) {
    # Throws before anything is started. Get-VM must answer (it needs an elevated shell and the Hyper-V module). A runner VM
    # (gitlab-runner*) must be Off.
    try { $vms = @(Get-VM -ErrorAction Stop) }
    catch { throw "pre-flight: Get-VM failed ($($_.Exception.Message)); run the script from an elevated shell with the Hyper-V module" }
    $notOff = @($vms | Where-Object { $_.Name -like 'gitlab-runner*' -and "$($_.State)" -ne 'Off' })
    if ($notOff.Count -gt 0) { throw "pre-flight: runner VM not Off: $(($notOff | ForEach-Object { $_.Name + ' (' + $_.State + ')' }) -join ', '); stop it first" }
    $freeKiB = [long](Get-CimInstance -ClassName Win32_OperatingSystem).FreePhysicalMemory
    $freeGiB = $freeKiB / 1024.0 / 1024.0
    if ($freeGiB -lt $MinFreeGiB) { throw ("pre-flight: {0:F1} GiB of free physical memory, below -MinFreeGiB {1}" -f $freeGiB, $MinFreeGiB) }
    $eng = Find-EngineProcs $ExePath   # one snapshot for the name scan and the path scan
    if ($eng.Named.Count -gt 0) { throw "pre-flight: an engine process runs: $($eng.Named -join '; '); stop it first" }
    $usedMiB = Get-GpuUsedMiB
    if ($usedMiB -gt $MaxIdleGpuMiB) { throw "pre-flight: the GPU has $usedMiB MiB in use (memory.used), above -MaxIdleGpuMiB $MaxIdleGpuMiB; free it first" }
    $gpu = Get-GpuComputeApps
    if ($gpu.Lines.Count -gt 0) { Write-Host "pre-flight: compute processes listed (information only, not counted): $($gpu.Lines -join '; ')" }
    if ($eng.ByPath.Count -gt 0) { throw "pre-flight: a process already runs from ${ExePath}: PID $($eng.ByPath -join ','); stop it first" }
    Assert-PortFree
    Write-Host ("pre-flight ok: {0:F1} GiB free, runner VMs Off, no engine running, GPU {1} MiB in use (at most {2})" -f $freeGiB, $usedMiB, $MaxIdleGpuMiB)
}

function Get-ServerProcs([int]$Root, $RootStart, [string]$ExePath) {
    # One Win32_Process snapshot, two lists. Tree: the server's processes by parent PID only, deepest first. The root seeds
    # the walk only when it is alive with a CreationDate that matches $RootStart, the StartTime of the process Start-Process
    # returned (within 1 ms: WMI keeps microseconds, the process object 100 ns), so a reused PID is not taken; a child is taken
    # only when created at or after its parent, so the older processes of a reused parent PID are not taken either. A child
    # whose parent has died is not found (its parent is not in the snapshot). Strays: the processes running from the engine
    # binary's path that are not in the tree; this script never stops them, it reports them.
    $rows = @(Get-CimInstance -ClassName Win32_Process -Property ProcessId, ParentProcessId, ExecutablePath, CreationDate)
    $tree = New-Object System.Collections.Generic.List[int]
    $born = New-Object System.Collections.Generic.List[datetime]
    if ($null -ne $RootStart) {
        foreach ($r in $rows) {
            if ([int]$r.ProcessId -eq $Root -and $null -ne $r.CreationDate -and
                [math]::Abs(($r.CreationDate - [datetime]$RootStart).TotalMilliseconds) -le 1.0) {
                $tree.Add($Root)
                $born.Add($r.CreationDate)
                break
            }
        }
    }
    for ($i = 0; $i -lt $tree.Count; $i++) {
        foreach ($r in $rows) {
            $id = [int]$r.ProcessId
            if ([int]$r.ParentProcessId -eq $tree[$i] -and -not $tree.Contains($id) -and $null -ne $r.CreationDate -and
                $r.CreationDate -ge $born[$i]) {
                $tree.Add($id)
                $born.Add($r.CreationDate)
            }
        }
    }
    $deep = New-Object System.Collections.Generic.List[int]
    $deepBorn = New-Object System.Collections.Generic.List[datetime]
    for ($i = $tree.Count - 1; $i -ge 0; $i--) { $deep.Add($tree[$i]); $deepBorn.Add($born[$i]) }
    $strays = New-Object System.Collections.Generic.List[int]
    foreach ($r in $rows) {
        $id = [int]$r.ProcessId
        if ($r.ExecutablePath -and [string]::Equals([string]$r.ExecutablePath, $ExePath, [StringComparison]::OrdinalIgnoreCase) -and
            -not $tree.Contains($id)) { $strays.Add($id) }
    }
    return [pscustomobject]@{ Tree = $deep.ToArray(); Born = $deepBorn.ToArray(); Strays = $strays.ToArray() }
}

function Stop-ServerTree([int]$Root, $RootStart, [string]$ExePath) {
    # Stops the server's process tree (Get-ServerProcs), waits for each to exit, then takes the snapshot again (a child started
    # meanwhile is caught) until a snapshot finds none; that empty snapshot is also the final check. Throws when one is still
    # alive, or when a process runs from the engine binary's path outside the tree: it is reported by PID and not stopped (the
    # next pre-flight refuses to start while it runs). Returns the PIDs stopped.
    # Without a start time ($RootStart null: StartTime could not be read) the root cannot be verified, so no tree is built. The
    # fail-safe is to throw with the PID while a process of that Id is alive, so the operator stops it by hand: rooting the tree
    # at the bare PID would kill whatever holds that Id if it was reused, and a silent empty stop would leave the server running.
    if ($null -eq $RootStart) {
        $alive = Get-Process -Id $Root -ErrorAction SilentlyContinue
        $running = $false
        if ($alive) { try { $running = -not $alive.HasExited } catch { $running = $true } }
        if ($running) { throw "the server's start time is unknown, so its process tree cannot be verified and was not stopped: PID $Root is still alive; stop it and its children by hand before another launch" }
    }
    $stopped = New-Object System.Collections.Generic.List[int]
    $left = $null   # the last snapshot, when it found no tree: the final check
    for ($pass = 1; $pass -le 5; $pass++) {
        $snap = Get-ServerProcs $Root $RootStart $ExePath
        $ids = @($snap.Tree)
        if ($ids.Count -eq 0) { $left = $snap; break }
        $procs = @()
        for ($j = 0; $j -lt $ids.Count; $j++) {
            $id = $ids[$j]
            $p = Get-Process -Id $id -ErrorAction SilentlyContinue
            if (-not $p) { continue }   # exited since the snapshot
            # the PID must still be the process of the snapshot (same start time), not a process that reused it since
            $same = $false
            try { $same = [math]::Abs(($p.StartTime - $snap.Born[$j]).TotalMilliseconds) -le 1.0 } catch { $same = $false }
            if (-not $same) { continue }
            $procs += $p
            if (-not $stopped.Contains($id)) { $stopped.Add($id) }
            try { $p.Kill() }
            catch { if (-not $p.HasExited) { Write-Warning "PID ${id}: stop failed: $($_.Exception.Message)" } }
        }
        foreach ($p in $procs) {
            try { if (-not $p.WaitForExit(30000)) { Write-Warning "PID $($p.Id) has not exited after 30 s" } }
            catch { Write-Warning "PID $($p.Id): wait failed: $($_.Exception.Message)" }
        }
    }
    if ($null -eq $left) { $left = Get-ServerProcs $Root $RootStart $ExePath }   # 5 passes all stopped something: a snapshot after the last
    if ($left.Tree.Count -gt 0) { throw "server processes still alive after the stop: PID $($left.Tree -join ','); stop them by hand before another launch" }
    if ($left.Strays.Count -gt 0) {
        throw ("processes running from ${ExePath} outside the server's process tree, not stopped: PID $($left.Strays -join ','); " +
               "check what they are and stop them by hand before another launch (stopped from the tree: PID $($stopped -join ','))")
    }
    return , $stopped.ToArray()
}
