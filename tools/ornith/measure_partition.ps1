<#
.SYNOPSIS
Measures where the q36 engine spends a decode token (Ornith-1.5-122B, hybrid mode), over several cold launches (issue #95).

.DESCRIPTION
For each launch: a pre-flight check that throws before anything starts (a running Hyper-V VM named gitlab-runner*, free
physical memory below -MinFreeGiB, a process named strata.exe or llama-server.exe running from any path, the whole GPU's
memory.used by nvidia-smi above -MaxIdleGpuMiB (per-process used_memory reads [N/A] under WDDM and decides nothing), a process
already running from the engine binary's path, or a server already answering on the port). Then a passive sampler (free RAM and
VRAM every 2.5 s into mem.log; it never stops anything: the engine runs with no memory guard), serve/server.py with the strata
engine, STRATA_Q36_STATS=1 and no working-set cap, a wait for /health, one throw-away warm-up request of -WarmupTokens tokens
(240 or more, so the automatic PCIe share has made its last update, after 48, 144 and 240 decode windows), then one greedy
streamed chat request of -Tokens tokens per prompt, each text chunk timestamped on arrival (chunks.tsv). A completion with
fewer tokens than asked is flagged (usage.completion_tokens, finish_reason) in response.json and by the summary. The PCIe
share the engine reports after each request is logged. In a finally block the server's processes are stopped (its process
tree by parent PID and any process running from the engine binary's path, the snapshot taken again until none is left),
each is waited for, and the script throws when one is still alive or when, after up to 60 s, a strata.exe or llama-server.exe
still runs, the GPU's memory.used is still above -MaxIdleGpuMiB or a stopped PID is still listed as a compute process.
Afterwards it runs partition_summary.py on the output folder.

The Hyper-V VMs are only listed with Get-VM (name, state, CPU usage, memory, uptime), never changed; Get-VM needs an elevated
shell, which the pre-flight checks. Inputs (binary, build folder, GGUF, expert profile) are only read. The output folder must
not exist yet or must hold no launch_* folder.

.EXAMPLE
pwsh -File tools\ornith\measure_partition.ps1 -Binary C:\b\strata.exe -BuildDir C:\b -Gguf D:\m\Ornith-1.5-122B-IQ3XXS-IQ4XS.gguf -RankFile D:\m\profile.strp -OutDir D:\runs\issue95
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Binary,                # strata.exe of the build (read only)
    [Parameter(Mandatory = $true)][string]$BuildDir,              # the build folder; its DLLs go on the engine's PATH
    [Parameter(Mandatory = $true)][string]$Gguf,                  # the model file (read only)
    [Parameter(Mandatory = $true)][string]$RankFile,              # expert profile read at start, --expert-profile (read only)
    [Parameter(Mandatory = $true)][string]$OutDir,                # new folder for the logs of this run
    [int]$Port = 8090,
    [int]$Tokens = 300,
    [int]$WarmupTokens = 256,                                     # the throw-away request before the measured one (240 or more)
    [int]$Launches = 3,
    [double]$MinFreeGiB = 100,                                    # the pre-flight refuses to launch below this free physical memory
    [int]$MaxIdleGpuMiB = 4096,                                   # the pre-flight refuses above this GPU memory.used (963 MiB at idle)
    [string]$Python = 'python',
    [string]$ServerScript = '',                                   # empty = serve\server.py of this tree
    [string]$Tokenizer = '',                                      # tokenizer folder with vocab.json; empty = pack\full\tokenizer
    [string]$ExtraEnv = '',                                       # more engine settings, 'NAME=value;NAME=value' (issues #96, #80)
    [string]$ExtraArgs = '',                                      # more engine arguments, space separated (a fixed --pcie-frac, issue #80)
    [string]$PromptsFile = ''                                     # one prompt per line, sent in order (greedy parity); empty = the fixed prompt
)

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 leaves $PSScriptRoot empty in parameter defaults, so the default path is set here.
if ($ServerScript -eq '') { $ServerScript = Join-Path $PSScriptRoot '..\..\serve\server.py' }

$SampleMs = 2500
$MinWarmupTokens = 240        # PcieAuto's last update comes after 240 decode windows (one token each without MTP)

# A fixed prompt, the same for every launch, and the warm-up's (ASCII only, so the script reads the same in Windows PowerShell 5.1).
$Prompt = 'Write a Python function that parses an ISO 8601 date-time string without the standard library, then explain ' +
          'each step in a short paragraph and give three unit tests.'
$WarmupPrompt = 'Describe in detail, step by step, how a hash table handles collisions with open addressing and with ' +
                'chaining, then compare their memory use and their speed on a long list of examples.'

function Assert-File([string]$Path, [string]$What) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "$What not found: $Path" }
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

function Get-GpuEngineProcs {
    # Every process named strata.exe or llama-server.exe, whatever its path ('name PID n'), from one Win32_Process snapshot.
    $rows = @(Get-CimInstance -ClassName Win32_Process -Filter "Name = 'strata.exe' OR Name = 'llama-server.exe'" -Property ProcessId, Name)
    return , @($rows | ForEach-Object { "$($_.Name) PID $($_.ProcessId)" })
}

function Get-EngineProcs([string]$ExePath) {
    # The processes running from the engine binary's path (ExecutablePath, case-insensitive), from one Win32_Process snapshot.
    $rows = @(Get-CimInstance -ClassName Win32_Process -Property ProcessId, ExecutablePath)
    return , @($rows | Where-Object { $_.ExecutablePath -and [string]::Equals([string]$_.ExecutablePath, $ExePath, [StringComparison]::OrdinalIgnoreCase) } |
               ForEach-Object { [int]$_.ProcessId })
}

function Assert-PortFree {
    # A server that already answers on the port (an earlier run's, say) would be measured instead of the new one.
    $busy = $false
    try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null; $busy = $true } catch { $busy = $false }
    if ($busy) { throw "port $Port already answers /health: another server runs; stop it first" }
}

function Assert-PreFlight([string]$ExePath) {
    # Throws before anything is started. Get-VM must answer (it needs an elevated shell and the Hyper-V module).
    try { $vms = @(Get-VM -ErrorAction Stop) }
    catch { throw "pre-flight: Get-VM failed ($($_.Exception.Message)); run the script from an elevated shell with the Hyper-V module" }
    $runners = @($vms | Where-Object { $_.Name -like 'gitlab-runner*' -and "$($_.State)" -eq 'Running' })
    if ($runners.Count -gt 0) { throw "pre-flight: Hyper-V VM running: $(($runners | ForEach-Object { $_.Name }) -join ', '); stop it first" }
    $freeKiB = [long](Get-CimInstance -ClassName Win32_OperatingSystem).FreePhysicalMemory
    $freeGiB = $freeKiB / 1024.0 / 1024.0
    if ($freeGiB -lt $MinFreeGiB) { throw ("pre-flight: {0:F1} GiB of free physical memory, below -MinFreeGiB {1}" -f $freeGiB, $MinFreeGiB) }
    $gpuEngines = @(Get-GpuEngineProcs)
    if ($gpuEngines.Count -gt 0) { throw "pre-flight: an engine process runs: $($gpuEngines -join '; '); stop it first" }
    $usedMiB = Get-GpuUsedMiB
    if ($usedMiB -gt $MaxIdleGpuMiB) { throw "pre-flight: the GPU has $usedMiB MiB in use (memory.used), above -MaxIdleGpuMiB $MaxIdleGpuMiB; free it first" }
    $gpu = Get-GpuComputeApps
    if ($gpu.Lines.Count -gt 0) { Write-Host "pre-flight: compute processes listed (information only, not counted): $($gpu.Lines -join '; ')" }
    $engines = @(Get-EngineProcs $ExePath)
    if ($engines.Count -gt 0) { throw "pre-flight: a process already runs from ${ExePath}: PID $($engines -join ','); stop it first" }
    Assert-PortFree
    Write-Host ("pre-flight ok: {0:F1} GiB free, no gitlab-runner VM running, no strata.exe or llama-server.exe running, GPU {1} MiB in use (at most {2})" -f $freeGiB, $usedMiB, $MaxIdleGpuMiB)
}

function Get-ServerProcs([int]$Root, [datetime]$Since, [string]$ExePath) {
    # The server's processes alive in one Win32_Process snapshot: the root and its descendants by parent PID, created at or
    # after $Since (a PID reused by an older process is not taken; a child whose parent died keeps its ParentProcessId, so it
    # is still found), and every process running from the engine binary's path. Deepest first.
    $rows = @(Get-CimInstance -ClassName Win32_Process -Property ProcessId, ParentProcessId, ExecutablePath, CreationDate)
    $parents = New-Object System.Collections.Generic.List[int]
    $found = New-Object System.Collections.Generic.List[int]
    $parents.Add($Root)
    foreach ($r in $rows) { if ([int]$r.ProcessId -eq $Root -and $r.CreationDate -ge $Since) { $found.Add($Root) } }
    for ($i = 0; $i -lt $parents.Count; $i++) {
        foreach ($r in $rows) {
            $id = [int]$r.ProcessId
            if ([int]$r.ParentProcessId -eq $parents[$i] -and -not $parents.Contains($id) -and $r.CreationDate -ge $Since) {
                $parents.Add($id)
                $found.Add($id)
            }
        }
    }
    $ids = New-Object System.Collections.Generic.List[int]
    for ($i = $found.Count - 1; $i -ge 0; $i--) { $ids.Add($found[$i]) }
    foreach ($r in $rows) {
        $id = [int]$r.ProcessId
        if ($r.ExecutablePath -and [string]::Equals([string]$r.ExecutablePath, $ExePath, [StringComparison]::OrdinalIgnoreCase) -and
            -not $ids.Contains($id)) { $ids.Insert(0, $id) }
    }
    return , $ids.ToArray()
}

function Stop-ServerTree([int]$Root, [datetime]$Since, [string]$ExePath) {
    # Stops the server's processes (Get-ServerProcs), waits for each to exit, then takes the snapshot again (a process started
    # meanwhile is caught) until a snapshot finds none. Throws when one is still alive. Returns the PIDs stopped.
    $stopped = New-Object System.Collections.Generic.List[int]
    for ($pass = 1; $pass -le 5; $pass++) {
        $ids = @(Get-ServerProcs $Root $Since $ExePath)
        if ($ids.Count -eq 0) { break }
        $procs = @()
        foreach ($id in $ids) {
            $p = Get-Process -Id $id -ErrorAction SilentlyContinue
            if (-not $p) { continue }   # exited since the snapshot
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
    $left = @(Get-ServerProcs $Root $Since $ExePath)
    if ($left.Count -gt 0) { throw "server processes still alive after the stop: PID $($left -join ','); stop them by hand before another launch" }
    return , $stopped.ToArray()
}

function Wait-GpuFree([int[]]$Stopped, [int]$Seconds) {
    # After the stop: no strata.exe or llama-server.exe may run, the GPU's memory.used must be at most -MaxIdleGpuMiB, and none
    # of the stopped PIDs may still be listed as a compute process. Polls, then throws.
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ($true) {
        $gpuEngines = @(Get-GpuEngineProcs)
        $usedMiB = Get-GpuUsedMiB
        $gpu = Get-GpuComputeApps
        $ours = @($gpu.Pids | Where-Object { $Stopped -contains $_ })
        if ($gpuEngines.Count -eq 0 -and $usedMiB -le $MaxIdleGpuMiB -and $ours.Count -eq 0) { return }
        if ((Get-Date) -gt $deadline) {
            $enginesText = 'none'
            if ($gpuEngines.Count -gt 0) { $enginesText = $gpuEngines -join '; ' }
            $oursText = 'none'
            if ($ours.Count -gt 0) { $oursText = $ours -join ',' }
            throw ("the GPU is not free $Seconds s after the stop: {0} MiB in use (memory.used, -MaxIdleGpuMiB {1}); engine processes: {2}; stopped PIDs still listed: {3}; compute processes: {4}" -f
                   $usedMiB, $MaxIdleGpuMiB, $enginesText, $oursText, ($gpu.Lines -join '; '))
        }
        Start-Sleep -Seconds 1
    }
}

function Write-HyperV([string]$Path, [string]$Label) {
    # Lists the Hyper-V VMs, their state and load; nothing is started, stopped or changed.
    Add-Content -LiteralPath $Path -Value "== $Label $(Get-Date -Format s)"
    try {
        $vms = Get-VM -ErrorAction Stop | Select-Object Name, State, CPUUsage, MemoryAssigned, Uptime
        if ($vms) { Add-Content -LiteralPath $Path -Value ($vms | Format-Table -AutoSize | Out-String) }
        else { Add-Content -LiteralPath $Path -Value 'no Hyper-V VM' }
    } catch {
        Add-Content -LiteralPath $Path -Value "Get-VM unavailable: $($_.Exception.Message)"
    }
}

function Read-SharedText([string]$Path) {
    # The engine log while the server writes it (shared read).
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return '' }
    $fs = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try { $sr = New-Object IO.StreamReader($fs, [Text.Encoding]::UTF8); return $sr.ReadToEnd() } finally { $fs.Close() }
}

function Get-ShareAfter([string]$EngineLog, [int]$Requests) {
    # The PCIe share the engine reported at the end of its last request ('..., PCIe share H'), once the log holds the lines of
    # $Requests requests (the engine writes them just after the stream ends: polled for up to 10 s), or 'n/a'.
    $deadline = (Get-Date).AddSeconds(10)
    while ($true) {
        $ms = [regex]::Matches((Read-SharedText $EngineLog), 'experts on the GPU [^\r\n]*?, PCIe share (-?[\d.]+)')
        if ($ms.Count -ge $Requests) { return $ms[$ms.Count - 1].Groups[1].Value }
        if ((Get-Date) -gt $deadline) { return 'n/a' }
        Start-Sleep -Milliseconds 250
    }
}

function Invoke-ChatStream([string]$Content, [int]$MaxTokens) {
    # One greedy chat request, streamed (server-sent events): each chunk is timestamped on arrival, in ms from the moment the
    # request was sent. Returns the text, the times and sizes of the chunks that carried text, finish_reason and usage (when
    # the server sends it; stream_options.include_usage asks for it).
    $body = [ordered]@{
        messages       = @(@{ role = 'user'; content = $Content })
        max_tokens     = $MaxTokens
        temperature    = 0
        stream         = $true
        stream_options = @{ include_usage = $true }
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($body | ConvertTo-Json -Depth 6))
    $req = [Net.HttpWebRequest]::Create("http://127.0.0.1:$Port/v1/chat/completions")
    $req.Method = 'POST'
    $req.ContentType = 'application/json; charset=utf-8'
    $req.Accept = 'text/event-stream'
    $req.Proxy = $null
    $req.Timeout = 7200000
    $req.ReadWriteTimeout = 7200000
    $req.ServicePoint.Expect100Continue = $false
    $req.ContentLength = $bytes.Length
    $times = New-Object System.Collections.Generic.List[double]
    $chars = New-Object System.Collections.Generic.List[int]
    $text = New-Object Text.StringBuilder
    $finish = $null
    $usage = $null
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $rs = $req.GetRequestStream()
    try { $rs.Write($bytes, 0, $bytes.Length) } finally { $rs.Close() }
    $resp = $req.GetResponse()
    try {
        $reader = New-Object IO.StreamReader($resp.GetResponseStream(), [Text.Encoding]::UTF8)
        while ($null -ne ($line = $reader.ReadLine())) {
            $t = $sw.Elapsed.TotalMilliseconds
            if (-not $line.StartsWith('data:')) { continue }
            $data = $line.Substring(5).Trim()
            if ($data -eq '[DONE]') { break }
            if ($data -eq '') { continue }
            $j = $data | ConvertFrom-Json
            if ($j.error) { throw "the server answered an error in the stream: $data" }
            if ($j.usage) { $usage = $j.usage }
            foreach ($c in @($j.choices)) {
                if ($null -eq $c) { continue }
                if ($c.finish_reason) { $finish = [string]$c.finish_reason }
                $piece = ''
                if ($c.delta) {
                    if ($null -ne $c.delta.reasoning_content) { $piece += [string]$c.delta.reasoning_content }
                    if ($null -ne $c.delta.content) { $piece += [string]$c.delta.content }
                }
                if ($piece.Length -gt 0) {
                    $times.Add($t)
                    $chars.Add($piece.Length)
                    [void]$text.Append($piece)
                }
            }
        }
    } finally { $resp.Close() }
    $completion = $null
    if ($usage -and $null -ne $usage.completion_tokens) { $completion = [int]$usage.completion_tokens }
    return [pscustomobject]@{
        Text = $text.ToString(); Times = $times.ToArray(); Chars = $chars.ToArray(); Finish = $finish; Usage = $usage
        CompletionTokens = $completion; TotalMs = $sw.Elapsed.TotalMilliseconds
    }
}

function Test-Short($R, [int]$Asked) {
    # Fewer tokens than asked: usage.completion_tokens below max_tokens, or, without usage, a finish_reason other than 'length'.
    if ($null -ne $R.CompletionTokens) { return ($R.CompletionTokens -lt $Asked) }
    return ($R.Finish -ne 'length')
}

function Save-Request($R, [string]$Dir, [string]$RespName, [string]$ChunksName, [string]$PromptText, [int]$Asked, [string]$Share) {
    # The response record (the summary reads max_tokens, completion_tokens and finish_reason) and the chunks' times.
    $short = Test-Short $R $Asked
    $rec = [ordered]@{
        prompt            = $PromptText
        max_tokens        = $Asked
        completion_tokens = $R.CompletionTokens
        finish_reason     = $R.Finish
        short             = $short
        text_chunks       = $R.Times.Count
        ttft_ms           = $(if ($R.Times.Count -gt 0) { [math]::Round($R.Times[0], 1) } else { $null })
        total_ms          = [math]::Round($R.TotalMs, 1)
        pcie_share_after  = $Share
        usage             = $R.Usage
        content           = $R.Text
    }
    $utf8 = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText((Join-Path $Dir $RespName), ($rec | ConvertTo-Json -Depth 10), $utf8)
    $sb = New-Object Text.StringBuilder
    [void]$sb.Append("chunk`tt_ms`tchars`n")
    for ($i = 0; $i -lt $R.Times.Count; $i++) {
        [void]$sb.Append(([string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0}`t{1:F1}`t{2}`n", $i, $R.Times[$i], $R.Chars[$i])))
    }
    [IO.File]::WriteAllText((Join-Path $Dir $ChunksName), $sb.ToString(), $utf8)
    return $short
}

# The sampler runs as a background job: it writes free RAM (KiB) and VRAM (MiB) into mem.log until the server exits. It
# never stops a process.
$SamplerBlock = {
    param([int]$ServerPid, [string]$MemLog, [int]$SampleMs)
    $ErrorActionPreference = 'Continue'
    $t0 = Get-Date
    while ($true) {
        $freeKiB = [long](Get-CimInstance -ClassName Win32_OperatingSystem).FreePhysicalMemory
        $used = -1
        $total = -1
        $gpu = @(& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>$null)
        if ($gpu.Count -gt 0 -and "$($gpu[0])" -match '^\s*(\d+)\s*,\s*(\d+)') { $used = [int]$Matches[1]; $total = [int]$Matches[2] }
        $elapsed = ((Get-Date) - $t0).TotalSeconds
        Add-Content -LiteralPath $MemLog -Value ([string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0:F1} {1} {2} {3}', $elapsed, $freeKiB, $used, $total))
        if (-not (Get-Process -Id $ServerPid -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds $SampleMs
    }
}

# ---- inputs and output folder ------------------------------------------------------------------------------------------
Assert-File $Binary 'engine binary'
Assert-File $Gguf 'GGUF'
Assert-File $RankFile 'expert profile'
Assert-File $ServerScript 'serve/server.py'
if (-not (Test-Path -LiteralPath $BuildDir -PathType Container)) { throw "build folder not found: $BuildDir" }
if ($Tokens -lt 1 -or $Launches -lt 1) { throw 'Tokens and Launches must be at least 1' }
if ($WarmupTokens -lt $MinWarmupTokens) { throw "WarmupTokens must be at least $MinWarmupTokens (the automatic PCIe share's last update)" }
if ((Test-Path -LiteralPath $OutDir) -and @(Get-ChildItem -LiteralPath $OutDir -Filter 'launch_*' -ErrorAction SilentlyContinue).Count -gt 0) {
    throw "$OutDir already holds launch folders; choose a new folder"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$summaryPy = Join-Path $PSScriptRoot 'partition_summary.py'
$exePath = (Resolve-Path -LiteralPath $Binary).Path
# Without a tokenizer the server falls back to a byte tokenizer, which would give other token counts: it is required.
if ($Tokenizer -eq '') { $Tokenizer = Join-Path $PSScriptRoot '..\..\pack\full\tokenizer' }
if (-not (Test-Path -LiteralPath (Join-Path $Tokenizer 'vocab.json') -PathType Leaf)) {
    throw "no tokenizer (vocab.json) in $Tokenizer; pass -Tokenizer with the model's tokenizer folder"
}
$Tokenizer = (Resolve-Path -LiteralPath $Tokenizer).Path
$engineEnv = [ordered]@{ STRATA_Q36_STATS = '1' }
foreach ($pair in ($ExtraEnv -split ';' | Where-Object { $_.Trim() -ne '' })) {
    if ($pair -notmatch '^\s*(STRATA_[A-Z0-9_]+)=(.*)$') { throw "ExtraEnv entry is not NAME=value with a STRATA_ name: $pair" }
    $engineEnv[$Matches[1]] = $Matches[2].Trim()
}
$Prompts = @($Prompt)
if ($PromptsFile -ne '') {
    Assert-File $PromptsFile 'prompts file'
    $Prompts = @(Get-Content -LiteralPath $PromptsFile -Encoding UTF8 | Where-Object { $_.Trim() -ne '' })
    if ($Prompts.Count -eq 0) { throw "no prompt in $PromptsFile" }
}
$hyper = Join-Path $OutDir 'hyperv.txt'

$runInfo = [ordered]@{
    started       = (Get-Date -Format s)
    binary        = $exePath
    binary_sha256 = (Get-FileHash -LiteralPath $Binary -Algorithm SHA256).Hash
    build_dir     = (Resolve-Path -LiteralPath $BuildDir).Path
    gguf          = (Resolve-Path -LiteralPath $Gguf).Path
    gguf_bytes    = (Get-Item -LiteralPath $Gguf).Length
    rank_file     = (Resolve-Path -LiteralPath $RankFile).Path
    port          = $Port
    tokens        = $Tokens
    warmup_tokens = $WarmupTokens
    min_free_gib  = $MinFreeGiB
    max_idle_gpu_mib = $MaxIdleGpuMiB
    launches      = $Launches
    prompts       = $Prompts
    warmup_prompt = $WarmupPrompt
    env           = $engineEnv
    extra_args    = $ExtraArgs
    server        = (Resolve-Path -LiteralPath $ServerScript).Path
}
[IO.File]::WriteAllText((Join-Path $OutDir 'run.json'), ($runInfo | ConvertTo-Json -Depth 4), (New-Object Text.UTF8Encoding($false)))
Write-HyperV $hyper 'before the first launch'

# ---- launches ----------------------------------------------------------------------------------------------------------
for ($n = 1; $n -le $Launches; $n++) {
    $dir = Join-Path $OutDir ('launch_{0:D2}' -f $n)
    Write-Host "launch $n of $Launches"

    Assert-PreFlight $exePath
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $engineLog = Join-Path $dir 'engine.log'
    $memLog = Join-Path $dir 'mem.log'
    $cfgPath = Join-Path $dir 'config.json'
    Write-HyperV $hyper "launch $n"

    # The server's config: the engine's arguments (Ornith's launcher options, docs/ORNITH_122B.md) and the expert profile
    # read at start. No --expert-slots, and no --pcie-frac unless -ExtraArgs gives one: the automatic settings are measured.
    $engineArgs = @('--native', $Gguf, '--max-context', '409600', '--prefill', 'auto', '--mtp', '0',
              '--rope-scaling', 'yarn', '--rope-scale', '2', '--rope-orig-ctx', '262144', '--kv', 'q4',
              '--expert-profile', $RankFile) + @($ExtraArgs -split '\s+' | Where-Object { $_ -ne '' })
    $cfg = [ordered]@{
        exe        = $exePath
        args       = $engineArgs
        cwd        = (Resolve-Path -LiteralPath $BuildDir).Path
        model_name = 'ornith-1.5-122b'
        arch       = 'qwen35moe'
        log        = $engineLog
        lib_dirs   = @((Resolve-Path -LiteralPath $BuildDir).Path)
        port       = $Port
        env        = $engineEnv
    }
    $cfg['tokenizer'] = $Tokenizer
    [IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json -Depth 6), (New-Object Text.UTF8Encoding($false)))

    $srv = $null
    $sampler = $null
    $since = (Get-Date).AddSeconds(-5)   # the processes of this launch are created after this (a little slack for the clock)
    try {
        $srvArgs = @($ServerScript, '--engine', 'strata', '--config', $cfgPath, '--port', "$Port")
        $srv = Start-Process -FilePath $Python -ArgumentList $srvArgs -WorkingDirectory $BuildDir -NoNewWindow -PassThru `
            -RedirectStandardOutput (Join-Path $dir 'server.out') -RedirectStandardError (Join-Path $dir 'server.err')
        Write-Host "server pid $($srv.Id)"

        $sampler = Start-Job -ScriptBlock $SamplerBlock -ArgumentList $srv.Id, $memLog, $SampleMs

        # ready: /health answers ok and the engine is loaded
        $deadline = (Get-Date).AddMinutes(30)
        while ($true) {
            if (-not (Get-Process -Id $srv.Id -ErrorAction SilentlyContinue)) { throw "the server exited before it was ready; see $dir" }
            $ready = $false
            try {
                $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
                $ready = ($h.status -eq 'ok') -and ($h.loaded -ne $false)
            } catch { $ready = $false }
            if ($ready) { break }
            if ((Get-Date) -gt $deadline) { throw "no ready /health after 30 minutes; see $dir" }
            Start-Sleep -Seconds 2
        }
        Write-Host 'ready'

        # the throw-away warm-up: the automatic PCIe share makes its last update during it, not during the measured request
        $w = Invoke-ChatStream $WarmupPrompt $WarmupTokens
        $share = Get-ShareAfter $engineLog 1
        # (named apart from response*.json and chunks*.tsv, which the summary reads as the measured request)
        $wShort = Save-Request $w $dir 'warmup_response.json' 'warmup_chunks.tsv' $WarmupPrompt $WarmupTokens $share
        Write-Host ('warm-up done: {0} completion tokens (finish_reason {1}), PCIe share {2}' -f $w.CompletionTokens, $w.Finish, $share)
        if ($wShort) { Write-Warning "launch ${n}: the warm-up stopped before $WarmupTokens tokens: the PCIe share may still move in the measured request" }

        # one greedy streamed chat request of N tokens per prompt; a single prompt keeps the names response.json and chunks.tsv
        for ($k = 0; $k -lt $Prompts.Count; $k++) {
            $r = Invoke-ChatStream $Prompts[$k] $Tokens
            $share = Get-ShareAfter $engineLog ($k + 2)
            $suffix = if ($Prompts.Count -eq 1) { '' } else { '_{0:D2}' -f ($k + 1) }
            $short = Save-Request $r $dir "response$suffix.json" "chunks$suffix.tsv" $Prompts[$k] $Tokens $share
            Write-Host ('request {0} done: {1} completion tokens (finish_reason {2}), {3} text chunks in {4:F1} s, PCIe share {5}' -f
                        ($k + 1), $r.CompletionTokens, $r.Finish, $r.Times.Count, ($r.TotalMs / 1000.0), $share)
            if ($short) { Write-Warning "launch ${n}, request $($k + 1): SHORT, fewer than $Tokens tokens (flagged in the response file and the summary)" }
        }
    }
    catch {
        Write-Warning "launch ${n} failed: $($_.Exception.Message)"
        throw
    }
    finally {
        # the server's processes, verified; then the sampler. A failed stop throws (it replaces an error of the launch, which
        # the warning above has printed).
        $stopError = $null
        if ($srv) {
            try {
                $stopped = Stop-ServerTree $srv.Id $since $exePath
                Write-Host "server stopped: PID $($stopped -join ',')"
                Wait-GpuFree $stopped 60
            } catch { $stopError = $_.Exception.Message }
        }
        if ($sampler) {
            Wait-Job -Job $sampler -Timeout 15 | Out-Null
            Receive-Job -Job $sampler -ErrorAction SilentlyContinue | Out-Null
            Stop-Job -Job $sampler -ErrorAction SilentlyContinue
            Remove-Job -Job $sampler -Force -ErrorAction SilentlyContinue
        }
        if ($stopError) { throw "launch ${n}: the server was not stopped cleanly: $stopError" }
    }
    if (-not (Select-String -LiteralPath $engineLog -Pattern 'stats: pool' -Quiet -ErrorAction SilentlyContinue)) {
        Write-Warning "launch ${n}: no stats line in $engineLog (the engine log is kept for reading)"
    }
}
Write-HyperV $hyper 'after the last launch'

# ---- summary -----------------------------------------------------------------------------------------------------------
& $Python -I $summaryPy $OutDir | Tee-Object -FilePath (Join-Path $OutDir 'summary.txt')
