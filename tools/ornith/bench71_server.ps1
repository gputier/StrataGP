<#
.SYNOPSIS
Issue #71 on the server path: prompt p00 resumed N times in greedy decoding on one strata serve process, each time while
two other sessions decode a long greedy answer on the same server; every resumed output is compared with the first one.

.DESCRIPTION
Starts one serve\server.py with the strata engine (a config of its own: the CLI bench's engine arguments and "parallel": 3 batch
slots; -ExtraArgs and -ExtraEnv add more), stopped by Stop-ServerTree as measure_partition.ps1 stops it (process tree by
parent PID, in a finally block; unlike measure_partition.ps1 it does not wait for the GPU memory to come back, the next
start's pre-flight refuses a busy GPU). Then, -Runs times (default 40, -Trial: 2): two greedy load requests of -LoadTokens tokens are kept in flight
(a finished one is replaced, then -LoadSettleS seconds of settling), one greedy chat request for p00 asks -Tokens tokens (default:
ids in p00.force + 1), and its output is compared with run 0. A run is "under load" when loads=2/2 at its start and its end.
The server takes chat messages only and returns text, usage and finish_reason: no token ids, no logprobs (see
docs/ORNITH_122B.md). So the script compares TEXT and usage.completion_tokens and reports the first differing CHARACTER.
SAME means the same greedy token choices, not bit-identical logits; DIFF is real.
Files in OutDir: status.txt (START, DONE, CMP per run, then RESULT), run_NN.txt, run_NN.json, loads.log, config.json, run.json,
engine.log, server.out, server.err.
REFUSED AT THE START (nothing is started): a gitlab-runner* VM that is not Off (Get-VM needs an elevated shell), free physical
memory below -MinFreeGiB, GPU memory.used above -MaxIdleGpuMiB, a strata.exe, strata-q36.exe or llama-server.exe process or one
running from the engine binary's path, a server already answering on the port, an OutDir that exists, a path parameter that
holds a double quote.

.EXAMPLE
powershell -File tools\ornith\bench71_server.ps1 -BuildDir E:\StrataGP-89\final -OutDir E:\ornith\runs\bench71_server_01
.EXAMPLE
powershell -File tools\ornith\bench71_server.ps1 -BuildDir E:\StrataGP-89\final -OutDir E:\ornith\runs\bench71_trial -Trial
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$BuildDir,              # the build folder (strata.exe and its DLLs; read only)
    [Parameter(Mandatory = $true)][string]$OutDir,                # new folder: it must not exist yet
    [int]$Runs = 40,                                              # prompt p00 resumed this many times (run 0 is the reference)
    [switch]$Trial,                                               # 2 runs only
    [string]$Gguf = 'E:\ornith\models\Ornith-1.5-122B-IQ3XXS-IQ4XS.gguf',
    [string]$IdsFile = 'E:\ornith\gates\p00.ids',                 # token ids of p00, space separated
    [string]$ForceFile = 'E:\ornith\gates\p00.force',             # the CLI's forced continuation; here only its length is used
    [string]$Binary = '',                                         # empty = strata.exe of -BuildDir
    [int]$Port = 8090,
    [int]$Tokens = 0,                                             # tokens asked for p00; 0 = ids in the force file + 1 (as the CLI)
    [int]$LoadTokens = 4096,                                      # tokens asked of each load request
    [int]$LoadSettleS = 15,                                       # seconds to wait after starting a load request
    [int]$MaxContext = 8192,                                      # per session: 62 slots and 3 sessions fit at 8192, not at 16384
    [int]$Slots = 3,                                              # the server batch slots: 2 loads + the resumed run, at least 3
    [int]$TimeoutS = 600,                                         # one resumed run (a timeout stops the server and the script)
    [double]$MinFreeGiB = 100,
    [int]$MaxIdleGpuMiB = 4096,
    [string]$ExtraArgs = '',                                      # more engine arguments, space separated
    [string]$ExtraEnv = '',                                       # more engine settings, 'STRATA_NAME=value;STRATA_NAME=value'
    [string]$Python = 'python',
    [string]$ServerScript = '',                                   # empty = serve\server.py of this tree
    [string]$Tokenizer = '',                                      # tokenizer folder with vocab.json; empty = pack\full\tokenizer
    [string]$PromptText = ''                                      # p00's user message; empty = the one of parity_prompts.json[0]
)

$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 leaves $PSScriptRoot empty in parameter defaults, so the defaults are set here.
if ($ServerScript -eq '') { $ServerScript = Join-Path $PSScriptRoot '..\..\serve\server.py' }
if ($Binary -eq '') { $Binary = Join-Path $BuildDir 'strata.exe' }
if ($Trial) { $Runs = 2 }
. (Join-Path $PSScriptRoot 'ornith_server_common.ps1')

# p00 as chat text: entry 0 of tools\qwen35\parity_prompts.json (ASCII only, so this file reads the same in Windows PowerShell 5.1)
$P00Text = 'Write a short story about a lighthouse keeper who finds a message in a bottle.'
$ParityIds = Join-Path $PSScriptRoot '..\qwen35\parity_prompts.ids'
# The two load prompts: long answers, greedy.
$LoadPrompts = @(
    'Write a long, detailed story about a lighthouse keeper who finds a message in a bottle and follows it across the sea.',
    'Explain in great detail, chapter by chapter, how a modern compiler turns source code into machine code, with examples.'
)

function Read-IdList([string]$Path, [string]$What) {
    # Token ids separated by spaces, commas or line breaks, as the engine's --tokens and --force take them.
    $raw = (Get-Content -LiteralPath $Path -Raw)
    if ($null -eq $raw) { $raw = '' }
    $parts = @(($raw -split '[,\s]+') | Where-Object { $_ -ne '' })
    foreach ($p in $parts) { if ($p -notmatch '^\d+$') { throw "$What ($Path) holds a value that is not a token id: $p" } }
    return , @($parts | ForEach-Object { [int64]$_ })
}

function Add-Status([string]$Line) {
    # One line of status.txt (ASCII), also shown.
    Add-Content -LiteralPath $script:StatusFile -Value $Line -Encoding ASCII
    Write-Host $Line
}

function Add-LoadLog([string]$Line) {
    Add-Content -LiteralPath $script:LoadLog -Value ("{0} {1}" -f (Get-Date -Format s), $Line) -Encoding ASCII
}

function Start-Chat([string]$Content, [int]$MaxTokens) {
    # One greedy chat request, not streamed, started without waiting: returns the Task. The sampling fields are all sent: the
    # server merges the model card's sampling (presence_penalty 1.5 for Ornith) under the request's own fields, so a field
    # that is not sent would be the card's; with these the engine gets a plain greedy request, as the CLI's. enable_thinking
    # false gives the empty think block that p00.ids ends with.
    $body = [ordered]@{
        messages             = @(@{ role = 'user'; content = $Content })
        max_tokens           = $MaxTokens
        temperature          = 0
        top_p                = 1
        presence_penalty     = 0
        frequency_penalty    = 0
        repetition_penalty   = 1
        stream               = $false
        chat_template_kwargs = @{ enable_thinking = $false }
    }
    $json = $body | ConvertTo-Json -Depth 6 -Compress
    $sc = New-Object System.Net.Http.StringContent($json, (New-Object Text.UTF8Encoding($false)), 'application/json')
    return $script:Http.PostAsync("http://127.0.0.1:$Port/v1/chat/completions", $sc)
}

function Get-InnerMessage($Exception) {
    # The message of the innermost exception (a failed Task wraps the real error in AggregateException layers).
    $e = $Exception
    while ($null -ne $e.InnerException) { $e = $e.InnerException }
    return $e.Message
}

function Wait-ChatTask($Task, [int]$Seconds, [string]$What) {
    # Waits for the Task; throws on a timeout (the caller's finally stops the server) or when the request failed.
    $done = $false
    try { $done = $Task.Wait($Seconds * 1000) }
    catch { throw "$What failed: $(Get-InnerMessage $_.Exception)" }
    if (-not $done) { throw "${What}: no answer after $Seconds s (timeout)" }
}

function Get-ChatResult($Task, [string]$What) {
    # The answer of a finished request: text (reasoning_content, if any, then content), completion tokens, prompt tokens,
    # cached tokens, finish_reason, and the raw JSON. Throws on a failed task or an HTTP error.
    if ($Task.IsFaulted) { throw "$What failed: $(Get-InnerMessage $Task.Exception)" }
    $resp = $Task.Result
    $bytes = $resp.Content.ReadAsByteArrayAsync().Result
    $raw = [Text.Encoding]::UTF8.GetString($bytes)   # (never the header's charset: a missing one means Latin-1 to HttpClient)
    if (-not $resp.IsSuccessStatusCode) { throw "${What}: HTTP $([int]$resp.StatusCode): $raw" }
    $j = $raw | ConvertFrom-Json
    if ($j.error) { throw "${What}: the server answered an error: $raw" }
    $c = @($j.choices)[0]
    if ($null -eq $c) { throw "${What}: no choice in the answer: $raw" }
    $content = ''
    if ($null -ne $c.message.content) { $content = [string]$c.message.content }
    $reason = ''
    if ($null -ne $c.message.reasoning_content) { $reason = [string]$c.message.reasoning_content }
    $text = $content
    if ($reason -ne '') { $text = '[reasoning]' + $reason + '[/reasoning]' + $content }
    $cached = 0
    if ($j.usage -and $j.usage.prompt_tokens_details -and $null -ne $j.usage.prompt_tokens_details.cached_tokens) { $cached = [int]$j.usage.prompt_tokens_details.cached_tokens }
    $completion = -1
    $prompt = -1
    if ($j.usage) {
        if ($null -ne $j.usage.completion_tokens) { $completion = [int]$j.usage.completion_tokens }
        if ($null -ne $j.usage.prompt_tokens) { $prompt = [int]$j.usage.prompt_tokens }
    }
    return [pscustomobject]@{ Text = $text; Completion = $completion; Prompt = $prompt; Cached = $cached; Finish = [string]$c.finish_reason; Raw = $raw }
}

function Get-LoadsAlive {
    # The number of load requests still running. A load that failed throws here (its answer is not a usable load).
    $n = 0
    foreach ($l in $script:Loads) {
        if ($null -eq $l) { continue }
        if ($l.Task.IsFaulted) { throw "load $($l.Id) failed: $(Get-InnerMessage $l.Task.Exception)" }
        if ($l.Task.IsCanceled) { throw "load $($l.Id) was cancelled" }
        if (-not $l.Task.IsCompleted) { $n++ }
    }
    return $n
}

function Start-MissingLoads {
    # Keeps two load requests in flight: one that ended is checked (an HTTP error throws) and replaced. Returns how many
    # were started.
    $started = 0
    for ($i = 0; $i -lt 2; $i++) {
        $l = $script:Loads[$i]
        if ($null -ne $l) {
            if (-not ($l.Task.IsCompleted)) { continue }
            $r = Get-ChatResult $l.Task "load $($l.Id)"
            Add-LoadLog ("load {0} ended: {1} completion tokens, finish_reason {2}, after {3:F0} s" -f $l.Id, $r.Completion, $r.Finish, ((Get-Date) - $l.Started).TotalSeconds)
        }
        $script:LoadSeq++
        $prompt = $LoadPrompts[($script:LoadSeq - 1) % $LoadPrompts.Count]
        $task = Start-Chat $prompt $LoadTokens
        $script:Loads[$i] = [pscustomobject]@{ Id = $script:LoadSeq; Task = $task; Started = (Get-Date) }
        Add-LoadLog "load $($script:LoadSeq) started in place $i ($LoadTokens tokens asked)"
        $started++
    }
    return $started
}

function Get-FirstDiff([string]$A, [string]$B) {
    # The index of the first character that differs (code unit by code unit; -1 when the strings are equal).
    $n = [Math]::Min($A.Length, $B.Length)
    for ($i = 0; $i -lt $n; $i++) { if ([int]$A[$i] -ne [int]$B[$i]) { return $i } }
    if ($A.Length -ne $B.Length) { return $n }
    return -1
}

function Get-Snippet([string]$S, [int]$At) {
    # 40 characters from $At, as ASCII on one line, for the status file.
    if ($At -lt 0 -or $At -ge $S.Length) { return '<end>' }   # (-1: the texts are equal)
    $len = [Math]::Min(40, $S.Length - $At)
    return (($S.Substring($At, $len) -replace "[\r\n]", '|') -replace '[^\x20-\x7E]', '?')
}

function Get-Sha256Text([string]$S) {
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return (($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($S)) | ForEach-Object { $_.ToString('x2') }) -join '') }
    finally { $sha.Dispose() }
}

# ---- inputs and refusals -----------------------------------------------------------------------------------------------
# A path parameter that holds a double quote is refused: the server's arguments are quoted for Start-Process, and a
# double quote inside one would end the quoting and add arguments.
$pathParams = [ordered]@{ BuildDir = $BuildDir; OutDir = $OutDir; Gguf = $Gguf; IdsFile = $IdsFile; ForceFile = $ForceFile
                          Binary = $Binary; Python = $Python; ServerScript = $ServerScript; Tokenizer = $Tokenizer }
Assert-NoDoubleQuote $pathParams
if (Test-Path -LiteralPath $OutDir) { throw "$OutDir already exists; choose a new folder" }
if ($Runs -lt 1) { throw 'Runs must be at least 1' }
if ($Slots -lt 3) { throw 'Slots must be at least 3 (two load requests and the resumed run decode together)' }
if ($LoadTokens -lt 1 -or $TimeoutS -lt 1 -or $LoadSettleS -lt 0 -or $Tokens -lt 0) { throw 'LoadTokens and TimeoutS must be at least 1, LoadSettleS and Tokens at least 0' }
Assert-File $Binary 'engine binary'
Assert-File $Gguf 'GGUF'
Assert-File $IdsFile 'p00 ids file'
Assert-File $ForceFile 'p00 force file'
Assert-File $ServerScript 'serve/server.py'
if (-not (Test-Path -LiteralPath $BuildDir -PathType Container)) { throw "build folder not found: $BuildDir" }
# Without a tokenizer the server falls back to a byte tokenizer, which would give other token counts: it is required.
if ($Tokenizer -eq '') { $Tokenizer = Join-Path $PSScriptRoot '..\..\pack\full\tokenizer' }
if (-not (Test-Path -LiteralPath (Join-Path $Tokenizer 'vocab.json') -PathType Leaf)) {
    throw "no tokenizer (vocab.json) in $Tokenizer; pass -Tokenizer with the model's tokenizer folder"
}
$Tokenizer = (Resolve-Path -LiteralPath $Tokenizer).Path

$idsList = Read-IdList $IdsFile 'p00 ids'
$forceList = Read-IdList $ForceFile 'p00 force'
if ($idsList.Count -lt 1) { throw "no token id in $IdsFile" }
if ($forceList.Count -lt 1) { throw "no token id in $ForceFile" }
$asked = $Tokens
if ($asked -eq 0) { $asked = $forceList.Count + 1 }   # the CLI decodes force count + 1 tokens (n_gen)

# The prompt the server gets: p00's user message. The ids file must be the templated form of that same message.
$promptText = $PromptText
if ($promptText -eq '') {
    $promptText = $P00Text
    if (-not (Test-Path -LiteralPath $ParityIds -PathType Leaf)) { throw "the tree's parity_prompts.ids is missing ($ParityIds): cannot check that $IdsFile is p00's prompt; pass -PromptText to skip the check" }
    $first = @((Get-Content -LiteralPath $ParityIds -TotalCount 1) -split '[,\s]+' | Where-Object { $_ -ne '' } | ForEach-Object { [int64]$_ })
    if (($first -join ' ') -ne ($idsList -join ' ')) {
        throw ("$IdsFile ($($idsList.Count) ids) is not line 1 of tools\qwen35\parity_prompts.ids ($($first.Count) ids): the prompt text this script sends " +
               "would not be the same prompt. Pass -PromptText with the user message that gives these ids (the first answer's usage.prompt_tokens is still checked against the id count).")
    }
}

$exePath = (Resolve-Path -LiteralPath $Binary).Path
$engineEnv = [ordered]@{}
foreach ($pair in ($ExtraEnv -split ';' | Where-Object { $_.Trim() -ne '' })) {
    if ($pair -notmatch '^\s*(STRATA_[A-Z0-9_]+)=(.*)$') { throw "ExtraEnv entry is not NAME=value with a STRATA_ name: $pair" }
    $engineEnv[$Matches[1]] = $Matches[2].Trim()
}

Assert-PreFlight $exePath

New-Item -ItemType Directory -Path $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$script:StatusFile = Join-Path $OutDir 'status.txt'
$script:LoadLog = Join-Path $OutDir 'loads.log'
$engineLog = Join-Path $OutDir 'engine.log'
$cfgPath = Join-Path $OutDir 'config.json'
New-Item -ItemType File -Path $script:StatusFile | Out-Null
New-Item -ItemType File -Path $script:LoadLog | Out-Null

# The server's config: the CLI bench's engine arguments, and "parallel" (batch slots), so that other sessions can decode
# while the resumed run does.
$engineArgs = @('--native', $Gguf, '--max-context', "$MaxContext", '--expert-slots', '62', '--chunk', '2048', '--pcie-frac', '0',
                '--rope-scaling', 'yarn', '--rope-scale', '1') + @($ExtraArgs -split '\s+' | Where-Object { $_ -ne '' })
$cfg = [ordered]@{
    exe        = $exePath
    args       = $engineArgs
    cwd        = (Resolve-Path -LiteralPath $BuildDir).Path
    model_name = 'ornith-1.5-122b'
    arch       = 'qwen35moe'
    log        = $engineLog
    lib_dirs   = @((Resolve-Path -LiteralPath $BuildDir).Path)
    port       = $Port
    parallel   = $Slots
    tokenizer  = $Tokenizer
    env        = $engineEnv
}
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json -Depth 6), $utf8)
$runInfo = [ordered]@{
    started       = (Get-Date -Format s)
    binary        = $exePath
    binary_sha256 = (Get-FileHash -LiteralPath $exePath -Algorithm SHA256).Hash
    gguf          = (Resolve-Path -LiteralPath $Gguf).Path
    gguf_bytes    = (Get-Item -LiteralPath $Gguf).Length
    ids_file      = $IdsFile
    ids_count     = $idsList.Count
    force_file    = $ForceFile
    force_count   = $forceList.Count
    tokens_asked  = $asked
    runs          = $Runs
    load_tokens   = $LoadTokens
    parallel      = $Slots
    prompt_text   = $promptText
    load_prompts  = $LoadPrompts
    compare       = 'text (content, reasoning_content) and usage.completion_tokens; the server returns no token ids'
    engine_args   = $engineArgs
    env           = $engineEnv
    server        = (Resolve-Path -LiteralPath $ServerScript).Path
}
[IO.File]::WriteAllText((Join-Path $OutDir 'run.json'), ($runInfo | ConvertTo-Json -Depth 4), $utf8)

# ---- the server and the runs -------------------------------------------------------------------------------------------
Add-Type -AssemblyName System.Net.Http
[Net.ServicePointManager]::DefaultConnectionLimit = 32   # (the default of 2 would queue the third request on the client side)
$handler = New-Object System.Net.Http.HttpClientHandler
$handler.UseProxy = $false
$script:Http = New-Object System.Net.Http.HttpClient($handler)
$script:Http.Timeout = [Threading.Timeout]::InfiniteTimeSpan   # (each wait has its own limit)
$script:Loads = @($null, $null)
$script:LoadSeq = 0

$srv = $null
$srvStart = $null   # the StartTime of the started process: the stop's tree is rooted at it only while it matches
$runError = $null
$sameCount = 0
$underLoad = 0
$firstDiffRun = -1
try {
    # each argument quoted by hand (Windows PowerShell 5.1 joins -ArgumentList with spaces and quotes nothing), one string
    $srvArgs = (@($ServerScript, '--engine', 'strata', '--config', $cfgPath, '--port', "$Port") |
                ForEach-Object { ConvertTo-ProcessArg $_ }) -join ' '
    $srv = Start-Process -FilePath $Python -ArgumentList $srvArgs -WorkingDirectory $BuildDir -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $OutDir 'server.out') -RedirectStandardError (Join-Path $OutDir 'server.err')
    try { $srvStart = $srv.StartTime } catch { Write-Warning "no start time for the server (PID $($srv.Id)): $($_.Exception.Message); the stop will throw while it is alive" }
    Write-Host "server pid $($srv.Id)"

    # ready: /health answers ok and the engine is loaded
    $deadline = (Get-Date).AddMinutes(30)
    while ($true) {
        if (-not (Get-Process -Id $srv.Id -ErrorAction SilentlyContinue)) { throw "the server exited before it was ready; see $OutDir" }
        $ready = $false
        try {
            $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
            $ready = ($h.status -eq 'ok') -and ($h.loaded -ne $false)
        } catch { $ready = $false }
        if ($ready) { break }
        if ((Get-Date) -gt $deadline) { throw "no ready /health after 30 minutes; see $OutDir" }
        Start-Sleep -Seconds 2
    }
    Write-Host 'ready'

    for ($k = 0; $k -lt $Runs; $k++) {
        $nn = '{0:D2}' -f $k
        $startedLoads = Start-MissingLoads
        if ($startedLoads -gt 0) { Start-Sleep -Seconds $LoadSettleS }   # both loads past their prompt read and decoding
        $aliveStart = Get-LoadsAlive
        Add-Status ("START {0} {1} loads={2}/2 tokens_asked={3}" -f $nn, (Get-Date -Format s), $aliveStart, $asked)
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $task = Start-Chat $promptText $asked
        Wait-ChatTask $task $TimeoutS "run $nn"
        $ms = $sw.Elapsed.TotalMilliseconds
        $r = Get-ChatResult $task "run $nn"
        $aliveEnd = Get-LoadsAlive
        if ($k -eq 0 -and $r.Prompt -ne $idsList.Count) {
            throw ("the server read $($r.Prompt) prompt tokens for p00, $IdsFile holds $($idsList.Count): the server's prompt is not p00's ids, so " +
                   "this is not the CLI bench's prompt (pass -PromptText with the user message that gives these ids)")
        }
        [IO.File]::WriteAllText((Join-Path $OutDir "run_$nn.txt"), $r.Text, $utf8)
        [IO.File]::WriteAllText((Join-Path $OutDir "run_$nn.json"), $r.Raw, $utf8)
        $short = 0
        if ($r.Completion -lt $asked -or $r.Finish -ne 'length') { $short = 1 }
        $sha = Get-Sha256Text $r.Text
        Add-Status ("DONE {0} {1} ms={2:F0} completion_tokens={3} prompt_tokens={4} cached_tokens={5} finish={6} short={7} chars={8} sha256={9} loads={10}/2" -f
                    $nn, (Get-Date -Format s), $ms, $r.Completion, $r.Prompt, $r.Cached, $r.Finish, $short, $r.Text.Length, $sha, $aliveEnd)
        if ($short -eq 1) { Write-Warning "run ${nn}: fewer than $asked tokens or finish_reason is not length (see the DONE line)" }
        $loaded = ($aliveStart -eq 2 -and $aliveEnd -eq 2)
        if ($loaded) { $underLoad++ } else { Write-Warning "run ${nn}: not under full load (loads $aliveStart/2 at the start, $aliveEnd/2 at the end)" }
        if ($k -eq 0) {
            $refText = $r.Text
            $refCompletion = $r.Completion
            $sameCount++
            Add-Status ("CMP {0} vs 00 REFERENCE" -f $nn)
        } else {
            $d = -1   # first differing character; -1 when the texts are equal
            if (-not ($refText -ceq $r.Text)) { $d = Get-FirstDiff $refText $r.Text }
            $verdict = 'DIFF'
            if ($d -lt 0 -and $r.Completion -eq $refCompletion) { $verdict = 'SAME'; $sameCount++ }
            elseif ($firstDiffRun -lt 0) { $firstDiffRun = $k }
            Add-Status ("CMP {0} vs 00 {1} first_char={2} ref_chars={3} run_chars={4} ref_tokens={5} run_tokens={6} ref_at='{7}' run_at='{8}'" -f
                        $nn, $verdict, $d, $refText.Length, $r.Text.Length, $refCompletion, $r.Completion, (Get-Snippet $refText $d), (Get-Snippet $r.Text $d))
        }
    }
    # a load that failed during the last run is an error too (its answer is read here)
    [void](Get-LoadsAlive)
    $diffText = 'none'
    if ($firstDiffRun -ge 0) { $diffText = ('{0:D2}' -f $firstDiffRun) }
    Add-Status ("RESULT {0} of {1} identical to run 00 (text and completion tokens, not token ids: the server returns none); first differing run {2}; {3} of {1} runs under full load (both loads decoding at start and end)" -f
                $sameCount, $Runs, $diffText, $underLoad)
}
catch {
    # the full error record (message, position) and the script stack, kept for the finally's error too
    $runError = $_
    Write-Warning ("failed: " + ($_ | Out-String) + $_.ScriptStackTrace)
    try { Add-Status ("FAIL {0} {1}" -f (Get-Date -Format s), (($_.Exception.Message -replace "[\r\n]", ' ') -replace '[^\x20-\x7E]', '?')) } catch { }
    throw
}
finally {
    # the in-flight requests are dropped, then the server's processes are stopped and verified. A failed stop throws, and its
    # error carries the run's error record and stack too when the run had failed first.
    try { $script:Http.Dispose() } catch { }
    $stopError = $null
    if ($srv) {
        try {
            $stopped = Stop-ServerTree $srv.Id $srvStart $exePath
            if ($stopped.Count -gt 0) { Write-Host "server stopped: PID $($stopped -join ',')" }
            else { Write-Warning "no server process was left to stop (PID $($srv.Id) had already exited)" }
        } catch { $stopError = $_ }
    }
    if ($stopError) {
        $msg = "the server was not stopped cleanly: " + ($stopError | Out-String) + $stopError.ScriptStackTrace
        if ($runError) { $msg += "`nthe run had failed first: " + ($runError | Out-String) + $runError.ScriptStackTrace }
        throw $msg
    }
}
