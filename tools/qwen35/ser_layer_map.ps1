# tools/qwen35/ser_layer_map.ps1 - SER sensitivity map (#108), Windows PowerShell 5.1. For each main layer, one engine run per
# parity prompt (p00 .. p09) with STRATA_Q36_SER_TABLE holding that layer alone at -TableKmin and --ser's setting everywhere
# else, teacher-forced on llama.cpp b11756's greedy continuation, then logit_cmp against its full logits. The engine flags
# are run-v15.ps1's (the KL series of docs/ORNITH_122B.md). Writes <OutDir>\ser_layer_map.csv with one line per layer:
#   layer,kl_mean,top1
# kl_mean is the mean of the ten prompts' KL(ref||test) means; top1 is the mean fraction of the 32 steps whose top-1 agrees.
# -Baseline adds a first line, layer -1, the same runs with no table (--ser alone), to read each layer's KL against.
# Nothing runs on the GPU until the script is started; each run passes the GPU pre-flight of ornith_server_common.ps1 first.
param(
    [Parameter(Mandatory = $true)] [string]$Exe,            # strata-q36.exe
    [Parameter(Mandatory = $true)] [string]$Cmp,            # logit_cmp.exe
    [Parameter(Mandatory = $true)] [string]$Gguf,           # the Ornith GGUF
    [Parameter(Mandatory = $true)] [string]$Gates,          # the references: pNN.ids, pNN.force, pNN.llama.full
    [Parameter(Mandatory = $true)] [string]$OutDir,         # the CSV and the per-run files
    [string]$Ser = '8,0.6',                                 # --ser Kmin,t, global: every layer but the one under test (Kmin 8: SER off there, t for the table's layer)
    [int[]]$Layers = (0..47),                               # the main layers to map
    [int]$TableKmin = 6,                                    # the Kmin of the layer under test
    [int]$Steps = 32,
    [int]$Vocab = 248320,
    [switch]$Baseline,                                      # also the runs with --ser alone (layer -1)
    [string]$Common = (Join-Path $PSScriptRoot '..\ornith\ornith_server_common.ps1')
)
$ErrorActionPreference = 'Continue'   # (a native command's stderr must not throw; the exit codes are checked)
$Port = 8091; $MinFreeGiB = 100; $MaxIdleGpuMiB = 4096   # (read by Assert-PreFlight, which the common script defines)
. $Common
Assert-File $Exe 'strata-q36.exe'
Assert-File $Cmp 'logit_cmp.exe'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$status = Join-Path $OutDir 'status.log'
$csv = Join-Path $OutDir 'ser_layer_map.csv'
$inv = [Globalization.CultureInfo]::InvariantCulture
function Log([string]$s) { $s | Add-Content $status }
function Get-Figure([string]$File, [string]$Pattern) {
    $m = Select-String -LiteralPath $File -Pattern $Pattern
    if (-not $m) { throw "no figure matching '$Pattern' in $File" }
    return $m.Matches[0].Groups[1].Value
}
# The ten prompts for one table (or none): returns the mean KL and the mean top-1 fraction. Exits the script on a failed run.
function Invoke-Prompts([string]$Tag, [string]$Dir, [string]$Table) {
    New-Item -ItemType Directory -Force -Path $Dir | Out-Null
    $kls = @(); $tops = @()
    foreach ($i in 0..9) {
        $p = 'p{0:D2}' -f $i
        try { Assert-PreFlight $Exe } catch { Log "REFUSED $Tag ${p}: $($_.Exception.Message)"; exit 3 }
        $ids = (Get-Content -LiteralPath (Join-Path $Gates "$p.ids") -Raw).Trim()
        $force = (Get-Content -LiteralPath (Join-Path $Gates "$p.force") -Raw).Trim()
        $dump = Join-Path $Dir "$p.full"
        $a = @('--model', $Gguf, '--tokens', $ids, '--force', $force, '--dump-full-logits', $dump, '--ser', $Ser)
        if ($Table) { $env:STRATA_Q36_SER_TABLE = $Table } else { Remove-Item Env:STRATA_Q36_SER_TABLE -ErrorAction SilentlyContinue }
        $proc = Start-Process -FilePath $Exe -ArgumentList ($a | ForEach-Object { ConvertTo-ProcessArg $_ }) `
            -RedirectStandardOutput (Join-Path $Dir "$p.out") -RedirectStandardError (Join-Path $Dir "$p.err") -NoNewWindow -PassThru
        $null = $proc.Handle
        if (-not $proc.WaitForExit(900 * 1000)) { $null = Stop-ServerTree $proc.Id $proc.StartTime $Exe; Log "TIMEOUT $Tag $p, tree $($proc.Id) stopped"; exit 4 }
        if ($proc.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $dump)) { Log "FAILED $Tag $p exit $($proc.ExitCode)"; exit 5 }
        $cmpOut = Join-Path $Dir "$p.cmp"
        & $Cmp --ref (Join-Path $Gates "$p.llama.full") --test $dump --vocab $Vocab --steps $Steps *> $cmpOut
        $kl = [double]::Parse((Get-Figure $cmpOut 'KL\(ref\|\|test\): mean ([\d.]+)'), $inv)
        $agree = [double]::Parse((Get-Figure $cmpOut 'top-1 agree (\d+)'), $inv)
        $kls += $kl; $tops += $agree / $Steps
        Log "KL $Tag $p $($kl.ToString('0.000000', $inv)) $agree"
        Remove-Item -LiteralPath $dump
    }
    Remove-Item Env:STRATA_Q36_SER_TABLE -ErrorAction SilentlyContinue
    return [pscustomobject]@{
        kl = ($kls | Measure-Object -Average).Average
        top1 = ($tops | Measure-Object -Average).Average
    }
}
if (-not (Test-Path -LiteralPath $csv)) { 'layer,kl_mean,top1' | Set-Content -LiteralPath $csv }
$env:PATH = (Split-Path -Parent $Exe) + ';' + $env:PATH
Log "START $(Get-Date -Format s) ser $Ser table kmin $TableKmin layers $($Layers -join ',') baseline $($Baseline.IsPresent)"
if ($Baseline) {
    $r = Invoke-Prompts 'base' (Join-Path $OutDir 'base') $null
    Add-Content -LiteralPath $csv ('-1,{0},{1}' -f $r.kl.ToString('0.000000', $inv), $r.top1.ToString('0.0000', $inv))
}
foreach ($layer in $Layers) {
    $tag = 'L{0:D2}' -f $layer
    $dir = Join-Path $OutDir $tag
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $table = Join-Path $dir 'table.txt'
    Set-Content -LiteralPath $table -Value "$layer $TableKmin"   # one line: the layer under test
    $r = Invoke-Prompts $tag $dir $table
    Add-Content -LiteralPath $csv ('{0},{1},{2}' -f $layer, $r.kl.ToString('0.000000', $inv), $r.top1.ToString('0.0000', $inv))
    Log "ROW $tag kl $($r.kl.ToString('0.000000', $inv)) top1 $($r.top1.ToString('0.0000', $inv))"
}
Log "END $(Get-Date -Format s)"
