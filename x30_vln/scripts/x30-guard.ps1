<#
  x30-guard.ps1 - memory / GPU guard for the Docker Desktop (WSL2) host.
  Every IntervalS seconds appends one record to results\infra\guard-<date>.log with: docker stats (per container CPU, memory),
  nvidia-smi (GPU used/total, utilisation, temperature) and the WSL VM memory (Windows process vmmemWSL / vmmem).
  When the memory figure exceeds LimitGB it stops the heavy containers named in KillList (docker kill, first match per round,
  largest memory first) and logs that. It never removes containers, images or volumes.

  Memory figure used for the limit: WSL VM used memory (MemTotal - MemAvailable from /proc/meminfo, page cache excluded);
  fallback the Windows vmmemWSL working set (includes cache!), then the sum of container usage. All are logged.
  Parameters (defaults from environment / .env): -LimitGB 21 (GUARD_LIMIT_GB), -IntervalS 10 (GUARD_INTERVAL_S),
  -KillList x30-sim,x30-framework (GUARD_KILL), -Once (one record, then exit), -DryRun (never kill).
#>
param(
    [double]$LimitGB = 0,
    [int]$IntervalS = 0,
    [string[]]$KillList = @(),
    [switch]$Once,
    [switch]$DryRun
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$EnvFile = Join-Path $Root '.env'
function Get-Setting([string]$name, [string]$default) {
    $v = [Environment]::GetEnvironmentVariable($name)
    if (-not $v -and (Test-Path $EnvFile)) {
        $m = Select-String -Path $EnvFile -Pattern ("^" + $name + "=(.+)$") | Select-Object -First 1
        if ($m) { $v = $m.Matches[0].Groups[1].Value.Trim() }
    }
    if ($v) { return $v } else { return $default }
}
if ($LimitGB -le 0) { $LimitGB = [double](Get-Setting 'GUARD_LIMIT_GB' '21') }
if ($IntervalS -le 0) { $IntervalS = [int](Get-Setting 'GUARD_INTERVAL_S' '10') }
if ($KillList.Count -eq 0) { $KillList = (Get-Setting 'GUARD_KILL' 'x30-sim,x30-framework').Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ } }

$LogDir = Join-Path $Root 'results\infra'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Log = Join-Path $LogDir ("guard-" + (Get-Date -Format 'yyyyMMdd') + ".log")

function Write-Log([string]$line) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Add-Content -Path $Log -Value "$ts $line"
    Write-Host "$ts $line"
}

function ConvertTo-GB([string]$text) {
    # "1.5GiB", "512MiB", "700kB", "12.3MB"
    if ($text -match '^([\d\.]+)\s*([A-Za-z]+)$') {
        $n = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
        switch -Regex ($Matches[2]) {
            '^[Tt]i?B$' { return $n * 1024 }
            '^[Gg]i?B$' { return $n }
            '^[Mm]i?B$' { return $n / 1024 }
            '^[Kk]i?B$' { return $n / 1048576 }
            '^B$'       { return $n / 1073741824 }
        }
    }
    return 0
}

Write-Log ("guard start: limit={0} GB interval={1}s kill={2} dryrun={3} log={4}" -f $LimitGB, $IntervalS, ($KillList -join ','), $DryRun.IsPresent, $Log)

while ($true) {
    $containers = @()
    $raw = & docker stats --no-stream --format '{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}' 2>$null
    foreach ($l in $raw) {
        if (-not $l) { continue }
        $p = $l.Split('|')
        $used = ConvertTo-GB (($p[2].Split('/')[0]).Trim())
        $containers += [pscustomobject]@{ Name = $p[0]; Cpu = $p[1]; MemGB = $used }
    }
    $sum = ($containers | Measure-Object MemGB -Sum).Sum
    if (-not $sum) { $sum = 0 }

    $gpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null
    $vm = Get-Process -Name vmmemWSL, vmmem -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
    $vmGB = $null
    if ($vm) { $vmGB = [math]::Round($vm.WorkingSet64 / 1GB, 2) }
    # used = MemTotal - MemAvailable inside the WSL VM: excludes the page cache, which vmmemWSL counts and which makes the
    # Windows-side figure read 10+ GB too high after reading model files
    $usedGB = $null
    $mi = (& wsl.exe -d docker-desktop -e cat /proc/meminfo 2>$null) -join "`n"
    $mi = $mi -replace "`0", ''
    if ($mi -match 'MemTotal:\s+(\d+) kB' ) { $tot = [double]$Matches[1]; if ($mi -match 'MemAvailable:\s+(\d+) kB') { $usedGB = [math]::Round(($tot - [double]$Matches[1]) / 1048576, 2) } }
    $measure = $sum
    if ($vmGB) { $measure = $vmGB }
    if ($usedGB) { $measure = $usedGB }

    $cs = ($containers | Sort-Object MemGB -Descending | ForEach-Object { "{0}={1:N2}GB/{2}" -f $_.Name, $_.MemGB, $_.Cpu }) -join ' '
    Write-Log ("wsl_used_gb={0} wsl_vmmem_gb={1} containers_sum_gb={2:N2} gpu[used_MiB,total_MiB,util%,C]={3} | {4}" -f $usedGB, $vmGB, $sum, ($gpu -join ';'), $cs)

    if ($measure -gt $LimitGB) {
        $victims = $containers | Where-Object { $KillList -contains $_.Name } | Sort-Object MemGB -Descending
        if (-not $victims) {
            Write-Log ("LIMIT EXCEEDED ({0:N1} > {1} GB) but none of [{2}] is running; nothing to stop" -f $measure, $LimitGB, ($KillList -join ','))
        } elseif ($DryRun) {
            Write-Log ("LIMIT EXCEEDED ({0:N1} > {1} GB): dry run, would kill {2}" -f $measure, $LimitGB, $victims[0].Name)
        } else {
            $v = $victims[0].Name
            Write-Log ("LIMIT EXCEEDED ({0:N1} > {1} GB): docker kill {2} ({3:N2} GB)" -f $measure, $LimitGB, $v, $victims[0].MemGB)
            & docker kill $v 2>&1 | ForEach-Object { Write-Log "docker kill: $_" }
        }
    }
    if ($Once) { break }
    Start-Sleep -Seconds $IntervalS
}
