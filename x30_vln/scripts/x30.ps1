<#
  x30.ps1 - launcher for the X30 VLN docker stack (Windows PowerShell 5.1 and later). Every subcommand prints the docker command it runs.
  Never runs a docker prune / volume or image removal.

  Usage:  powershell -ExecutionPolicy Bypass -File scripts\x30.ps1 <command> [args]
          (or dot-free alias:  Set-Alias x30 "<path>\scripts\x30.ps1")

  docker-start                      start Docker Desktop via docker-clean-start.ps1 (stale-socket workaround) if the engine is down
  up <sim|real|replay|models>       start a profile (flags: -TwoVlm  -HostNet  -Build)      e.g.  x30.ps1 up models -TwoVlm
  down [profile]                    stop and remove the containers of a profile (default: ALL x30 services incl. the models; volumes are kept)
  status                            containers, GPU memory, WSL memory
  shell <service>                   interactive shell in x30-<service> (framework, tools, vlm, perception, sim, ros1_bridge ...)
  exec <service> <command...>       run a command in x30-<service>
  build [pkg ...]                   colcon build in the framework image (named volumes); no args = all packages except $X30_SKIP_PKGS
  test [pkg ...]                    colcon test in the framework image
  rosdep [--check|--dry-run]        resolve package.xml dependencies in the framework image
  images [framework|perception|ros1_bridge ...]   build docker images (takes the shared build lock)
  logs <service> [-Tail n]          follow logs of x30-<service>
  guard [start|stop|run]            memory/GPU guard (run = foreground, default; start = background; logs in results\infra\)
  replay <bagdir>                   play a rosbag2 directory (inside the project) with the replay profile
#>
param(
    [Parameter(Position = 0)][string]$Command = 'help',
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest,
    [switch]$TwoVlm,
    [switch]$HostNet,
    [switch]$Build,
    [int]$Tail = 100
)

$ErrorActionPreference = 'Continue'
$Root      = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$DockerDir = Join-Path $Root 'docker'
$EnvFile   = Join-Path $Root '.env'
$LockDir   = Join-Path $Root '.locks\docker-build'
$CleanStart = 'C:\Users\LAB USER 1\Documents\ros2-docker-setup\docker-clean-start.ps1'

function Show([string]$text) { Write-Host (">> " + $text) -ForegroundColor Cyan }

function Show-Help {
    $txt = Get-Content $PSCommandPath -Raw
    $i = $txt.IndexOf('<#') + 2; $j = $txt.IndexOf('#>')
    Write-Host $txt.Substring($i, $j - $i)
}

function Test-Engine {
    $job = Start-Job { docker version --format '{{.Server.Version}}' 2>$null }
    $done = Wait-Job $job -Timeout 15
    $v = $null
    if ($done) { $v = Receive-Job $job }
    Remove-Job $job -Force
    return [bool]$v
}

function Get-ComposeArgs([string]$Profile) {
    $a = @('compose')
    if (Test-Path $EnvFile) { $a += @('--env-file', $EnvFile) }
    $a += @('-f', (Join-Path $DockerDir 'compose.yml'), '-f', (Join-Path $DockerDir 'compose.models.yml'))
    $sim = Join-Path $DockerDir 'compose.sim.yml'
    if (Test-Path $sim) { $a += @('-f', $sim) }
    if ($HostNet) { $a += @('-f', (Join-Path $DockerDir 'compose.real.yml')) }
    if ($Profile) { foreach ($p in $Profile.Split(',')) { $a += @('--profile', $p) } }
    return $a
}

function Invoke-Docker([string[]]$DockerArgs) {
    Show ("docker " + ($DockerArgs -join ' '))
    & docker @DockerArgs | Out-Host
    return $LASTEXITCODE
}

function Require-Engine {
    if (-not (Test-Engine)) {
        Write-Warning "Docker engine does not answer. Run: scripts\x30.ps1 docker-start"
        exit 2
    }
}

function Get-ServiceName([string]$name) {
    if (-not $name) { Write-Warning 'service name required'; exit 2 }
    if ($name.StartsWith('x30-')) { return $name }
    return "x30-$name"
}

function Take-BuildLock {
    $deadline = (Get-Date).AddMinutes(40)
    New-Item -ItemType Directory -Path (Split-Path $LockDir) -Force | Out-Null
    while ($true) {
        & cmd /c "mkdir `"$LockDir`" 2>nul"
        if ($LASTEXITCODE -eq 0) { Show "build lock taken: $LockDir"; return }
        $age = (Get-Date) - (Get-Item $LockDir).CreationTime
        if ($age.TotalMinutes -gt 90) {
            Write-Warning "lock older than 90 min ($([int]$age.TotalMinutes)); removing it as allowed by the brief"
            Remove-Item $LockDir -Force -Recurse
            continue
        }
        if ((Get-Date) -gt $deadline) { Write-Warning 'build lock still held after 40 min'; exit 3 }
        Write-Host "build lock held by someone else (age $([int]$age.TotalMinutes) min); waiting 60 s ..."
        Start-Sleep -Seconds 60
    }
}

function Release-BuildLock { if (Test-Path $LockDir) { Remove-Item $LockDir -Force -Recurse; Show 'build lock released' } }

function Get-ProfileForUp([string]$p) {
    switch ($p) {
        'sim'    { return 'sim' }
        'real'   { return 'real' }
        'replay' { return 'replay' }
        'models' { return 'models' }
        default  { Write-Warning "unknown profile '$p' (sim|real|replay|models)"; exit 2 }
    }
}

function Run-InFramework([string[]]$Cmd) {
    Require-Engine
    $c = Get-ComposeArgs 'sim'
    $c += @('run', '--rm', '--no-deps', '-T', 'framework') + $Cmd
    exit (Invoke-Docker $c)
}

switch ($Command) {
    'help' { Show-Help }

    'docker-start' {
        if (Test-Engine) { Write-Host 'Docker engine is already up.'; break }
        Show "powershell -ExecutionPolicy Bypass -File `"$CleanStart`""
        & powershell -ExecutionPolicy Bypass -File $CleanStart
    }

    'up' {
        Require-Engine
        $prof = Get-ProfileForUp $Rest[0]
        $profiles = $prof
        if ($TwoVlm) { $profiles += ',two-vlm' }
        if ($prof -eq 'real') {
            $uri = $null
            if (Test-Path $EnvFile) { $uri = (Select-String -Path $EnvFile -Pattern '^ROS_MASTER_URI=(.+)$' | Select-Object -First 1) }
            if (-not $uri) { Write-Warning 'ROS_MASTER_URI is not set in .env; the bridge falls back to localhost. Copy .env.example to .env and set the robot addresses.' }
            if (-not $HostNet) { Write-Warning 'real profile without -HostNet: the robot cannot call back into a NATed container. Fine for bag tests; see docs\INFRA.md (Real robot networking).' }
        }
        if ($prof -eq 'sim' -and -not (Test-Path (Join-Path $DockerDir 'compose.sim.yml'))) {
            Write-Warning 'docker\compose.sim.yml does not exist yet: starting framework/tools/models without the simulator.'
        }
        $env:X30_PROFILE = $prof
        $c = Get-ComposeArgs $profiles
        $c += @('up', '-d')
        if ($Build) { $c += '--build' }
        exit (Invoke-Docker $c)
    }

    'down' {
        Require-Engine
        $dp = 'sim,real,replay,models,two-vlm'
        if ($Rest -and $Rest[0]) { $dp = $Rest[0] }
        $c = Get-ComposeArgs $dp
        $c += @('down')
        exit (Invoke-Docker $c)
    }

    'status' {
        Require-Engine
        Invoke-Docker @('ps', '-a', '--filter', 'name=x30-', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}') | Out-Null
        Invoke-Docker @('stats', '--no-stream', '--format', 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}') | Out-Null
        Show 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
        & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
        $vm = Get-Process -Name vmmemWSL, vmmem -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
        if ($vm) { Write-Host ("WSL VM memory ({0}): {1:N1} GB of 24 GB cap" -f $vm.Name, ($vm.WorkingSet64 / 1GB)) }
    }

    'shell' {
        Require-Engine
        $n = Get-ServiceName $Rest[0]
        Show "docker exec -it $n bash -l  (falls back to sh)"
        & docker exec -it $n bash -l
        if ($LASTEXITCODE -ne 0) { & docker exec -it $n sh }
    }

    'exec' {
        Require-Engine
        $n = Get-ServiceName $Rest[0]
        $cmd = @()
        if ($Rest.Count -gt 1) { $cmd = $Rest[1..($Rest.Count - 1)] }
        exit (Invoke-Docker (@('exec', '-i', $n, 'bash', '-lc', ($cmd -join ' '))))
    }

    'build'  { Run-InFramework (@('x30-build') + $Rest) }
    'test'   { Run-InFramework (@('x30-test') + $Rest) }
    'rosdep' { Run-InFramework (@('x30-rosdep') + $Rest) }

    'images' {
        Require-Engine
        $targets = @($Rest)
        if ($targets.Count -eq 0) { $targets = @('framework', 'ros1_bridge', 'perception') }
        Take-BuildLock
        try {
            $env:DOCKER_BUILDKIT = '1'
            $c = Get-ComposeArgs 'sim,real,models'
            $c += @('build') + $targets
            $rc = Invoke-Docker $c
        } finally { Release-BuildLock }
        exit $rc
    }

    'logs' {
        Require-Engine
        $n = Get-ServiceName $Rest[0]
        exit (Invoke-Docker @('logs', '-f', '--tail', "$Tail", $n))
    }

    'guard' {
        $sub = 'run'
        if ($Rest.Count -gt 0) { $sub = $Rest[0] }
        $g = Join-Path $PSScriptRoot 'x30-guard.ps1'
        $pidFile = Join-Path $Root 'results\infra\guard.pid'
        switch ($sub) {
            'run'   { Show "powershell -File $g"; & powershell -ExecutionPolicy Bypass -File $g }
            'start' {
                New-Item -ItemType Directory -Path (Split-Path $pidFile) -Force | Out-Null
                Show "Start-Process powershell -File $g (hidden)"
                $p = Start-Process powershell -ArgumentList @('-ExecutionPolicy', 'Bypass', '-File', "`"$g`"") -WindowStyle Hidden -PassThru
                $p.Id | Set-Content $pidFile
                Write-Host "guard running, pid $($p.Id); log in results\infra\guard-*.log"
            }
            'stop'  {
                if (Test-Path $pidFile) { Stop-Process -Id ([int](Get-Content $pidFile)) -Force -ErrorAction SilentlyContinue; Remove-Item $pidFile; Write-Host 'guard stopped' }
                else { Write-Host 'no guard pid file' }
            }
            default { Write-Warning 'guard [run|start|stop]' }
        }
    }

    'replay' {
        Require-Engine
        if (-not $Rest -or -not $Rest[0]) { Write-Warning 'usage: x30.ps1 replay <bagdir inside the project>'; exit 2 }
        $full = (Resolve-Path $Rest[0]).Path
        if (-not $full.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) { Write-Warning "bag directory must be inside $Root (mounted as /x30)"; exit 2 }
        $rel = $full.Substring($Root.Length).TrimStart('\').Replace('\', '/')
        $env:REPLAY_BAG = "/x30/$rel"
        Show "REPLAY_BAG=$env:REPLAY_BAG"
        $env:X30_PROFILE = 'replay'
        $c = Get-ComposeArgs 'replay'
        $c += @('up', '-d', '--force-recreate', 'replay')
        exit (Invoke-Docker $c)
    }

    default { Write-Warning "unknown command '$Command'"; Show-Help; exit 2 }
}
