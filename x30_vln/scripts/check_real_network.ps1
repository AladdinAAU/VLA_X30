<#
  check_real_network.ps1 - reachability checks for the REAL ROBOT network. FOR THE USER TO RUN, later, with the robot powered
  on and connected. It was written and parse-checked but deliberately never executed by the infra agent.

  Only passive tests: ICMP ping, one TCP connect to the ROS 1 master port (11311, no data sent), a listing of local interfaces
  so you can pick ROS_IP, and a note on UDP. It never sends ROS or UDP control traffic to the robot.

  Usage:  powershell -ExecutionPolicy Bypass -File scripts\check_real_network.ps1 -MasterHost 192.168.1.106 [-LocalIp 192.168.1.131]
#>
param(
    [Parameter(Mandatory = $true)][string]$MasterHost,
    [int]$MasterPort = 11311,
    [string]$LocalIp = ''
)

Write-Host '== 1. local IPv4 addresses (pick the one on the robot network for ROS_IP) =='
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -ne '127.0.0.1' } |
    Select-Object InterfaceAlias, IPAddress, PrefixLength | Format-Table -AutoSize

Write-Host "== 2. ping $MasterHost (4 packets) =="
$ping = Test-Connection -ComputerName $MasterHost -Count 4 -ErrorAction SilentlyContinue
if ($ping) { $ping | Select-Object Address, ResponseTime | Format-Table -AutoSize } else { Write-Warning 'no ICMP reply (the robot may block ping; continue with the TCP test)' }

Write-Host "== 3. TCP connect to ROS master ${MasterHost}:$MasterPort =="
$client = New-Object System.Net.Sockets.TcpClient
$task = $client.ConnectAsync($MasterHost, $MasterPort)
if ($task.Wait(3000) -and $client.Connected) { Write-Host 'OK: master port reachable' -ForegroundColor Green } else { Write-Warning 'FAILED: master port not reachable (roscore not running, wrong IP, or firewall)' }
$client.Close()

Write-Host '== 4. what this does NOT prove =='
Write-Host @'
 - ROS 1 is a callback protocol: after you reach the master, the master and every publisher open NEW connections BACK to
   ROS_IP:<random port>. That works only if the robot can reach the address in ROS_IP directly (no NAT in between).
   With Docker Desktop's default bridge/NAT it will NOT. Use a Linux host with network_mode: host, or the WSL2 mirrored-networking
   route described in docs\INFRA.md ("Real robot networking").
 - Windows Defender Firewall / Hyper-V firewall must allow inbound TCP from the robot subnet to the bridge (random high ports)
   and, for DDS between machines, UDP 7400-7500 (+ ephemeral). The Hyper-V firewall rule for WSL is in docs\INFRA.md.
 - UDP reachability cannot be tested without sending UDP packets; to avoid touching the robot, test inbound UDP from another
   PC on the same switch with: (receiver here)  python -c "import socket;s=socket.socket(2,2);s.bind(('0.0.0.0',7400));print(s.recvfrom(100))"
'@
if ($LocalIp) { Write-Host "Reminder: set ROS_IP=$LocalIp and ROS_MASTER_URI=http://${MasterHost}:$MasterPort in .env" }
