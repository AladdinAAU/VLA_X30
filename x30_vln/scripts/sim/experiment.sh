#!/usr/bin/env bash
# Start the simulator with a modified copy of robot_sim.yaml, measure RTF/CPU/RAM, stop it. Run inside the sim container.
#   experiment.sh NAME 'python statement editing dict c' [launch args...] [--seconds N]
# e.g. experiment.sh nocam "c['camera']['width']=640;c['camera']['height']=480;c['thermal']['enabled']=False" locomotion:=kinematic
set -u
name=$1; edit=$2; shift 2
secs=${SECONDS_MEASURE:-60}
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$here/kill_sim.sh" >/dev/null 2>&1
mkdir -p /tmp/exp
python3 - "$name" "$edit" <<'PY'
import sys, yaml
name, edit = sys.argv[1], sys.argv[2]
c = yaml.safe_load(open('/x30/configs/robot_sim.yaml'))
exec(edit)
yaml.safe_dump(c, open(f'/tmp/exp/{name}.yaml', 'w'))
PY
nohup ros2 launch x30_sim sim.launch.py robot_config:=/tmp/exp/$name.yaml "$@" > /tmp/exp/$name.launch.log 2>&1 &
sleep "${WARMUP:-40}"
python3 "$here/measure_rtf.py" --seconds "$secs" --interval 10 --csv /tmp/exp/$name.csv | tail -8
bash "$here/kill_sim.sh" >/dev/null 2>&1
