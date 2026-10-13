import sys, time, socket
sys.path.insert(0, "ros2_ws/src/x30_hw")
from x30_hw import protocol as P
from x30_hw.tools.fake_x30 import FakeX30Server
from x30_hw.tools.fake_core import FakeConfig
srv = FakeX30Server(FakeConfig(time_scale=5.0), listen_port=0, control_port=None).start()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dest = ("127.0.0.1", srv.listen_port)
t0 = time.time(); sent = False
while time.time() - t0 < 2.0:
    s.sendto(P.simple_cmd(P.CMD_HEARTBEAT), dest)
    if not sent and time.time() - t0 > 0.4:
        s.sendto(P.simple_cmd(P.CMD_STAND_TOGGLE), dest); sent = True
    time.sleep(0.2)
st = srv.status(); print(st["state_name"], st["stats"], st["violations"])
srv.stop()
