#!/usr/bin/env python3
"""
On-screen joystick that publishes geometry_msgs/Twist on /cmd_vel - the same
topic Nav2 and the real X30 velocity input use.

  left pad    drag: up/down = forward/back (vx), left/right = strafe (vy)
  right bar   drag: left/right = turn (wz)
  keyboard    W/S forward/back, A/D strafe, Q/E turn, SPACE = stop

Publishes only while an input is active, then one zero Twist on release, so it
doesn't fight Nav2 for /cmd_vel while idle.
"""
import tkinter as tk

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

VX_MAX = 0.4
VY_MAX = 0.3
WZ_MAX = 0.5

PAD = 200          # joystick pad size, px
BAR_W, BAR_H = 200, 50
KEY_SCALE = 0.75   # fraction of max speed for keyboard control


class JoystickNode(Node):
    def __init__(self):
        super().__init__('x30_joystick')
        self.declare_parameter('topic', '/cmd_vel')
        self.declare_parameter('publish_rate', 20.0)
        self.pub = self.create_publisher(Twist, self.get_parameter('topic').value, 10)
        self.rate = float(self.get_parameter('publish_rate').value)

    def send(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = vx, vy, wz
        self.pub.publish(msg)


class JoystickGui:
    def __init__(self, node: JoystickNode):
        self.node = node
        self.pad = (0.0, 0.0)   # normalized (-1..1): x right, y up
        self.bar = 0.0          # normalized, +1 = turn left (ccw)
        self.keys = set()
        self.was_active = False

        self.root = tk.Tk()
        self.root.title('X30 joystick')
        self.root.resizable(False, False)

        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10)
        self.pad_canvas = tk.Canvas(frame, width=PAD, height=PAD, bg='#222')
        self.pad_canvas.grid(row=0, column=0, padx=10)
        self.bar_canvas = tk.Canvas(frame, width=BAR_W, height=BAR_H, bg='#222')
        self.bar_canvas.grid(row=0, column=1, padx=10, sticky='s')
        self.label = tk.Label(self.root, text='', font=('monospace', 10))
        self.label.pack(pady=(0, 6))
        tk.Label(self.root, text='pad: move | bar: turn | WASD/QE keys | SPACE stop',
                 font=('monospace', 8)).pack(pady=(0, 8))

        for c, handler in ((self.pad_canvas, self._pad_event), (self.bar_canvas, self._bar_event)):
            c.bind('<Button-1>', handler)
            c.bind('<B1-Motion>', handler)
        self.pad_canvas.bind('<ButtonRelease-1>', lambda e: self._set_pad(0.0, 0.0))
        self.bar_canvas.bind('<ButtonRelease-1>', lambda e: self._set_bar(0.0))
        self.root.bind('<KeyPress>', self._key_down)
        self.root.bind('<KeyRelease>', self._key_up)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

        self._draw()
        self.root.after(int(1000 / node.rate), self._tick)

    # --- input ---------------------------------------------------------------
    @staticmethod
    def _clamp(v):
        return max(-1.0, min(1.0, v))

    def _set_pad(self, x, y):
        self.pad = (self._clamp(x), self._clamp(y))
        self._draw()

    def _set_bar(self, v):
        self.bar = self._clamp(v)
        self._draw()

    def _pad_event(self, e):
        half = PAD / 2
        self._set_pad((e.x - half) / half, (half - e.y) / half)

    def _bar_event(self, e):
        half = BAR_W / 2
        self._set_bar((half - e.x) / half)  # drag left = turn left (+wz)

    def _key_down(self, e):
        k = e.keysym.lower()
        if k == 'space':
            self.keys.clear()
            self._set_pad(0.0, 0.0)
            self._set_bar(0.0)
        else:
            self.keys.add(k)

    def _key_up(self, e):
        self.keys.discard(e.keysym.lower())

    def _command(self):
        x, y = self.pad
        bar = self.bar
        k = self.keys
        x += KEY_SCALE * ((('d' in k) - ('a' in k)))
        y += KEY_SCALE * ((('w' in k) - ('s' in k)))
        bar += KEY_SCALE * ((('q' in k) - ('e' in k)))
        # pad x (right) = strafe right = negative vy in ROS (+y is left)
        return (self._clamp(y) * VX_MAX, -self._clamp(x) * VY_MAX, self._clamp(bar) * WZ_MAX)

    # --- loop / drawing --------------------------------------------------------
    def _tick(self):
        vx, vy, wz = self._command()
        active = any(abs(v) > 1e-3 for v in (vx, vy, wz))
        if active or self.was_active:
            self.node.send(vx, vy, wz)
        self.was_active = active
        self.label.config(text=f'vx {vx:+.2f}  vy {vy:+.2f}  wz {wz:+.2f}')
        rclpy.spin_once(self.node, timeout_sec=0)
        self.root.after(int(1000 / self.node.rate), self._tick)

    def _draw(self):
        c = self.pad_canvas
        c.delete('all')
        h = PAD / 2
        c.create_oval(4, 4, PAD - 4, PAD - 4, outline='#666')
        c.create_line(h, 4, h, PAD - 4, fill='#444')
        c.create_line(4, h, PAD - 4, h, fill='#444')
        sx, sy = h + self.pad[0] * (h - 14), h - self.pad[1] * (h - 14)
        c.create_oval(sx - 14, sy - 14, sx + 14, sy + 14, fill='#4ac', outline='')
        b = self.bar_canvas
        b.delete('all')
        bh = BAR_W / 2
        b.create_line(10, BAR_H / 2, BAR_W - 10, BAR_H / 2, fill='#666')
        bx = bh - self.bar * (bh - 14)
        b.create_oval(bx - 12, BAR_H / 2 - 12, bx + 12, BAR_H / 2 + 12, fill='#ca4', outline='')

    def close(self):
        self.node.send(0.0, 0.0, 0.0)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    node = JoystickNode()
    try:
        JoystickGui(node).run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
