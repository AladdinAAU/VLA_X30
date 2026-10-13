"""teach node: the learner phase. Service /teach/say {"text": "this is Marcus"} (x30_msgs/srv/Json).

What happens on a call:
  1. where: the position the person talks about = the person followed by follow_person (/follow/status, x y in the map frame, not older than
     max_person_age_s), else the robot's position (TF map -> base_link), else {"x", "y"} given in the request (these two override: "at": "robot")
  2. context from the semantic mapper (/semantic_map/place_context): the region at that position (room / corridor / hall, area), objects nearby,
     taught places nearby
  3. the language model (reasoner role of configs/models.yaml, Qwen3-VL-8B by default) picks ONE tool: save_place / rename_place / forget_place / ignore
     (interpreter.py)
  4. the result is applied to the semantic map (/semantic_map/add_place, remove_place): a room place names the region the person stands in, a spot
     stores a point; places are saved with the map and appear in the reasoner's map summary ('Places:')
  5. a short deterministic confirmation is returned and published on /teach/reply; every call is appended to <log_dir>/teach_<date>.jsonl
Response JSON: {"ok", "action": save|rename|forget|ignore|error, "reply", "interpretation", "applied", "context", "position", "position_source", "latency_s"}.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from x30_common.vlm_client import VLMClient
from x30_msgs.srv import Json

from . import interpreter as itp


class Teach(Node):
    def __init__(self):
        super().__init__('teach')
        d = self.declare_parameter
        d('map_frame', 'map'), d('base_frame', 'base_link'), d('status_topic', '/follow/status'), d('max_person_age_s', 3.0)
        d('log_dir', '/x30/results/teach'), d('context_radius_m', 4.0), d('mapper', '/semantic_map'), d('model_role', 'reasoner'), d('llm_timeout_s', 30.0)
        g = self.get_parameter
        self.P = {k: g(k).value for k in ('map_frame', 'base_frame', 'status_topic', 'max_person_age_s', 'log_dir', 'context_radius_m', 'mapper', 'model_role', 'llm_timeout_s')}
        self.client = VLMClient.from_config(self.P['model_role'], max_tokens=200, temperature=0.0, timeout_s=float(self.P['llm_timeout_s']))
        self.cb = ReentrantCallbackGroup()
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.person = None                                  # (sim time, x, y) of the followed person
        self.last_place = None
        self.mx = threading.Lock()
        self.create_subscription(String, self.P['status_topic'], self.on_status, 10)
        self.reply_pub = self.create_publisher(String, '/teach/reply', 10)
        self.cli = {n: self.create_client(Json, f"{self.P['mapper']}/{n}", callback_group=self.cb) for n in ('place_context', 'add_place', 'remove_place', 'places')}
        self.create_service(Json, '/teach/say', self.svc_say, callback_group=self.cb)
        self.get_logger().info(f"teach up: model {self.client.model} at {self.client.base_url}, mapper {self.P['mapper']}")

    # ------------------------------------------------------------------ helpers
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_status(self, m: String) -> None:
        try:
            s = json.loads(m.data)
        except ValueError:
            return
        if s.get('state') == 'follow' and 'x' in s:
            self.person = (self.now(), float(s['x']), float(s['y']))

    def robot_xy(self):
        try:
            tf = self.buf.lookup_transform(self.P['map_frame'], self.P['base_frame'], rclpy.time.Time())
            return tf.transform.translation.x, tf.transform.translation.y
        except Exception:
            return None

    def where(self, body: dict):
        if body.get('x') is not None and body.get('y') is not None:
            return float(body['x']), float(body['y']), 'request'
        if body.get('at') != 'robot' and self.person is not None and self.now() - self.person[0] <= float(self.P['max_person_age_s']):
            return self.person[1], self.person[2], 'person'
        r = self.robot_xy()
        return (r[0], r[1], 'robot') if r else (None, None, 'none')

    def call(self, name: str, req: dict, timeout: float = 20.0) -> dict:
        cli = self.cli[name]
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f'mapper service {name} not available')
        fut = cli.call_async(Json.Request(request=json.dumps(req)))
        ev = threading.Event()
        fut.add_done_callback(lambda _: ev.set())
        if not ev.wait(timeout):
            raise RuntimeError(f'mapper service {name} timed out')
        r = fut.result()
        out = json.loads(r.response) if r.response else {}
        if not r.ok:
            raise RuntimeError(f"{name}: {out.get('error', 'failed')}")
        return out

    def chat(self, messages, tools):
        return self.client.chat(messages, tools=tools, tool_choice='required', max_tokens=200, temperature=0.0)

    # ------------------------------------------------------------------ service
    def svc_say(self, req, resp):
        t0 = time.time()
        try:
            body = json.loads(req.request) if req.request else {}
        except ValueError:
            body = {}
        text = str(body.get('text', '')).strip()
        out = {'ok': False, 'action': 'error', 'reply': '', 'text': text}
        try:
            if not text:
                raise ValueError('empty text')
            x, y, src = self.where(body)
            if x is None:
                raise RuntimeError('no position: no followed person, no robot pose')
            out.update(position=[round(x, 2), round(y, 2)], position_source=src)
            ctx = self.call('place_context', {'x': x, 'y': y, 'radius': self.P['context_radius_m']})
            out['context'] = ctx
            with self.mx:
                last = self.last_place
            act = itp.interpret(self.chat, text, ctx, last)
            out['interpretation'] = vars(act)
            out['action'] = act.kind
            if act.kind == 'save':
                aliases = list(act.aliases)
                if act.owner and act.owner.lower() != act.name.lower():
                    aliases.append(act.owner)
                r = self.call('add_place', {'name': act.name, 'kind': act.place_kind, 'x': x, 'y': y, 'owner': act.owner,
                                            'function': act.function, 'aliases': aliases, 'utterance': text})
                out['applied'] = r
                pl = r['place']
                where = 'the room you are in' if pl['kind'] == 'room' else 'a spot here'
                out['reply'] = (f"{'Updated' if r['action'] == 'updated' else 'Saved'} '{pl['name']}' as {where}"
                                + (f" ({pl['area_m2']:.0f} m2)" if pl['kind'] == 'room' and pl['area_m2'] else '')
                                + (f", owner {pl['owner']}" if pl['owner'] else '') + (f", {pl['function']}" if pl['function'] else '') + '.'
                                + (f" Note: {r['note']}." if r.get('note') else ''))
                with self.mx:
                    self.last_place = pl['name']
                out['ok'] = True
            elif act.kind == 'rename':
                places = self.call('places', {})['places']
                old = next((p for p in places if p['name'].lower() == act.old_name.lower() or act.old_name.lower() in [a.lower() for a in p['aliases']]), None)
                if old is None:
                    out['reply'] = f"I have no place called '{act.old_name}'."
                else:
                    self.call('remove_place', {'id': old['id']})
                    owner = act.new_name if old['owner'] and old['owner'].lower() == old['name'].lower() else old['owner']
                    r = self.call('add_place', {'name': act.new_name, 'kind': old['kind'], 'x': old['x'], 'y': old['y'], 'owner': owner,
                                                'function': old['function'], 'aliases': [], 'utterance': text})
                    out['applied'] = r
                    out['reply'] = f"Renamed '{old['name']}' to '{act.new_name}'."
                    with self.mx:
                        self.last_place = act.new_name
                    out['ok'] = True
            elif act.kind == 'forget':
                r = self.call('remove_place', {'name': act.name})
                out['applied'] = r
                out['ok'] = bool(r.get('ok'))
                out['reply'] = f"Forgot '{act.name}'." if r.get('ok') else f"I have no place called '{act.name}'."
            elif act.kind == 'ignore':
                out['ok'] = True
                out['reply'] = ''
            else:
                out['reply'] = "Sorry, I did not understand that."
                out['error'] = act.reason
        except Exception as ex:
            self.get_logger().warning(f'/teach/say failed: {type(ex).__name__}: {ex}')
            out['error'] = f'{type(ex).__name__}: {ex}'
            out['reply'] = 'Sorry, I could not process that.'
        out['latency_s'] = round(time.time() - t0, 2)
        self.log(out)
        if out.get('reply'):
            self.reply_pub.publish(String(data=json.dumps({'reply': out['reply'], 'action': out['action']})))
        resp.ok, resp.response = bool(out['ok']), json.dumps(out)
        return resp

    def log(self, out: dict) -> None:
        try:
            os.makedirs(self.P['log_dir'], exist_ok=True)
            with open(os.path.join(self.P['log_dir'], f"teach_{time.strftime('%Y%m%d')}.jsonl"), 'a', encoding='utf-8') as f:
                f.write(json.dumps({'wall': time.time(), **out}) + '\n')
        except OSError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = Teach()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
