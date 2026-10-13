"""A tiny OpenAI-compatible chat server for the annotator tests (standard library only)."""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


class FakeVLM:
    def __init__(self, responder: Callable[[str, int, dict], str | None] | None = None):
        self.calls: list[dict[str, Any]] = []
        self.responder = responder or self.default_responder
        self.delay = 0.0                       # seconds before answering
        self.fail_status = 0                   # answer every request with this HTTP status
        self.reject_response_format = False    # answer 400 when response_format is sent
        self.garbage_first = 0                 # number of initial answers that are not valid JSON
        self.same_object = True
        self.crop_match = True
        self.hallucinate = False
        self._srv: ThreadingHTTPServer | None = None
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ default behaviour per task
    def default_responder(self, text: str, n_images: int, body: dict) -> str | None:
        if 'crop of one object' in text:
            label = re.search(r'called it "([^"]*)"', text).group(1)
            return json.dumps({'caption': f'a {label} in the room', 'label_matches': True, 'label': label,
                               'colour': 'red', 'material': 'plastic', 'state': 'closed'})
        if 'An area of the map' in text:
            return json.dumps({'room_type': 'kitchen' if 'refrigerator' in text else 'corridor', 'confidence': 0.9,
                               'reason': 'objects'})
        if 'same physical object' in text:
            return json.dumps({'same_object': self.same_object, 'confidence': 0.9})
        if text.startswith('Request:'):
            req = re.search(r'Request: "([^"]*)"', text).group(1).lower().split()
            rows = [ln for ln in text.splitlines() if re.match(r'^\d+ \|', ln)]
            hits = [int(r.split(' | ')[0]) for r in rows if any(w in r.lower() for w in req if len(w) > 3)]
            ranking = [{'id': i, 'score': round(0.9 - 0.1 * k, 2)} for k, i in enumerate(hits)]
            if self.hallucinate:
                ranking.insert(0, {'id': 9999, 'score': 1.0})
            return json.dumps({'ranking': ranking, 'reason': 'matched words'})
        if text.startswith('Description:'):
            return json.dumps({'match': self.crop_match, 'confidence': 0.8})
        return '{}'

    # ------------------------------------------------------------------ server
    def start(self) -> 'FakeVLM':
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                content = body['messages'][-1]['content']
                text = ' '.join(p['text'] for p in content if p.get('type') == 'text') if isinstance(content, list) else content
                n_images = sum(1 for p in content if isinstance(p, dict) and p.get('type') == 'image_url') if isinstance(content, list) else 0
                with fake.lock:
                    fake.calls.append({'text': text, 'n_images': n_images, 'body': body, 't': time.time()})
                    idx = len(fake.calls)
                if fake.delay:
                    time.sleep(fake.delay)
                status = fake.fail_status
                if fake.reject_response_format and 'response_format' in body:
                    status = 400
                if status:
                    data = json.dumps({'error': {'message': f'fake failure {status}'}}).encode()
                    self.send_response(status)
                else:
                    answer = 'I think this is a nice picture.' if idx <= fake.garbage_first else fake.responder(text, n_images, body)
                    data = json.dumps({'choices': [{'index': 0, 'finish_reason': 'stop',
                                                    'message': {'role': 'assistant', 'content': answer}}],
                                       'usage': {'prompt_tokens': 100 + 50 * n_images, 'completion_tokens': 20}}).encode()
                    self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self._srv.server_address[1]}/v1'

    def stop(self) -> None:
        if self._srv is not None:
            self._srv.shutdown()
            self._srv.server_close()
            self._srv = None
