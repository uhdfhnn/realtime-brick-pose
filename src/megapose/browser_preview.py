"""Localhost-only latest-image HTTP preview, without GUI mounts or extra packages."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b'''<!doctype html><html><head><title>DUPLO live inspection</title>
<style>body{background:#171717;color:#eee;font:18px sans-serif;margin:24px}img{max-width:100%;border:1px solid #555}p{max-width:960px}</style></head>
<body><h2>DUPLO live inspection</h2><p id="status">Connecting...</p>
<img id="frame"><p>Blue = 2x4, red = 2x2 by configuration. Color-based detection can mistake other objects for bricks. A high pose score does not verify accuracy. Pixel grid is not a calibrated table grid.</p>
<script>const img=document.getElementById('frame');
function next(){img.src='/frame.jpg?t='+Date.now()}
img.onload=()=>setTimeout(next,100);img.onerror=()=>setTimeout(next,500);next();
async function status(){try{const s=await(await fetch('/status',{cache:'no-store'})).json();document.getElementById('status').textContent=s.message+(s.image_age_s==null?'':' | displayed image updated '+s.image_age_s.toFixed(1)+' s ago');}catch(e){document.getElementById('status').textContent='Disconnected - displayed image is stale';}setTimeout(status,500)}status();</script></body></html>'''


class BrowserPreview:
    def __init__(self, port=8090):
        self._lock = threading.Lock()
        self._jpeg = None
        self._updated = None
        self._message = 'Starting'
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = self.path.split('?', 1)[0]
                with owner._lock:
                    jpeg, updated, message = owner._jpeg, owner._updated, owner._message
                if path == '/':
                    payload, mime, code = PAGE, 'text/html', 200
                elif path == '/status':
                    payload = json.dumps(dict(message=message, image_age_s=None if updated is None
                                              else time.monotonic()-updated)).encode()
                    mime, code = 'application/json', 200
                elif path == '/frame.jpg' and jpeg is not None:
                    payload, mime, code = jpeg, 'image/jpeg', 200
                else:
                    payload, mime, code = b'No image available', 'text/plain', 404
                try:
                    self.send_response(code)
                    self.send_header('Content-Type', mime)
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def status(self, message):
        with self._lock:
            self._message = message

    def publish(self, bgr, message):
        ok, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError('JPEG preview encoding failed')
        with self._lock:
            self._jpeg = jpeg.tobytes()
            self._updated = time.monotonic()
            self._message = message

    def close(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
