"""A minimal round-robin reverse proxy over several Ollama servers.

On a shared machine one Ollama server's slot loop is CPU-bound well before the
GPU is, so exp18 fans LOTUS's requests out over two or three server instances
(same model, same weights) behind this proxy.  Both arms of exp18 go through
it, so the comparison is unaffected; absolute latencies are those of this
deployment.

With --max-inflight K each backend receives at most K requests at a time
and a request goes to the least loaded backend; excess requests wait here
instead of in Ollama's own queue.  The reported exp18 run does not use it
(default 0, plain round robin): a run with K = 8 stalled after eleven minutes
(see README.md, exp18), and plain round robin is the configuration
that ran for eight hours on 2026-09-29.

Usage:
    .venv/bin/python experiments/rr_proxy.py --port 11500 \
        --backends http://127.0.0.1:11435,http://127.0.0.1:11437
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
       "te", "trailers", "transfer-encoding", "upgrade", "content-length"}


def make_handler(backends: list[str], max_inflight: int = 0):
    ring = itertools.cycle(backends)
    lock = threading.Lock()
    free = threading.Condition(lock)
    inflight = dict.fromkeys(backends, 0)

    def acquire() -> str:
        with free:
            if max_inflight <= 0:
                return next(ring)
            while True:
                b = min(backends, key=inflight.__getitem__)
                if inflight[b] < max_inflight:
                    inflight[b] += 1
                    return b
                free.wait()

    def release(b: str) -> None:
        if max_inflight > 0:
            with free:
                inflight[b] -= 1
                free.notify()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _forward(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in HOP and k.lower() != "host"}
            b = acquire()
            try:
                base = urllib.parse.urlparse(b)
                conn = http.client.HTTPConnection(base.hostname, base.port,
                                                  timeout=600)
                conn.request(self.command, self.path, body=body,
                             headers=headers)
                resp = conn.getresponse()
                data = resp.read()
            finally:
                release(b)
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            conn.close()

        do_GET = do_POST = do_HEAD = _forward

        def log_message(self, *args):          # quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=11500)
    ap.add_argument("--backends", required=True,
                    help="comma-separated Ollama base URLs")
    ap.add_argument("--max-inflight", type=int, default=0,
                    help="requests in flight per backend (0 = unlimited)")
    args = ap.parse_args()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              make_handler(backends, args.max_inflight))
    srv.daemon_threads = True
    print(f"[proxy] :{args.port} -> {backends}, max in flight per backend "
          f"{args.max_inflight or 'unlimited'}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
