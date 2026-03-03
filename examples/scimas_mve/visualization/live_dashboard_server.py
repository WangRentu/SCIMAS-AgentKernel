from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from examples.scimas_mve.visualization.trend_dashboard import build_dashboard_payload


def _json_bytes(data: Dict[str, Any]) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _collect_gpu_metrics() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=1.2).strip()
        if not out:
            return {"available": False, "reason": "empty_nvidia_smi"}
        line = out.splitlines()[0]
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3:
            return {"available": False, "reason": "bad_nvidia_smi_format"}
        util = float(parts[0])
        used = float(parts[1])
        total = float(parts[2]) if float(parts[2]) > 0 else 0.0
        ratio = (used / total) if total > 0 else 0.0
        return {
            "available": True,
            "util_percent": util,
            "memory_used_mb": used,
            "memory_total_mb": total,
            "memory_ratio": ratio,
        }
    except Exception:
        return {"available": False, "reason": "nvidia_smi_unavailable"}


class LiveDashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass,
        *,
        base_dir: str,
        static_dir: str,
        sse_interval_s: float,
    ):
        super().__init__(server_address, RequestHandlerClass)
        self.base_dir = base_dir
        self.static_dir = static_dir
        self.sse_interval_s = sse_interval_s
        self.snapshot_seq = 0

    def build_snapshot(self) -> Tuple[Dict[str, Any], str]:
        payload = build_dashboard_payload(base_dir=self.base_dir)
        self.snapshot_seq += 1
        payload["server"] = {
            "seq": self.snapshot_seq,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": _collect_gpu_metrics(),
        }
        body = _json_bytes(payload)
        sig = hashlib.sha1(body).hexdigest()
        return payload, sig


class Handler(BaseHTTPRequestHandler):
    server: LiveDashboardServer

    def log_message(self, fmt: str, *args):
        print(f"[live-dashboard] {self.address_string()} - {fmt % args}")

    def _send_json(self, code: int, data: Dict[str, Any]) -> None:
        body = _json_bytes(data)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel_path: str, content_type: str) -> None:
        safe_path = os.path.normpath(rel_path).lstrip(os.sep)
        path = os.path.join(self.server.static_dir, safe_path)
        if not os.path.isfile(path):
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"file_not_found:{rel_path}"})
            return
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_sig = ""
        try:
            while True:
                payload, sig = self.server.build_snapshot()
                if sig != last_sig:
                    data = json.dumps(payload, ensure_ascii=False)
                    self.wfile.write(b"event: snapshot\n")
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_sig = sig
                else:
                    self.wfile.write(b"event: heartbeat\ndata: {}\n\n")
                    self.wfile.flush()
                time.sleep(self.server.sse_interval_s)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self):
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self._send_file("index.html", "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._send_file("app.js", "application/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            self._send_file("styles.css", "text/css; charset=utf-8")
            return
        if path == "/api/snapshot":
            payload, _ = self.server.build_snapshot()
            self._send_json(HTTPStatus.OK, payload)
            return
        if path == "/api/events":
            self._serve_sse()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown_path:{path}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="SCIMAS live dashboard server (frontend/backend separated via SSE).")
    parser.add_argument("--host", default=os.getenv("SCIMAS_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SCIMAS_DASHBOARD_PORT", "8787")))
    parser.add_argument(
        "--base-dir",
        default=os.getenv(
            "SCIMAS_DASHBOARD_BASE_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        ),
    )
    parser.add_argument(
        "--static-dir",
        default=os.path.join(os.path.dirname(__file__), "web"),
    )
    parser.add_argument("--sse-interval-s", type=float, default=float(os.getenv("SCIMAS_DASHBOARD_SSE_INTERVAL_S", "2")))
    args = parser.parse_args()

    srv = LiveDashboardServer(
        (args.host, args.port),
        Handler,
        base_dir=os.path.abspath(args.base_dir),
        static_dir=os.path.abspath(args.static_dir),
        sse_interval_s=max(0.5, float(args.sse_interval_s)),
    )
    print(f"[live-dashboard] serving http://{args.host}:{args.port} base_dir={srv.base_dir}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
