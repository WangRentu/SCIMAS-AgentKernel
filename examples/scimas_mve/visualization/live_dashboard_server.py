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
from urllib.parse import parse_qs, urlparse

from examples.scimas_mve.visualization.trend_dashboard import (
    build_dashboard_payload,
    normalize_llm_io_rows,
    normalize_rag_io_rows,
    normalize_retrieve_evidence_rows,
    normalize_retrieve_guardrail_rows,
    normalize_retrieve_pipeline_rows,
)


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
        self.log_cache = LogCache()

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


class LogCache:
    """Lightweight mtime-based JSONL cache to avoid full rescans each SSE cycle."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def read_jsonl(self, path: str) -> list[Dict[str, Any]]:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            self._cache[path] = {"mtime_ns": None, "size": None, "rows": []}
            return []
        cached = self._cache.get(path)
        if cached and cached.get("mtime_ns") == st.st_mtime_ns and cached.get("size") == st.st_size:
            return list(cached.get("rows") or [])
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            rows = []
        self._cache[path] = {"mtime_ns": st.st_mtime_ns, "size": st.st_size, "rows": rows}
        return list(rows)


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

    @staticmethod
    def _qparam(params: Dict[str, list[str]], key: str, default: str = "") -> str:
        vals = params.get(key) or []
        if not vals:
            return default
        return str(vals[-1] or default)

    @staticmethod
    def _to_int(v: str, default: int) -> int:
        try:
            return int(v)
        except Exception:
            return default

    @staticmethod
    def _row_text(row: Dict[str, Any]) -> str:
        try:
            return json.dumps(row, ensure_ascii=False).lower()
        except Exception:
            return str(row).lower()

    def _resolve_episode(self, rows: list[Dict[str, Any]], episode_param: str) -> int | None:
        ep_raw = (episode_param or "").strip().lower()
        if ep_raw in {"", "current"}:
            eps = [int(r.get("episode_id") or 0) for r in rows if int(r.get("episode_id") or 0) > 0]
            return max(eps) if eps else None
        if ep_raw in {"all", "0"}:
            return None
        try:
            ep = int(ep_raw)
            return ep if ep > 0 else None
        except Exception:
            return None

    def _filter_rows(
        self,
        rows: list[Dict[str, Any]],
        *,
        episode: int | None,
        agent: str,
        action: str,
        task_id: str,
        run_id: str,
    ) -> list[Dict[str, Any]]:
        out = []
        agent_l = agent.lower().strip()
        action_l = action.lower().strip()
        task_l = task_id.lower().strip()
        run_l = run_id.lower().strip()
        for row in rows:
            if episode is not None and int(row.get("episode_id") or 0) != int(episode):
                continue
            if agent_l and str(row.get("agent_id") or "").lower() != agent_l:
                continue
            if action_l and str(row.get("action") or "").lower() != action_l:
                continue
            row_text = ""
            if task_l:
                row_text = self._row_text(row)
                if task_l not in row_text:
                    continue
            if run_l:
                if not row_text:
                    row_text = self._row_text(row)
                if run_l not in row_text:
                    continue
            out.append(row)
        return out

    def _paginate(self, rows: list[Dict[str, Any]], *, cursor: str, limit: int) -> tuple[list[Dict[str, Any]], str]:
        start = self._to_int(cursor, 0)
        start = max(0, start)
        lim = max(1, min(200, limit))
        end = start + lim
        sliced = rows[start:end]
        next_cursor = str(end) if end < len(rows) else ""
        return sliced, next_cursor

    def _kind_conf(self, kind: str) -> tuple[str, Any]:
        base = self.server.base_dir
        if kind == "llm_io":
            return os.path.join(base, "logs", "app", "audit", "llm_io.jsonl"), normalize_llm_io_rows
        if kind == "rag_io":
            return os.path.join(base, "logs", "app", "audit", "rag_io.jsonl"), normalize_rag_io_rows
        if kind == "retrieve_pipeline":
            return os.path.join(base, "logs", "app", "action", "retrieve_pipeline.jsonl"), normalize_retrieve_pipeline_rows
        if kind == "retrieve_guardrail":
            return os.path.join(base, "logs", "app", "action", "retrieve_guardrail.jsonl"), normalize_retrieve_guardrail_rows
        if kind == "retrieve_evidence":
            return os.path.join(base, "logs", "app", "action", "retrieve_evidence.jsonl"), normalize_retrieve_evidence_rows
        return "", None

    def _query_kind(self, *, kind: str, params: Dict[str, list[str]]) -> Dict[str, Any]:
        path, normalizer = self._kind_conf(kind)
        if not path or normalizer is None:
            return {"ok": False, "error": f"unknown_kind:{kind}"}
        raw_rows = self.server.log_cache.read_jsonl(path)
        norm = normalizer(raw_rows, max_rows=3000)
        summary_rows = list((norm or {}).get("rows_summary") or [])
        detail_rows = list((norm or {}).get("rows_detail") or [])
        details_by_id = {str(r.get("id") or ""): r for r in detail_rows if str(r.get("id") or "")}
        summary_rows = sorted(summary_rows, key=lambda x: x.get("ts", ""), reverse=True)

        episode_param = self._qparam(params, "episode", "current")
        episode = self._resolve_episode(summary_rows, episode_param)
        agent = self._qparam(params, "agent", "")
        action = self._qparam(params, "action", "")
        task_id = self._qparam(params, "task_id", "")
        run_id = self._qparam(params, "run_id", "")
        cursor = self._qparam(params, "cursor", "")
        limit = self._to_int(self._qparam(params, "limit", "50"), 50)

        filtered = self._filter_rows(
            summary_rows,
            episode=episode,
            agent=agent,
            action=action,
            task_id=task_id,
            run_id=run_id,
        )
        paged, next_cursor = self._paginate(filtered, cursor=cursor, limit=limit)
        return {
            "ok": True,
            "kind": kind,
            "rows": paged,
            "next_cursor": next_cursor,
            "total": len(filtered),
            "episode": episode if episode is not None else "all",
            "details_by_id": details_by_id,
        }

    def _fetch_detail(self, *, kind: str, item_id: str) -> Dict[str, Any]:
        path, normalizer = self._kind_conf(kind)
        if not path or normalizer is None:
            return {"ok": False, "error": f"unknown_kind:{kind}"}
        raw_rows = self.server.log_cache.read_jsonl(path)
        norm = normalizer(raw_rows, max_rows=3000)
        detail_rows = list((norm or {}).get("rows_detail") or [])
        target = None
        for row in detail_rows:
            if str(row.get("id") or "") == str(item_id or ""):
                target = row
                break
        if target is None:
            return {"ok": False, "error": f"detail_not_found:{kind}:{item_id}"}

        if kind == "retrieve_pipeline":
            agent_id = str(target.get("agent_id") or "")
            episode_id = int(target.get("episode_id") or 0)
            ts = str(target.get("ts") or "")
            guard_norm = normalize_retrieve_guardrail_rows(
                self.server.log_cache.read_jsonl(os.path.join(self.server.base_dir, "logs", "app", "action", "retrieve_guardrail.jsonl")),
                max_rows=3000,
            )
            evid_norm = normalize_retrieve_evidence_rows(
                self.server.log_cache.read_jsonl(os.path.join(self.server.base_dir, "logs", "app", "action", "retrieve_evidence.jsonl")),
                max_rows=3000,
            )
            guard_rows = [
                r
                for r in (guard_norm.get("rows_detail") or [])
                if str(r.get("agent_id") or "") == agent_id and int(r.get("episode_id") or 0) == episode_id and str(r.get("ts") or "") <= ts
            ]
            evid_rows = [
                r
                for r in (evid_norm.get("rows_detail") or [])
                if str(r.get("agent_id") or "") == agent_id and int(r.get("episode_id") or 0) == episode_id and str(r.get("ts") or "") <= ts
            ]
            target = dict(target)
            target["related_guardrail"] = guard_rows[:3]
            target["related_evidence"] = evid_rows[:3]

        return {"ok": True, "kind": kind, "record": target}

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
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query or "", keep_blank_values=True)
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
        if path == "/api/audit/llm":
            out = self._query_kind(kind="llm_io", params=params)
            self._send_json(HTTPStatus.OK if out.get("ok") else HTTPStatus.BAD_REQUEST, out)
            return
        if path == "/api/audit/rag":
            out = self._query_kind(kind="rag_io", params=params)
            self._send_json(HTTPStatus.OK if out.get("ok") else HTTPStatus.BAD_REQUEST, out)
            return
        if path == "/api/retrieve/pipeline":
            out = self._query_kind(kind="retrieve_pipeline", params=params)
            self._send_json(HTTPStatus.OK if out.get("ok") else HTTPStatus.BAD_REQUEST, out)
            return
        if path == "/api/retrieve/guardrail":
            out = self._query_kind(kind="retrieve_guardrail", params=params)
            self._send_json(HTTPStatus.OK if out.get("ok") else HTTPStatus.BAD_REQUEST, out)
            return
        if path == "/api/retrieve/evidence":
            out = self._query_kind(kind="retrieve_evidence", params=params)
            self._send_json(HTTPStatus.OK if out.get("ok") else HTTPStatus.BAD_REQUEST, out)
            return
        if path == "/api/audit/detail":
            kind = self._qparam(params, "kind", "")
            item_id = self._qparam(params, "id", "")
            out = self._fetch_detail(kind=kind, item_id=item_id)
            self._send_json(HTTPStatus.OK if out.get("ok") else HTTPStatus.NOT_FOUND, out)
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
