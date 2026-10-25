"""Serve the exact App Lab assets locally, using the same routing model."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))
from bonus import select_depths  # noqa: E402
from routing import solve  # noqa: E402


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "assets"), **kwargs)

    def do_POST(self):
        if self.path not in ("/api/route", "/api/bonus"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("Request too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be an object")
            if self.path == "/api/bonus":
                result = select_depths(payload.get("N"), payload.get("n"))
            else:
                result = solve(
                    payload.get("scenario", "crossbar"),
                    payload.get("weak_error_percent", 12),
                    payload.get("block_weak", False),
                )
            result["board_status"] = "Local preview; board not connected"
            status = 200
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            result, status = {"error": str(exc)}, 400
        body = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port}/?preview=1")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
