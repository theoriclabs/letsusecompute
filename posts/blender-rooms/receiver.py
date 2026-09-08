"""Private, bounded transfer of this experiment's archives only."""

import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LIMIT = 64 * 1024 * 1024
SLOTS = {"probe", "smoke", "sft-checkpoint", "sft", "test"}


def server(root, key_file, port):
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    if not key_file.exists():
        key_file.write_text(secrets.token_urlsafe(32))
        key_file.chmod(0o600)
    token = key_file.read_text().strip()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def slot(self, verb):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "unauthorized"})
                return None
            match = re.fullmatch("/" + verb + "/([a-z-]+)", self.path)
            if not match or match[1] not in SLOTS:
                self.reply(404, {"error": "unknown slot"})
                return None
            return match[1]

        def do_PUT(self):
            slot = self.slot("upload")
            if slot is None:
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                digest = self.headers.get("X-Content-SHA256", "")
                if not 0 < length <= LIMIT or not re.fullmatch("[0-9a-f]{64}", digest):
                    self.reply(400, {"error": "invalid size or checksum"})
                    return
                self.connection.settimeout(120)
                with lock:
                    target = root / (slot + ".tar.gz")
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
                            temporary = Path(stream.name)
                            remaining = length
                            actual = hashlib.sha256()
                            while remaining:
                                chunk = self.rfile.read(min(256 * 1024, remaining))
                                if not chunk:
                                    raise ValueError("incomplete upload")
                                stream.write(chunk)
                                actual.update(chunk)
                                remaining -= len(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                        if actual.hexdigest() != digest:
                            self.reply(400, {"error": "checksum mismatch"})
                            return
                        if target.exists():
                            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                                self.reply(409, {"error": "slot already complete"})
                                return
                        else:
                            temporary.replace(target)
                        receipt = {"slot": slot, "bytes": length, "sha256": digest}
                        (root / (slot + ".json")).write_text(json.dumps(receipt, indent=2))
                        self.reply(200, receipt)
                        print(json.dumps({"received": receipt}), flush=True)
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
            except (OSError, ValueError):
                self.close_connection = True

        def do_GET(self):
            slot = self.slot("download")
            if slot is None:
                return
            path = root / (slot + ".tar.gz")
            if not path.exists():
                self.reply(404, {"error": "not uploaded"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as stream:
                while chunk := stream.read(256 * 1024):
                    self.wfile.write(chunk)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8891)
    args = parser.parse_args()
    server(args.root, args.key_file, args.port).serve_forever()
