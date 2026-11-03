# mrun pod bootstrap (013, research R6). Runs as `python3 -c "exec(os.environ['MRUN_BOOTSTRAP'])"`
# in the pod: fetches the agent from MRUN_AGENT_URL, or accepts one upload (`PUT /mrun/agent`
# with the pod token) on the agent port; checks MRUN_AGENT_SHA256; execs `<agent> agent`.
import hashlib, hmac, http.server, json, os, sys, urllib.request

DST = "/tmp/mrun-agent"
WANT = os.environ["MRUN_AGENT_SHA256"].strip().lower()
PORT = int(os.environ.get("MRUN_AGENT_PORT", "8000"))
TOKEN = os.environ["MRUN_POD_TOKEN"]
URL = os.environ.get("MRUN_AGENT_URL", "")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch():
    req = urllib.request.Request(URL, headers={"User-Agent": "@USER_AGENT@"})
    with urllib.request.urlopen(req, timeout=300) as r, open(DST, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)


class Upload(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "bootstrap"})
        else:
            self.reply(404, {"error": "not found"})

    def do_PUT(self):
        auth = self.headers.get("Authorization", "")
        if self.path != "/mrun/agent" or not hmac.compare_digest(auth, "Bearer " + TOKEN):
            return self.reply(401, {"error": "unauthorized"})
        left = int(self.headers.get("Content-Length", "0"))
        with open(DST, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left or digest(DST) != WANT:
            os.remove(DST)
            return self.reply(400, {"error": "incomplete upload or sha256 mismatch"})
        self.reply(200, {"status": "ok"})
        self.server.done = True


if URL:
    print("mrun bootstrap: fetching the agent", flush=True)
    fetch()
else:
    print("mrun bootstrap: waiting for the agent upload", flush=True)
    server = http.server.HTTPServer(("0.0.0.0", PORT), Upload)
    server.done = False
    while not server.done:
        server.handle_request()
    server.server_close()
if digest(DST) != WANT:
    print("mrun bootstrap: sha256 mismatch, not starting the agent", flush=True)
    sys.exit(1)
os.chmod(DST, 0o755)
print("mrun bootstrap: starting the agent", flush=True)
os.execv(DST, [DST, "agent"])
