#!/usr/bin/env python3
"""Client gate (002 SC-001, D057): scripted agent tasks through `mrun launch`.

For every client × run × task: copy the task fixture into a fresh git repository, start the
client non-interactively through `mrun launch`, then let the task's `check.sh` decide pass or
fail (never the agent's own claim). Prints pass rates with Wilson 95 % intervals per client and
writes every run as a JSON line.

usage: scripts/client_gate.py [--model qwen3.6] [--clients claude-code,codex,opencode]
                              [--runs 3] [--tasks 01,02] [--timeout 600] [--out results.jsonl]
Needs: a release build of mrun (target/release/mrun or MRUN), the clients on PATH, the model
pulled. Each task runs in a temporary directory; nothing outside it is touched.

Before each run a preflight (D062) refuses to go on while another inference server runs (mrun
on another port, Ollama, llama-server, mlx_lm, oMLX, Rapid-MLX, Splash) or, while mrun's server is not up yet,
when the available memory (vm_stat: free + inactive + purgeable + speculative) does not hold
the model plus a 6 GiB margin; it prints the largest memory users.
"""
import argparse, hashlib, json, math, os, pathlib, shutil, signal, subprocess, sys, tempfile, time, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
TASKS = ROOT / "crates/mrun/tests/clients/tasks"
MRUN = os.environ.get("MRUN", str(ROOT / "target/release/mrun"))


def client_args(client, prompt):
    if client == "opencode":
        # A private server per run: the shared background service keeps the working directory
        # it was started in and would edit files there instead of in the task's copy.
        return ["run", "--standalone", "--auto", prompt]
    if client == "claude-code":
        tools = ["Read", "Edit", "Write", "Glob", "Grep", "Bash(python3:*)", "Bash(ls:*)",
                 "Bash(cat:*)", "Bash(grep:*)"]
        return ["-p", prompt, "--permission-mode", "acceptEdits", "--allowedTools", *tools]
    if client == "codex":
        return ["exec", "--skip-git-repo-check", "-s", "workspace-write", prompt]
    raise SystemExit(f"unknown client {client}")


GIB = 1 << 30
MARGIN = 6 * GIB


def server_kind(command):
    """The known inference server a command line runs, or None (mirrors bench/preflight.rs)."""
    tokens = command.split()
    base = lambda t: t.rsplit("/", 1)[-1]
    followed = lambda name, nxt: any(base(a) == name and b in nxt for a, b in zip(tokens, tokens[1:]))
    if followed("mrun", ("serve",)):
        return "mrun"
    if followed("ollama", ("serve", "runner")):
        return "ollama"
    if any(base(t) == "llama-server" for t in tokens):
        return "llama-server"
    if any("mlx_lm.server" in t or t.endswith("mlx_lm/server.py") for t in tokens) \
            or followed("mlx_lm", ("server",)):
        return "mlx_lm"
    for name in ("omlx", "rapid-mlx", "splash"):
        if followed(name, ("serve",)):
            return name
    return None


def servers_in(ps_text, own):
    """Known server kinds in `ps -axo pid=,command=` output, without the process `own`."""
    kinds = set()
    for line in ps_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != own:
            kind = server_kind(parts[1])
            if kind:
                kinds.add(kind)
    return sorted(kinds)


def available_from_vm_stat(text):
    """free + inactive + purgeable + speculative pages, in bytes (as mrun's budget counts)."""
    lines = text.splitlines()
    page = int(lines[0].split("page size of ")[1].split()[0])
    counts = {}
    for line in lines[1:]:
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            counts[key.strip()] = int(value)
    keys = ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative")
    return sum(counts.get(k, 0) for k in keys) * page


def top_users(ps_text, n=3):
    """The largest processes by resident memory from `ps -axo rss=,comm=`: names only."""
    rows = []
    for line in ps_text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            rows.append((int(parts[0]) * 1024, parts[1].rsplit("/", 1)[-1]))
    rows.sort(reverse=True)
    return ", ".join(f"{name} {size / GIB:.1f} GiB" for size, name in rows[:n])


def model_size(model):
    """Bytes of `model` from `mrun ls --json` (an alias or a repo id); None when unknown."""
    try:
        rows = json.loads(subprocess.run([MRUN, "ls", "--json"], capture_output=True,
                                         check=True).stdout)
    except Exception:
        return None
    for row in rows:
        if model in (row.get("alias"), row.get("repo")):
            return row.get("size")
    return None


def preflight(model, port):
    """D062: one server at a time, and only into memory that is there."""
    if sys.platform != "darwin":
        return
    # A server already serving on the gate's port (a local model, or a relay to a cloud pod, whose
    # status has no engine cache fields) is the one the gate uses.
    ours = serving(port)
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    others = [k for k in servers_in(ps, os.getpid()) if not (ours and k == "mrun")]
    if others:
        raise SystemExit(f"preflight: {', '.join(others)} is running; stop it (and wait until it "
                         f"has exited) before the gate starts mrun")
    if ours:
        return  # the model is loaded already; its memory is in use, not missing
    size = model_size(model)
    if size is None:
        raise SystemExit(f"preflight: {model} is not in 'mrun ls --json'; pull it first")
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    available = available_from_vm_stat(vm)
    if available < size + MARGIN:
        rss = subprocess.run(["ps", "-axo", "rss=,comm="], capture_output=True, text=True).stdout
        raise SystemExit(f"preflight: {model} needs {size / GIB:.1f} GiB plus a "
                         f"{MARGIN / GIB:.0f} GiB margin, but only {available / GIB:.1f} GiB are "
                         f"available; largest memory users: {top_users(rss)}. Quit some and rerun")


def serving(port):
    """Whether an mrun server on `port` reports a loaded model on /health."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return bool(json.load(r).get("loaded"))
    except Exception:
        return False


def status(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/mrun/status", timeout=5) as r:
            return json.load(r).get("cache") or {}
    except Exception:
        return {}


def tree_hash(path):
    """Content hash of a fixture tree (bytecode ignored): the run must never change it."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(path)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def run_one(client, task, model, port, timeout, logdir):
    fixture = TASKS / task
    pristine = tree_hash(fixture)
    prompt = (fixture / "PROMPT.txt").read_text().strip()
    work = pathlib.Path(tempfile.mkdtemp(prefix=f"gate-{task}-"))
    skip = shutil.ignore_patterns("__pycache__", "*.pyc")
    for item in fixture.iterdir():
        if item.name in ("PROMPT.txt", "check.sh", "__pycache__"):
            continue
        if item.is_dir():
            shutil.copytree(item, work / item.name, ignore=skip)
        else:
            shutil.copy2(item, work / item.name)
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(["git", "-c", "user.name=gate", "-c", "user.email=gate@local", "commit", "-qm",
                    "fixture"], cwd=work, check=True)
    # PWD too: OpenCode takes its project root from PWD, not from the process working directory.
    env = dict(os.environ, MRUN_PORT=str(port), PWD=str(work))
    before = status(port)
    started = time.time()
    log = logdir / f"{client}-{task}-{int(started)}.log"
    stalled = None
    with open(log, "w") as out:
        # Own process group: a timeout must also end the client `mrun launch` started, or it
        # keeps the server's single lane busy and every later run queues behind it.
        p = subprocess.Popen([MRUN, "launch", client, "--model", model, "--yes", "--",
                              *client_args(client, prompt)], cwd=work, env=env, stdout=out,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        try:
            exit_code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            exit_code = "timeout"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/mrun/status", timeout=5) as r:
                    stalled = json.load(r).get("requests")
            except Exception:
                pass
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    elapsed = time.time() - started
    check = subprocess.run(["sh", str(fixture / "check.sh")], cwd=work,
                           env=dict(os.environ, FIXTURE=str(fixture)), capture_output=True)
    after = status(port)
    delta = lambda k: (after.get(k) or 0) - (before.get(k) or 0)
    # Turns that could reuse the model's own previous reply, and turns that did (003 FR-020).
    splice = lambda st, field: sum(v.get(field, 0) for v in (st.get("splice") or {}).values())
    turns_seen = splice(after, "seen") - splice(before, "seen")
    turns_spliced = splice(after, "spliced") - splice(before, "spliced")
    shutil.rmtree(work, ignore_errors=True)
    for cache in fixture.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    if tree_hash(fixture) != pristine:
        raise SystemExit(f"{client} changed the fixture {fixture} outside its working copy; "
                         f"restore it (git status) and fix the client's isolation. Log: {log}")
    return {"client": client, "task": task, "model": model, "passed": check.returncode == 0,
            "exit": exit_code, "seconds": round(elapsed, 1), "prompt_tokens": delta("prompt_tokens_total"),
            "cached_tokens": delta("cached_tokens_total"), "spliced_tokens": delta("spliced_tokens_total"),
            "turns_seen": turns_seen, "turns_spliced": turns_spliced,
            "log": str(log), "stalled_requests": stalled}


def main():
    # Keep the Mac awake for the whole gate (an overnight sleep broke a 90-run gate).
    if sys.platform == "darwin" and not os.environ.get("CLIENT_GATE_AWAKE"):
        os.environ["CLIENT_GATE_AWAKE"] = "1"
        os.execvp("caffeinate", ["caffeinate", "-dimsu", sys.executable, *sys.argv])
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6")
    ap.add_argument("--clients", default="claude-code,codex,opencode")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--port", type=int, default=int(os.environ.get("MRUN_PORT", 8484)))
    ap.add_argument("--out", default="client-gate.jsonl")
    ap.add_argument("--self-test", action="store_true", help="check the Wilson bound and exit")
    a = ap.parse_args()
    if a.self_test:
        # 002 SC-001: 30/30 → 0.886, 29/30 → 0.833 meet the 0.8 bound; 28/30 → 0.787 does not.
        for k, want, meets in ((30, 0.886, True), (29, 0.833, True), (28, 0.787, False)):
            lo = wilson(k, 30)[0]
            assert abs(lo - want) < 0.0015 and (lo >= 0.8) == meets, (k, lo)
        print("wilson ok")
        vm = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\n"
              "Pages active: 99.\nPages inactive: 20.\nPages speculative: 3.\nPages purgeable: 1.\n")
        assert available_from_vm_stat(vm) == 34 * 16384
        ps = (" 1 /sbin/launchd\n 7 /Applications/Ollama.app/Contents/Resources/ollama serve\n"
              " 8 /opt/homebrew/bin/mrun serve --model qwen3.6\n 9 /venv/bin/python -m mlx_lm.server\n"
              " 10 mrun bench --compare\n 11 llama-server -m q.gguf\n 12 omlx serve --model-dir d\n")
        assert servers_in(ps, 1) == ["llama-server", "mlx_lm", "mrun", "ollama", "omlx"], servers_in(ps, 1)
        assert servers_in(ps, 8) == ["llama-server", "mlx_lm", "ollama", "omlx"]
        rss = "33868800 /S/com.apple.Virtualization.VirtualMachine\n100 launchd\n4508876 /x/rust-analyzer\n"
        assert top_users(rss, 2) == "com.apple.Virtualization.VirtualMachine 32.3 GiB, rust-analyzer 4.3 GiB"
        print("preflight ok")
        return 0
    tasks = sorted(p.name for p in TASKS.iterdir() if p.is_dir())
    if a.tasks:
        tasks = [t for t in tasks if t.split("-")[0] in a.tasks.split(",")]
    logdir = pathlib.Path(tempfile.mkdtemp(prefix="gate-logs-"))
    rows = []
    with open(a.out, "a") as out:
        for client in a.clients.split(","):
            for run in range(1, a.runs + 1):
                for task in tasks:
                    preflight(a.model, a.port)
                    r = run_one(client, task, a.model, a.port, a.timeout, logdir)
                    r["run"] = run
                    rows.append(r)
                    out.write(json.dumps(r) + "\n")
                    out.flush()
                    mark = "pass" if r["passed"] else "FAIL"
                    print(f"{client:12} run {run} {task:24} {mark}  {r['seconds']:6.1f}s  "
                          f"cached {r['cached_tokens']}/{r['prompt_tokens']}", flush=True)
    print(f"\nmodel {a.model}; logs in {logdir}")
    print(f"{'client':12} {'passed':>9} {'rate':>6}  {'wilson 95%':>15}  {'median s':>8}  {'cached':>7}  {'spliced':>9}")
    for client in a.clients.split(","):
        rs = [r for r in rows if r["client"] == client]
        k, n = sum(r["passed"] for r in rs), len(rs)
        lo, hi = wilson(k, n)
        secs = sorted(r["seconds"] for r in rs)
        med = secs[len(secs) // 2] if secs else 0
        p, c = sum(r["prompt_tokens"] for r in rs), sum(r["cached_tokens"] for r in rs)
        gate = "meets gate" if lo >= 0.8 else "below gate"
        seen, spliced = sum(r["turns_seen"] for r in rs), sum(r["turns_spliced"] for r in rs)
        print(f"{client:12} {k:>4}/{n:<4} {k / max(n, 1):6.0%}  [{lo:.2f}, {hi:.2f}]  {med:8.1f}  "
              f"{c / max(p, 1):7.0%}  {spliced:>4}/{seen:<4}  {gate}")
    failed = [f"{r['client']}/{r['task']}/run{r['run']}" for r in rows if not r["passed"]]
    if failed:
        print("failed:", ", ".join(failed))


if __name__ == "__main__":
    sys.exit(main())
