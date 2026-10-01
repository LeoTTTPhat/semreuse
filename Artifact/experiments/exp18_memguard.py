"""Run exp18 (SemReuse under LOTUS, both repetitions) under a memory guard.

One supervisor owns the whole stack -- the Ollama servers, the round-robin
proxy and one exp18 process per repetition -- so nothing restarts behind its
back.  Every --poll seconds it measures

  * the project's memory: the physical footprint (macOS ``footprint``) of
    every process it started and all their descendants (Ollama's model
    runners included), and
  * the machine's used memory: app + wired + compressed pages (``vm_stat``),
    what Activity Monitor calls "Memory Used";

and if the project reaches --project-limit-gb (200) or the machine reaches
--system-limit-gb (490) it stops the whole stack.  It restarts only once the
machine's used memory has stayed at or below --resume-below-gb (420) for
three consecutive polls and a back-off has passed (5 min, doubling to 1 h),
and it restarts slowly: one server at a time, each model loaded and memory
re-checked before the next, then the proxy, then the repetitions several
minutes apart.  A repetition resumes from its last completed query pair
(exp18_lotus_scale.py --resume), so a stop costs at most one pair.  A
repetition that crashes, or whose log is silent for --stall-min minutes, is
restarted the same way.

Start it detached (it survives the shell that launched it):
    .venv/bin/python experiments/exp18_memguard.py --detach
Log: results/logs/exp18_memguard.log.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
OLLAMA = os.environ.get("OLLAMA_BIN",
                        "/Applications/Ollama.app/Contents/Resources/ollama")
MODEL = "llama3.1:8b-instruct-q4_K_M"
GB = 2 ** 30


def log(msg: str) -> None:
    line = f"[memguard {time.strftime('%F %T')}] {msg}"
    with open(LOGS / "exp18_memguard.log", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


# -- measuring ---------------------------------------------------------------

def system_used_gb() -> float:
    t = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", t).group(1))

    def g(k: str) -> int:
        return int(re.search(k + r":\s+(\d+)", t).group(1))
    pages = (g("Anonymous pages") - g("Pages purgeable")
             + g("Pages wired down") + g("Pages occupied by compressor"))
    return pages * page / GB


def descendants(pids: list[int]) -> list[int]:
    out = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True,
                         text=True).stdout.split()
    kids: dict[int, list[int]] = {}
    for pid, ppid in zip(out[0::2], out[1::2]):
        kids.setdefault(int(ppid), []).append(int(pid))
    seen, todo = set(), [p for p in pids if p]
    while todo:
        p = todo.pop()
        if p not in seen:
            seen.add(p)
            todo += kids.get(p, [])
    return sorted(seen)


def footprint_gb(pids: list[int]) -> float:
    """Sum of physical footprints; falls back to RSS for a pid footprint
    cannot read."""
    if not pids:
        return 0.0
    total = 0.0
    args = ["footprint"] + [a for p in pids for a in ("-p", str(p))]
    r = subprocess.run(args, capture_output=True, text=True, timeout=120)
    got = set()
    # One header line per process, "<name> [<pid>]: 64-bit  Footprint: <n>
    # <unit> (...)"; "Shared with <name> [<pid>]" lines are not headers.
    for m in re.finditer(r"\[(\d+)\]:\s+\d+-bit\s+Footprint:\s+([\d.]+)"
                         r"\s+([KMGT]?B)", r.stdout):
        pid, v, u = int(m.group(1)), float(m.group(2)), m.group(3)
        total += v * {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30,
                      "TB": 2**40}[u] / GB
        got.add(pid)
    rest = [p for p in pids if p not in got]
    if rest:
        out = subprocess.run(["ps", "-o", "rss=", "-p",
                              ",".join(map(str, rest))],
                             capture_output=True, text=True).stdout.split()
        total += sum(int(x) for x in out) * 1024 / GB
    return total


# -- the stack ---------------------------------------------------------------

class Stack:
    def __init__(self, a):
        self.a = a
        self.ports = [a.first_port + i for i in range(a.servers)]
        self.servers: dict[int, subprocess.Popen] = {}
        self.proxy: subprocess.Popen | None = None
        self.reps: dict[int, subprocess.Popen] = {}

    def pids(self) -> list[int]:
        top = [p.pid for p in self.servers.values()]
        top += [p.pid for p in self.reps.values()]
        if self.proxy:
            top.append(self.proxy.pid)
        return descendants(top)

    def _popen(self, cmd, logf, env=None) -> subprocess.Popen:
        f = open(logf, "a")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                env=env, cwd=ROOT, start_new_session=True)

    def start_server(self, port: int) -> bool:
        env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}",
                   OLLAMA_NUM_PARALLEL="8", OLLAMA_MAX_LOADED_MODELS="1",
                   OLLAMA_KEEP_ALIVE="600m", OLLAMA_CONTEXT_LENGTH="4096")
        self.servers[port] = self._popen([OLLAMA, "serve"],
                                         LOGS / f"ollama_{port}.log", env)
        for _ in range(60):
            try:
                v = urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/version", timeout=2).read()
                log(f"server {port}: Ollama {json.loads(v)['version']}")
                break
            except Exception:
                time.sleep(1)
        else:
            return False
        body = json.dumps({"model": MODEL, "prompt": "Answer True.",
                           "stream": False,
                           "options": {"temperature": 0,
                                       "num_predict": 1}}).encode()
        try:                                  # load the model now
            urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/api/generate", body,
                {"Content-Type": "application/json"}), timeout=600).read()
            return True
        except Exception as e:
            log(f"server {port}: warm-up failed: {e}")
            return False

    def start_proxy(self) -> None:
        backends = ",".join(f"http://127.0.0.1:{p}" for p in self.ports)
        self.proxy = self._popen(
            [str(ROOT / ".venv/bin/python"), "experiments/rr_proxy.py",
             "--port", str(self.a.proxy_port),
             "--max-inflight", str(self.a.proxy_max_inflight),
             "--backends", backends], LOGS / "exp18_proxy.log")
        time.sleep(2)

    def rep_files(self, rep: int, tag: str):
        n, q = self.a.n, self.a.queries_tag
        return (ROOT / "results" / f"exp18_lotus_{n}_{q}{tag}.json",
                ROOT / "results" / f"exp18_state_{n}_{q}{tag}.pkl",
                LOGS / f"exp18_lotus_{n}{tag or ''}.log")

    def finished(self, rep: int, tag: str) -> bool:
        js, _, _ = self.rep_files(rep, tag)
        if not js.exists():
            return False
        try:
            d = json.loads(js.read_text())
        except Exception:
            return False
        return "reduction" in d and "progress" not in d

    def start_rep(self, rep: int, tag: str) -> None:
        js, state, logf = self.rep_files(rep, tag)
        cmd = [str(ROOT / ".venv-lotus/bin/python"), "-u",
               "experiments/exp18_lotus_scale.py", "--n", str(self.a.n),
               "--repeats", "2", "--only-rep", str(rep),
               "--api-base", f"http://127.0.0.1:{self.a.proxy_port}"]
        if self.a.queries:
            cmd += ["--queries", str(self.a.queries)]
        if tag:
            cmd += ["--out-tag", tag]
        if state.exists():
            cmd.append("--resume")
        with open(logf, "a") as f:
            f.write(f"[memguard] start rep {rep} "
                    f"{'(resume)' if state.exists() else '(fresh)'} "
                    f"{time.strftime('%F %T')}\n")
        # LiteLLM looks up Ollama model metadata (/api/show) at
        # OLLAMA_API_BASE, else at the default port 11434 -- someone else's
        # server; point it at this stack.  Every model the run needs is in
        # the local Hugging Face cache, so the Hub is never contacted.
        env = dict(os.environ,
                   OLLAMA_API_BASE=f"http://127.0.0.1:{self.a.proxy_port}",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.reps[rep] = self._popen(cmd, logf, env)

    @staticmethod
    def _kill(p: subprocess.Popen, grace: float) -> None:
        if p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        t = time.time()
        while p.poll() is None and time.time() - t < grace:
            time.sleep(0.5)
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=30)

    def stop_all(self) -> None:
        for p in self.reps.values():
            self._kill(p, 30)
        self.reps.clear()
        if self.proxy:
            self._kill(self.proxy, 5)
            self.proxy = None
        for p in self.servers.values():
            self._kill(p, 20)
        self.servers.clear()


# -- supervising -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-limit-gb", type=float, default=200)
    ap.add_argument("--system-limit-gb", type=float, default=490)
    ap.add_argument("--resume-below-gb", type=float, default=420)
    ap.add_argument("--servers", type=int, default=4)
    ap.add_argument("--first-port", type=int, default=11440)
    ap.add_argument("--proxy-port", type=int, default=11500)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=0, help="0 = whole log")
    ap.add_argument("--reps", default="0:,1:_rep1",
                    help="rep:out_tag pairs, comma-separated")
    ap.add_argument("--poll", type=float, default=15)
    ap.add_argument("--step-pause", type=float, default=30,
                    help="seconds between starting servers")
    ap.add_argument("--rep-pause", type=float, default=180,
                    help="seconds between starting repetitions")
    ap.add_argument("--stall-min", type=float, default=20)
    ap.add_argument("--proxy-max-inflight", type=int, default=0,
                    help="per-backend cap in rr_proxy (0 = plain round "
                         "robin, the configuration of the 2026-09-29 run)")
    ap.add_argument("--max-restarts", type=int, default=8)
    ap.add_argument("--backoff", type=float, default=300,
                    help="first wait (s) after a stop; doubles on failure")
    ap.add_argument("--test-stop-after", type=float, default=0,
                    help="testing: force one stop this many seconds in")
    ap.add_argument("--detach", action="store_true",
                    help="re-launch in a new session and return")
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.detach:
        argv = [sys.executable, __file__] + [x for x in sys.argv[1:]
                                             if x != "--detach"]
        f = open(LOGS / "exp18_memguard.out", "a")
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT,
                             cwd=ROOT, start_new_session=True)
        print(f"memguard started, pid {p.pid}; log "
              f"{LOGS / 'exp18_memguard.log'}")
        return
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    a.queries_tag = a.queries or 47
    reps = [(int(r.split(":")[0]), r.split(":", 1)[1])
            for r in a.reps.split(",")]
    st = Stack(a)

    def stop(reason: str) -> None:
        log(f"STOP ({reason}); stopping the whole stack")
        st.stop_all()
        log(f"stopped; machine used {system_used_gb():.1f} GB")

    def over(tag: str) -> str | None:
        sysu, proj = system_used_gb(), footprint_gb(st.pids())
        if proj >= a.project_limit_gb:
            return f"{tag}project {proj:.1f} GB >= {a.project_limit_gb:.0f}"
        if sysu >= a.system_limit_gb:
            return f"{tag}machine {sysu:.1f} GB >= {a.system_limit_gb:.0f}"
        return None

    def start_slowly() -> bool:
        """Start what is not running, one piece at a time; False if memory
        crossed a limit on the way (the stack is then stopped)."""
        for port in st.ports:
            if port in st.servers:
                continue
            if not st.start_server(port):
                stop(f"server {port} failed to start")
                return False
            time.sleep(a.step_pause)
            why = over(f"after server {port}: ")
            log(f"server {port} up; machine {system_used_gb():.1f} GB, "
                f"project {footprint_gb(st.pids()):.1f} GB")
            if why:
                stop(why)
                return False
        if st.proxy is None:
            st.start_proxy()
            log(f"proxy :{a.proxy_port} up")
        first = True
        for rep, tag in reps:
            if rep in st.reps or st.finished(rep, tag):
                continue
            if not first:
                t = time.time()
                while time.time() - t < a.rep_pause:
                    time.sleep(5)
                    why = over(f"while starting reps: ")
                    if why:
                        stop(why)
                        return False
            st.start_rep(rep, tag)
            first = False
            log(f"rep {rep} started (pid {st.reps[rep].pid}); machine "
                f"{system_used_gb():.1f} GB, project "
                f"{footprint_gb(st.pids()):.1f} GB")
        return True

    log(f"supervisor pid {os.getpid()}: limits project "
        f"{a.project_limit_gb:.0f} GB, machine {a.system_limit_gb:.0f} GB; "
        f"restart below {a.resume_below_gb:.0f} GB; reps {reps}")
    running = False
    stopped_at, backoff, calm = 0.0, a.backoff, 0
    t0, test_stopped = time.time(), False
    restarts = dict.fromkeys((r for r, _ in reps), 0)
    last_report = 0.0
    if system_used_gb() <= a.resume_below_gb:
        running = start_slowly()
        if not running:
            stopped_at = time.time()
    else:
        log(f"machine at {system_used_gb():.1f} GB; waiting to start")
        stopped_at = time.time() - backoff
    while True:
        if all(st.finished(r, t) for r, t in reps):
            log("both repetitions finished; stopping servers and proxy")
            st.stop_all()
            return
        sysu, proj = system_used_gb(), footprint_gb(st.pids())
        if time.time() - last_report > 600:
            prog = []
            for r, t in reps:
                _, _, lf = st.rep_files(r, t)
                n = (sum(1 for ln in open(lf, errors="ignore")
                         if " pair " in ln) if lf.exists() else 0)
                prog.append(f"rep {r}: {n} pair lines"
                            + (" (done)" if st.finished(r, t) else ""))
            log(f"{'running' if running else 'stopped'}: machine "
                f"{sysu:.1f} GB, project {proj:.1f} GB; " + "; ".join(prog))
            last_report = time.time()
        if running and a.test_stop_after and not test_stopped \
                and time.time() - t0 >= a.test_stop_after:
            test_stopped = True
            stop("test: forced stop")
            running, stopped_at, calm = False, time.time(), 0
            continue
        if running:
            if proj >= a.project_limit_gb or sysu >= a.system_limit_gb:
                stop(f"project {proj:.1f} GB, machine {sysu:.1f} GB")
                running, stopped_at, calm = False, time.time(), 0
                continue
            for rep, tag in reps:
                p = st.reps.get(rep)
                if p is None:
                    continue
                rc = p.poll()
                _, _, lf = st.rep_files(rep, tag)
                silent = (time.time() - lf.stat().st_mtime) / 60 \
                    if lf.exists() else 0
                if rc is None and silent < a.stall_min:
                    continue
                if rc == 0 and st.finished(rep, tag):
                    log(f"rep {rep} finished")
                    del st.reps[rep]
                    continue
                restarts[rep] += 1
                what = (f"exited with {rc}" if rc is not None
                        else f"log silent {silent:.0f} min")
                if restarts[rep] > a.max_restarts:
                    log(f"rep {rep} {what}; giving up after "
                        f"{a.max_restarts} restarts")
                    st._kill(p, 30)
                    del st.reps[rep]
                    continue
                log(f"rep {rep} {what}; restart {restarts[rep]} "
                    f"(resume) in 60 s")
                if rc is None:           # still alive: dump its stacks first
                    try:
                        os.kill(p.pid, signal.SIGUSR1)
                        time.sleep(5)
                    except ProcessLookupError:
                        pass
                st._kill(p, 30)
                del st.reps[rep]
                time.sleep(60)
                if silent >= a.stall_min:    # servers may be wedged
                    stop(f"rep {rep} stalled")
                    running, stopped_at, calm = False, time.time(), 0
                    backoff = 60.0
                    break
                st.start_rep(rep, tag)
            if not st.reps and not all(st.finished(r, t) for r, t in reps) \
                    and running:
                # every live rep ended but some are unfinished and gave up
                if all(restarts[r] > a.max_restarts or st.finished(r, t)
                       for r, t in reps):
                    log("no repetition left to run; stopping")
                    st.stop_all()
                    return
        else:
            calm = calm + 1 if sysu <= a.resume_below_gb else 0
            if calm >= 3 and time.time() - stopped_at >= backoff:
                log(f"machine {sysu:.1f} GB for {calm} polls; restarting "
                    "slowly")
                if start_slowly():
                    running = True
                    backoff = a.backoff
                else:
                    stopped_at, calm = time.time(), 0
                    backoff = min(backoff * 2, 3600.0)
                    log(f"restart aborted; next try in {backoff / 60:.0f} "
                        "min at the earliest")
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
