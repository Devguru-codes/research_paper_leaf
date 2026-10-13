"""Run Kaggle kernels strictly one at a time (pushing while one runs kills it).

  python kaggle/kqueue.py main_cnx_s0 main_cnx_s12 main_vit ...
For each name: push configs/jobs/<name>.yaml, poll until terminal, pull outputs into
results/kaggle/<name>/, append a line to results/kaggle/queue_log.txt. A kernel that ends in
ERROR is retried once (fresh version).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
try:  # verify TLS against the Windows certificate store (kept current by Windows Update), not only certifi
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
ROOT = Path(__file__).resolve().parents[1]
USER = os.environ.get("MEDLEAF_KAGGLE_USER", "dev123123456")
LOG = ROOT / "results" / "kaggle" / ("queue_log.txt" if USER == "dev123123456" else f"queue_log_{USER}.txt")


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def status(name):
    for _ in range(5):
        r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", f"{USER}/medleaf-esa-{name.replace('_', '-')}"],
                           capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip().splitlines()
        if out and "status" in out[-1]:
            return out[-1].split('"')[-2] if '"' in out[-1] else out[-1]
        time.sleep(30)
    return "UNKNOWN"


def pull(name):
    from kaggle.api.kaggle_api_extended import KaggleApi
    out = ROOT / "results" / "kaggle" / name
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        try:
            api = KaggleApi()
            api.authenticate()
            api.kernels_output(f"{USER}/medleaf-esa-{name.replace('_', '-')}", path=str(out), file_pattern=r".*\.(log|json|csv|npz|pt)$")
            return True
        except Exception as e:  # network hiccups
            log(f"pull {name} failed: {e}; retrying")
            time.sleep(60)
    return False


def run(name):
    s0 = status(name)
    if "COMPLETE" in s0 and any((ROOT / "results" / "kaggle" / name).rglob("metrics.json")):
        log(f"{name} already COMPLETE and pulled; skipping")  # safe to restart the queue
        return True
    if "COMPLETE" in s0 and "--fresh" not in sys.argv:  # finished earlier but the pull failed: pull, don't re-run
        log(f"{name} already COMPLETE, not pulled yet; pulling: {'ok' if pull(name) else 'FAILED'}")
        return True
    if any(k in s0 for k in ("RUNNING", "QUEUED")):  # adopt a kernel that is already running
        log(f"{name} already {s0}; waiting without re-pushing")
        while True:
            s = status(name)
            if any(k in s for k in ("COMPLETE", "ERROR", "CANCEL")):
                break
            time.sleep(180)
        log(f"{name} finished: {s}")
        if "COMPLETE" in s:
            log(f"pull {name}: {'ok' if pull(name) else 'FAILED'}")
            return True
    for attempt in (1, 2):
        while True:  # a refused push (quota exhausted / session limit) is retried hourly
            r = subprocess.run([sys.executable, str(ROOT / "kaggle" / "launch.py"), "push", "--name", name,
                                "--jobs", f"configs/jobs/{name}.yaml"], capture_output=True, text=True, cwd=ROOT)
            msg = (r.stdout + r.stderr).strip()
            log(f"push {name} attempt {attempt}: {msg.splitlines()[-1:]}")
            if r.returncode == 0 and "error" not in msg.lower():
                break
            time.sleep(3600)
        time.sleep(90)
        for _ in range(10):  # right after a re-push Kaggle may still report the previous version's ERROR
            if any(k in status(name) for k in ("RUNNING", "QUEUED")):
                break
            time.sleep(90)
        while True:
            s = status(name)
            if any(k in s for k in ("COMPLETE", "ERROR", "CANCEL")):
                break
            time.sleep(180)
        log(f"{name} finished: {s}")
        if "COMPLETE" in s:
            log(f"pull {name}: {'ok' if pull(name) else 'FAILED'}")
            return True
        pull(name)  # keep whatever exists
    return False


if __name__ == "__main__":
    for n in [a for a in sys.argv[1:] if not a.startswith("--")]:
        for _ in range(48):  # transient network/TLS errors must not kill the queue
            try:
                run(n)
                break
            except Exception as e:
                log(f"{n}: queue error {type(e).__name__}: {str(e)[:200]}; retrying in 5 min")
                time.sleep(300)
    log("QUEUE DONE")
