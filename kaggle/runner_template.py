# Auto-generated Kaggle runner (see kaggle/launch.py). Code snapshot is embedded below.
import base64
import glob
import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import zipfile

CODE_B64 = "__CODE_B64__"
JOBS = json.loads(r'''__JOBS_JSON__''')
PIP = __PIP__

T0 = time.time()
STOP_NEW_AFTER_H = 10.3  # Kaggle kills sessions at 12 h; stop starting jobs early so outputs are saved
WORK = "/kaggle/working"
PROJ = os.path.join(WORK, "proj")
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))).extractall(PROJ)
os.chdir(PROJ)

# ---- locate the preprocessing cache (directory with index.csv, or a zip of it) ----
idx = glob.glob("/kaggle/input/**/index.csv", recursive=True)
if idx:
    cache = os.path.dirname(idx[0])
else:
    z = glob.glob("/kaggle/input/**/medleaf_cache*.zip", recursive=True)[0]
    cache = "/kaggle/tmp/cache"
    os.makedirs(cache, exist_ok=True)
    zipfile.ZipFile(z).extractall(cache)
    cache = os.path.dirname(glob.glob(cache + "/**/index.csv", recursive=True)[0])
print("cache:", cache, flush=True)
os.environ["MEDLEAF_CACHE"] = cache
os.environ["MEDLEAF_RESULTS"] = os.path.join(WORK, "results")
os.makedirs(os.path.join(WORK, "results", "logs"), exist_ok=True)
if PIP:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + PIP, check=False)

PREFETCH = __PREFETCH__
if PREFETCH:  # download weights once before parallel jobs (avoids concurrent hub-cache writes)
    r = subprocess.run([sys.executable, "-m", "src.prefetch"] + PREFETCH)
    print("prefetch rc", r.returncode, flush=True)

try:
    import torch
    ngpu = max(1, torch.cuda.device_count())
    print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], flush=True)
except Exception:
    ngpu = 1

q = queue.Queue()
for i, j in enumerate(JOBS):
    q.put((i, j))
status = {}


def worker(gpu):
    while True:
        try:
            i, args = q.get_nowait()
        except queue.Empty:
            return
        if time.time() - T0 > STOP_NEW_AFTER_H * 3600:
            status[i] = dict(args=args, rc="skipped_time", minutes=0, gpu=gpu)
            print(f"job {i} skipped: {STOP_NEW_AFTER_H} h session budget reached", flush=True)
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = os.path.join(WORK, "results", "logs", f"job{i:03d}.log")
        t = time.time()
        chain = args if isinstance(args[0], list) else [args]  # a job may be a chain of commands
        with open(log, "w") as fh:
            for cmd in chain:
                fh.write(" ".join(cmd) + "\n")
                fh.flush()
                r = subprocess.run([sys.executable, "-m"] + cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
                if r.returncode != 0:
                    break
        status[i] = dict(args=chain, rc=r.returncode, minutes=round((time.time() - t) / 60, 2), gpu=gpu)
        desc = " | ".join(" ".join(c) for c in chain)[:300]
        print(f"job {i} rc={r.returncode} {status[i]['minutes']}min gpu{gpu}: {desc}", flush=True)
        if r.returncode != 0:
            print(open(log).read()[-3000:], flush=True)


def heartbeat():
    """Every 60 s: host RAM, GPU memory and the last line of each running job log."""
    while any(th.is_alive() for th in threads):
        try:
            mem = open("/proc/meminfo").read().splitlines()
            kv = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in mem if l and l.split()[1].isdigit()}
            used = kv["MemTotal"] - kv["MemAvailable"]
            g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip().replace(chr(10), " | ")
            tails = []
            for f in sorted(glob.glob(os.path.join(WORK, "results", "logs", "job*.log"))):
                lines = [l for l in open(f, errors="ignore").read().splitlines() if l.strip()]
                if lines:
                    tails.append(os.path.basename(f) + ": " + lines[-1][:160])
            print(f"[hb {time.strftime('%H:%M:%S')}] RAM {used}/{kv['MemTotal']} MB | GPU {g}", flush=True)
            for t_ in tails:
                print("   " + t_, flush=True)
        except Exception as e:  # never let the heartbeat kill the run
            print("hb error", e, flush=True)
        time.sleep(60)


threads = [threading.Thread(target=worker, args=(g,)) for g in range(ngpu)]
for th in threads:
    th.start()
hb = threading.Thread(target=heartbeat, daemon=True)
hb.start()
for th in threads:
    th.join()
json.dump(status, open(os.path.join(WORK, "results", "job_status.json"), "w"), indent=1)
print("DONE", json.dumps(status), flush=True)
