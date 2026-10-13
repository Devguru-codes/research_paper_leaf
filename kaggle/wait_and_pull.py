"""Wait for one kernel to finish, pull its log/json outputs, print a summary.

  python kaggle/wait_and_pull.py smoke4
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kqueue import pull, status  # noqa: E402  (kaggle/kqueue.py)

name = sys.argv[1]
while True:
    s = status(name)
    if any(k in s for k in ("COMPLETE", "ERROR", "CANCEL")):
        break
    time.sleep(60)
print(name, s, time.ctime(), flush=True)
pull(name)
root = Path(__file__).resolve().parents[1] / "results" / "kaggle" / name
logf = root / f"medleaf-esa-{name}.log"
if logf.exists():
    L = json.loads(logf.read_text(encoding="utf-8") or "[]")
    t = "".join(x["data"] for x in L)
    print("log entries:", len(L))
    keep = ("job", "[hb", "Traceback", "RuntimeError", "torch.OutOfMemory", "DONE")
    print("\n".join(l[:220] for l in t.splitlines() if l.startswith(keep))[-4000:])
for m in root.rglob("metrics.json"):
    d = json.loads(m.read_text())
    print(m.parent.parent.name, m.parent.name, "train_min=%.2f" % (d["train_time_s"] / 60), "test_acc=%.4f" % d["test"]["acc"])
