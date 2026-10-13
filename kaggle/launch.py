"""Build / push / monitor Kaggle kernels and the private data dataset.

  python kaggle/launch.py data-create            # first upload of the cache as a private dataset
  python kaggle/launch.py data-version -m "msg"  # new version of the cache dataset
  python kaggle/launch.py push --name pilot --jobs configs/jobs/pilot.yaml
  python kaggle/launch.py status --name pilot
  python kaggle/launch.py pull --name pilot      # -> results/kaggle/<name>/
All kernels are private. The Kaggle credential file is used by the official CLI only.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MAIN_USER = "dev123123456"
USER = os.environ.get("MEDLEAF_KAGGLE_USER", MAIN_USER)  # second account: set this + KAGGLE_API_TOKEN
DATA_SLUG = "medleaf-esa-cache"
STAGE = ROOT / "kaggle" / "_stage"


def kaggle(*args) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise SystemExit(out)
    return out


def code_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "src").rglob("*.py")):
            z.write(p, p.relative_to(ROOT).as_posix())
        for p in sorted((ROOT / "configs").rglob("*.yaml")):
            z.write(p, p.relative_to(ROOT).as_posix())
    return base64.b64encode(buf.getvalue()).decode()


def stage_data() -> Path:
    d = STAGE / "data"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    zpath = d / "medleaf_cache.zip"
    cache = ROOT / "cache"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for f in ["index.csv", "audit.json"]:
            z.write(cache / f, f"cache/{f}")
        for sub in ["img", "mask", "vein"]:
            for p in sorted((cache / sub).iterdir()):
                z.write(p, f"cache/{sub}/{p.name}")
    meta = dict(title="MedLeaf ESA cache", id=f"{USER}/{DATA_SLUG}",
                licenses=[dict(name="CC-BY-4.0")],
                description="Preprocessed crops/masks/vein pseudo-labels of the Mendeley Medicinal Leaf "
                            "Dataset (segmented subset; Roopashree & Anitha 2020, DOI 10.17632/nnytj2v3n5.1, "
                            "CC BY 4.0). Private working copy.")
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    return d


def stage_dimpsar() -> Path:
    """Copy of the maskhead kernel's DIMPSAR build (results/kaggle/maskhead/dimpsar) as a private dataset."""
    d = STAGE / "dimpsar"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = ROOT / "results" / "kaggle" / "maskhead" / "dimpsar"
    with zipfile.ZipFile(d / "dimpsar_build.zip", "w", zipfile.ZIP_STORED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, "dimpsar/" + p.relative_to(src).as_posix())
    meta = dict(title="MedLeaf ESA maskhead", id=f"{USER}/{kernel_slug('maskhead')}", licenses=[dict(name="CC-BY-4.0")],
                description="Leaf-centred crops of DIMPSAR (Pushpa et al., CC BY 4.0) built with our own masker. "
                            "Private working copy.")
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
    return d


def kernel_slug(name: str) -> str:
    """Kaggle turns underscores into hyphens in kernel slugs."""
    return f"medleaf-esa-{name.replace('_', '-')}"


def build_kernel(name: str, jobs: list, pip: list[str], extra_sources: list[str], kernel_sources: list[str],
                 prefetch: list[str] = ()) -> Path:
    d = STAGE / "kernels" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = (ROOT / "kaggle" / "runner_template.py").read_text()
    src = src.replace("__CODE_B64__", code_zip_b64()).replace("__JOBS_JSON__", json.dumps(jobs)) \
             .replace("__PIP__", json.dumps(pip)).replace("__PREFETCH__", json.dumps(list(prefetch)))
    slug = kernel_slug(name)
    (d / f"{slug}.py").write_text(src)
    meta = dict(id=f"{USER}/{slug}", title=f"medleaf esa {name}", code_file=f"{slug}.py", language="python",
                kernel_type="script", is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
                dataset_sources=[f"{USER}/{DATA_SLUG}"] + extra_sources, competition_sources=[],
                kernel_sources=[f"{USER}/{kernel_slug(k)}" for k in kernel_sources] if USER == MAIN_USER else [],
                model_sources=[], machine_shape="NvidiaTeslaT4")
    if USER != MAIN_USER:  # other accounts cannot read private kernel outputs: use dataset copies instead
        meta["dataset_sources"] += [f"{USER}/{kernel_slug(k)}" for k in kernel_sources]
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return d


def jobs_from_yaml(path: Path) -> tuple[list[list[str]], list[str], list[str]]:
    spec = yaml.safe_load(path.read_text())
    common = spec.get("common", {})
    jobs = []
    for j in spec["jobs"]:
        if "chain" in j:  # explicit list of module commands, run sequentially
            jobs.append([[str(x) for x in c] for c in j["chain"]])
            continue
        cfg = {**common, **j}
        args = ["src.train", "--set"] + [f"{k}={json.dumps(v)}" for k, v in cfg.items()]
        jobs.append(args)
    backbones = sorted({str(j.get("backbone", common.get("backbone", "convnext_tiny"))) for j in spec["jobs"]
                        if "chain" not in j})
    prefetch = spec.get("prefetch", backbones + (["lpips"] if "lpips" in spec.get("pip", ["lpips"]) else []))
    return jobs, spec.get("pip", ["lpips"]), spec.get("extra_sources", []), spec.get("kernel_sources", []), prefetch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data-create", "data-version", "dimpsar-create", "push", "status", "pull", "build"])
    ap.add_argument("--name")
    ap.add_argument("--jobs")
    ap.add_argument("-m", "--message", default="update")
    a = ap.parse_args()
    if a.cmd == "data-create":
        print(kaggle("datasets", "create", "-p", str(stage_data()), "-r", "skip"))
    elif a.cmd == "dimpsar-create":
        print(kaggle("datasets", "create", "-p", str(stage_dimpsar()), "-r", "skip"))
    elif a.cmd == "data-version":
        print(kaggle("datasets", "version", "-p", str(stage_data()), "-m", a.message, "-r", "skip"))
    elif a.cmd in ("push", "build"):
        jobs, pip, extra, ksrc, pre = jobs_from_yaml(ROOT / a.jobs)
        d = build_kernel(a.name, jobs, pip, extra, ksrc, pre)
        print(f"built {d} with {len(jobs)} jobs")
        if a.cmd == "push":
            print(kaggle("kernels", "push", "-p", str(d)))
    elif a.cmd == "status":
        print(kaggle("kernels", "status", f"{USER}/{kernel_slug(a.name)}"))
    elif a.cmd == "pull":
        out = ROOT / "results" / "kaggle" / a.name
        out.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{USER}/{kernel_slug(a.name)}", "-p", str(out)))


if __name__ == "__main__":
    main()
