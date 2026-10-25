"""Small, portable run records. Never capture arbitrary environment variables."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_value(*args):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(["git", "-C", str(root), *args],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def start_run(config_path: str, output_path: str):
    raw = Path(config_path).read_bytes()
    config = json.loads(raw)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(raw).hexdigest(), "config": config,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "python": platform.python_version(), "machine": platform.machine(),
        "hostname": platform.node(), "command": sys.argv,
        "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).resolve().parent.glob("*.py"))},
        "packages": versions, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "classification": "exploratory pilot; not confirmatory evidence",
    }
    (output / "config.json").write_bytes(raw)
    save_manifest(output, manifest)
    return config, output, manifest


def save_manifest(output, manifest):
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "manifest.json")


def finish_run(output, manifest):
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["output_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"
    }
    save_manifest(output, manifest)
