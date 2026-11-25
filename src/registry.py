"""File-based model registry with versioning, promotion and rollback.

Layout::

    models/
      registry.json            {"active": "v2", "versions": {...}, "events": [...]}
      v1/model.keras           serialised Keras model
      v1/metadata.json         metrics, data lineage, hyperparameters, reference stats

``registry.json`` is the single source of truth for which model serves traffic. It is
written atomically (temp file + rename) under an exclusive file lock, so concurrent
API replicas and the retraining worker never observe a half-written registry. API
replicas watch its mtime and hot-reload when the active version changes.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STATUSES = {"candidate", "active", "archived", "rejected"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "registry.json"
        self.lock_path = self.dir / ".registry.lock"

    # ------------------------------------------------------------------ internals
    @contextmanager
    def _locked(self):
        with open(self.lock_path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _read(self) -> dict:
        if not self.path.exists():
            return {"active": None, "versions": {}, "events": []}
        return json.loads(self.path.read_text())

    def _write(self, reg: dict) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=".registry.", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(reg, fh, indent=2)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; other users/containers must be able to read it
        os.replace(tmp, self.path)

    # --------------------------------------------------------------------- reads
    def state(self) -> dict:
        return self._read()

    def mtime(self) -> float:
        return self.path.stat().st_mtime if self.path.exists() else 0.0

    def active_version(self) -> str | None:
        return self._read()["active"]

    def model_path(self, version: str) -> Path:
        return self.dir / version / "model.keras"

    def metadata(self, version: str) -> dict:
        return json.loads((self.dir / version / "metadata.json").read_text())

    def next_version(self) -> str:
        nums = [int(v[1:]) for v in self._read()["versions"] if v[1:].isdigit()]
        on_disk = [int(p.name[1:]) for p in self.dir.glob("v*") if p.name[1:].isdigit()]
        return f"v{max(nums + on_disk + [0]) + 1}"

    # -------------------------------------------------------------------- writes
    def register(self, model, metadata: dict, status: str = "candidate", version: str | None = None) -> str:
        """Serialise ``model`` and its metadata as a new immutable version."""
        if status not in STATUSES:
            raise RegistryError(f"bad status {status}")
        with self._locked():
            reg = self._read()
            version = version or self.next_version()
            vdir = self.dir / version
            if vdir.exists():
                raise RegistryError(f"{version} already exists")
            tmp_dir = Path(tempfile.mkdtemp(dir=self.dir, prefix=f".{version}."))
            model.save(tmp_dir / "model.keras")
            meta = {**metadata, "version": version, "created_at": utcnow()}
            (tmp_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
            os.chmod(tmp_dir, 0o755)  # mkdtemp creates 0700
            os.replace(tmp_dir, vdir)
            reg["versions"][version] = {"status": status, "created_at": meta["created_at"],
                                        "parent": metadata.get("parent_version"),
                                        "summary": metadata.get("summary", {})}
            reg["events"].append({"at": utcnow(), "event": "registered", "version": version, "status": status})
            self._write(reg)
        return version

    def set_status(self, version: str, status: str, reason: str = "") -> None:
        with self._locked():
            reg = self._read()
            if version not in reg["versions"]:
                raise RegistryError(f"unknown version {version}")
            reg["versions"][version]["status"] = status
            reg["events"].append({"at": utcnow(), "event": status, "version": version, "reason": reason})
            self._write(reg)

    def promote(self, version: str, reason: str = "") -> None:
        """Make ``version`` the serving model; the previous active one is archived."""
        with self._locked():
            reg = self._read()
            if version not in reg["versions"]:
                raise RegistryError(f"unknown version {version}")
            if not self.model_path(version).exists():
                raise RegistryError(f"artifact for {version} missing")
            prev = reg["active"]
            if prev and prev != version:
                reg["versions"][prev]["status"] = "archived"
            reg["versions"][version]["status"] = "active"
            reg["active"] = version
            reg["events"].append({"at": utcnow(), "event": "promoted", "version": version,
                                  "previous": prev, "reason": reason})
            self._write(reg)

    def rollback(self, reason: str = "manual rollback") -> str:
        """Re-activate the most recently active version before the current one."""
        reg = self._read()
        current = reg["active"]
        previous = [e["previous"] for e in reversed(reg["events"])
                    if e["event"] == "promoted" and e.get("version") == current and e.get("previous")]
        if not previous:
            raise RegistryError("no previous active version to roll back to")
        self.promote(previous[0], reason=reason)
        return previous[0]

    def delete_version(self, version: str) -> None:
        with self._locked():
            reg = self._read()
            if reg["active"] == version:
                raise RegistryError("cannot delete the active version")
            reg["versions"].pop(version, None)
            shutil.rmtree(self.dir / version, ignore_errors=True)
            self._write(reg)
