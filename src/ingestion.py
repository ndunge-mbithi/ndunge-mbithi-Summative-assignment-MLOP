"""Bulk labelled-data ingestion.

Each uploaded file goes through: size/format/corruption validation -> SHA-256 de-dup
(against everything ever uploaded) -> raw bytes persisted under ``storage/uploads`` ->
the *same* preprocessing as training (``to_uint8_array``) persisted as ``.npy`` ->
a ``samples`` row with label, lineage and image statistics. Rejections are recorded
per file with the reason, so the UI can show exactly what was refused and why.

Two input shapes are accepted: a list of image files with one label for the batch,
or a ZIP whose top-level folders are class names (``Parasitized/``, ``Uninfected/``).
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

from .config import CLASS_NAMES, Settings
from .data_store import Store, new_id
from .preprocessing import ImageValidationError, compute_stats, load_image, sha256_bytes, to_uint8_array

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
MAX_ZIP_MEMBERS = 5000


def normalise_label(label: str | None) -> str:
    if label is None:
        raise ValueError("label is required")
    for name in CLASS_NAMES:
        if label.strip().lower() == name.lower():
            return name
    raise ValueError(f"unknown label '{label}'; expected one of {list(CLASS_NAMES)}")


@dataclass
class IngestReport:
    batch_id: str
    received: int = 0
    accepted: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"batch_id": self.batch_id, "received": self.received, "accepted": len(self.accepted),
                "rejected": len(self.rejected), "duplicates": len(self.duplicates),
                "accepted_items": self.accepted[:50], "rejected_items": self.rejected, "duplicate_items": self.duplicates[:50]}


def iter_zip(data: bytes):
    """Yield (filename, label, bytes) from a class-folder ZIP, ignoring OS metadata files."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("archive is not a valid ZIP file") from exc
    members = [m for m in zf.infolist() if not m.is_dir()]
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError(f"archive has {len(members)} files; limit is {MAX_ZIP_MEMBERS}")
    for m in members:
        p = PurePosixPath(m.filename)
        if any(part.startswith(("__MACOSX", ".")) for part in p.parts) or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        folders = [x for x in p.parts[:-1]]
        label = next((f for f in reversed(folders) if f.lower() in {c.lower() for c in CLASS_NAMES}), None)
        if m.file_size > 20 * 1024 * 1024:
            yield p.name, label, b""  # will be rejected as too large/empty without reading
            continue
        yield p.name, label, zf.read(m)


class Ingestor:
    def __init__(self, settings: Settings, store: Store):
        self.s = settings
        self.store = store
        self.s.uploads_dir.mkdir(parents=True, exist_ok=True)

    def ingest(self, items: list[tuple[str, str | None, bytes]], default_label: str | None = None,
               note: str = "", source: str = "upload") -> IngestReport:
        batch_id = new_id("batch")
        self.store.create_batch(batch_id, default_label, note)
        rep = IngestReport(batch_id)
        bdir = self.s.uploads_dir / batch_id
        for name, label, data in items:
            rep.received += 1
            try:
                lbl = normalise_label(label or default_label)
            except ValueError as exc:
                rep.rejected.append({"file": name, "reason": str(exc)})
                continue
            try:
                img = load_image(data, max_bytes=self.s.max_upload_bytes, min_side=self.s.min_image_side,
                                 max_side=self.s.max_image_side)
            except ImageValidationError as exc:
                rep.rejected.append({"file": name, "reason": str(exc)})
                continue
            sha = sha256_bytes(data)
            if self.store.sample_exists(sha):
                rep.duplicates.append({"file": name, "sha256": sha[:16]})
                continue
            bdir.mkdir(parents=True, exist_ok=True)
            suffix = Path(name).suffix.lower() if Path(name).suffix.lower() in IMAGE_SUFFIXES else ".png"
            raw_path = bdir / f"{sha[:24]}{suffix}"
            raw_path.write_bytes(data)
            stats = compute_stats(img)
            sid = self.store.add_sample(sha256=sha, label=lbl, batch_id=batch_id, original_name=name[:200],
                                        raw_path=str(raw_path.relative_to(self.s.storage_dir)), width=img.width,
                                        height=img.height, stats_json=json.dumps(stats.as_dict()), source=source,
                                        status="validated")
            if sid is None:  # lost a race with a concurrent upload of the same file
                raw_path.unlink(missing_ok=True)
                rep.duplicates.append({"file": name, "sha256": sha[:16]})
                continue
            arr_path = raw_path.with_suffix(".npy")
            np.save(arr_path, to_uint8_array(img, self.s.image_size))
            self.store.update_sample(sid, status="preprocessed",
                                     array_path=str(arr_path.relative_to(self.s.storage_dir)))
            rep.accepted.append({"file": name, "sample_id": sid, "label": lbl})
        self.store.finish_batch(batch_id, rep.received, len(rep.accepted), len(rep.rejected), len(rep.duplicates),
                                rep.rejected)
        return rep
