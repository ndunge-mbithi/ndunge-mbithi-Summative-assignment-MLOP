"""Central configuration.

Every runtime knob is read from environment variables (prefix ``SMEAR_``) so the
same image runs locally, under docker-compose and on a cloud host without code
changes. Nothing secret lives in this file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Class order is part of the model contract: index 0 / 1 of the sigmoid output.
CLASS_NAMES: tuple[str, ...] = ("Uninfected", "Parasitized")
POSITIVE_CLASS = "Parasitized"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMEAR_", env_file=".env", extra="ignore")

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    storage_dir: Path = PROJECT_ROOT / "storage"
    pretrained_weights: Path = PROJECT_ROOT / "models" / "pretrained" / "mobilenet_v2_1.0_128_no_top.h5"

    # Image contract
    image_size: int = 128
    max_upload_bytes: int = 5 * 1024 * 1024
    max_bulk_files: int = 2000
    min_image_side: int = 16
    max_image_side: int = 2048

    # Inference
    decision_threshold: float = 0.5
    low_confidence_margin: float = 0.15  # |p - 0.5| below this is flagged for review

    # Retraining
    retrain_min_new_samples: int = 20
    retrain_replay_ratio: float = 3.0  # replayed original samples per new sample
    retrain_replay_min: int = 600
    retrain_max_epochs: int = 6
    retrain_learning_rate: float = 1e-5
    retrain_batch_size: int = 32
    retrain_patience: int = 2
    # Acceptance gate on the frozen gold test set
    gate_max_f1_drop: float = 0.005
    gate_max_recall_drop: float = 0.01
    gate_eval_limit: int = 0  # 0 = full gold test set; >0 caps it (used in tests)

    # Monitoring
    drift_window: int = 500
    drift_psi_alert: float = 0.2

    log_level: str = "INFO"
    cors_origins: str = "*"

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "smearscan.db"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def registry_path(self) -> Path:
        return self.models_dir / "registry.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
