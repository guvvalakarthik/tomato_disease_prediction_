from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    model_path: Path
    metadata_path: Path
    allowed_origins: tuple[str, ...]
    max_upload_bytes: int
    max_image_pixels: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.getenv(
            "TOMATOGUARD_ALLOWED_ORIGINS", "http://localhost:3000"
        )
        return cls(
            model_path=_project_path(
                os.getenv("TOMATOGUARD_MODEL_PATH", "model/tomatoes.h5")
            ),
            metadata_path=_project_path(
                os.getenv("TOMATOGUARD_METADATA_PATH", "model/metadata.json")
            ),
            allowed_origins=tuple(
                origin.strip() for origin in origins.split(",") if origin.strip()
            ),
            max_upload_bytes=int(
                os.getenv("TOMATOGUARD_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))
            ),
            max_image_pixels=int(
                os.getenv("TOMATOGUARD_MAX_IMAGE_PIXELS", "25000000")
            ),
            log_level=os.getenv("TOMATOGUARD_LOG_LEVEL", "INFO").upper(),
        )
