from __future__ import annotations

from pydantic import BaseModel, Field


class FeedbackIn(BaseModel):
    true_label: str = Field(..., description="Confirmed label: Parasitized or Uninfected", examples=["Parasitized"])


class RetrainIn(BaseModel):
    epochs: int | None = Field(default=None, ge=1, le=30)
    learning_rate: float | None = Field(default=None, gt=0, le=1e-2)
    seed: int | None = None
