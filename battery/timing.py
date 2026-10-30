"""Per-trial timing record (S3-BAT-01.5.4): callback time, refresh estimate and
photodiode onset are stored as separate fields and never merged."""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class TrialTimingRecord:
    trial_index: int
    requested_onset_ms: float
    callback_time_ms: float  # performance.now() inside the rAF callback
    refresh_estimate_ms: float  # measured frame interval for this session
    photodiode_onset_ms: Optional[float] = None  # hardware ground truth, when present
    visibility_changed: bool = False
    timing_provenance: str = "raf_callback"

    def __post_init__(self) -> None:
        if self.refresh_estimate_ms <= 0:
            raise ValueError("refresh_estimate_ms must be positive")
        if (
            self.photodiode_onset_ms is not None
            and self.timing_provenance == "raf_callback"
        ):
            object.__setattr__(self, "timing_provenance", "photodiode")

    @property
    def callback_error_ms(self) -> float:
        return self.callback_time_ms - self.requested_onset_ms

    @property
    def photodiode_error_ms(self) -> Optional[float]:
        if self.photodiode_onset_ms is None:
            return None
        return self.photodiode_onset_ms - self.requested_onset_ms

    def to_dict(self) -> dict:
        d = asdict(self)
        d["callback_error_ms"] = self.callback_error_ms
        d["photodiode_error_ms"] = self.photodiode_error_ms
        return d
