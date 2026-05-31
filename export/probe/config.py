from __future__ import annotations
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# Minimum pass rate per domain. A run fails if any domain drops below its threshold.
# Blocking probes marked blocking=False are excluded from the threshold denominator.
THRESHOLDS: dict[str, float] = {
    "ADVERSARIAL":  1.00,
    "SCOPE":        1.00,
    "AMBIGUOUS":    1.00,
    "TEMPORAL":     1.00,
    "NOTIFICATION": 1.00,
    "FEED_CARD":    1.00,
    "CALENDAR":     0.91,  # 11/12 — CoT calendar failure accepted
    "DISSONANCE":   0.88,  # 8/9  — duration contradiction accepted
    "ARBITRATION":  0.75,  # 3/4  — new category, baseline TBD after first run
}


@dataclass
class Config:
    """Probe run configuration.

    Priority (highest → lowest): CLI args → env vars → THRESHOLDS defaults.

    Env vars:
      PROBE_MAX_TOKENS              int, default 150
      PROBE_VERBOSE                 1|true|yes to enable verbose output
      PROBE_THRESHOLD_<DOMAIN>      float override per domain,
                                    e.g. PROBE_THRESHOLD_CALENDAR=0.85
    """

    model_path: str
    max_tokens: int = 150
    verbose: bool = False
    thresholds: dict[str, float] = field(default_factory=lambda: dict(THRESHOLDS))

    @classmethod
    def from_env(cls, model_path: str) -> "Config":
        thresholds = dict(THRESHOLDS)
        for domain in list(thresholds):
            val = os.getenv(f"PROBE_THRESHOLD_{domain}")
            if val is not None:
                thresholds[domain] = float(val)
        return cls(
            model_path=model_path,
            max_tokens=int(os.getenv("PROBE_MAX_TOKENS", "150")),
            verbose=os.getenv("PROBE_VERBOSE", "").lower() in ("1", "true", "yes"),
            thresholds=thresholds,
        )

    def apply_cli_overrides(
        self,
        max_tokens: int | None = None,
        verbose: bool | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> "Config":
        return Config(
            model_path=self.model_path,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            verbose=verbose if verbose is not None else self.verbose,
            thresholds={**self.thresholds, **(thresholds or {})},
        )
