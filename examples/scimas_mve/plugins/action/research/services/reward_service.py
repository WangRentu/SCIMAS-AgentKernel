from typing import Dict, Optional


class RewardService:
    """Shared reward helpers for operator modules."""

    @staticmethod
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, float(value)))

    @staticmethod
    def merge(base: Optional[Dict[str, float]] = None, extra: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        data: Dict[str, float] = dict(base or {})
        for key, val in (extra or {}).items():
            data[str(key)] = float(val)
        return data
