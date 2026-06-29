"""Token-bucket rate limiter for API calls."""

from __future__ import annotations

import time
from collections import deque


class RateLimiter:
    """Limits calls to max_calls within a rolling time window."""

    def __init__(self, max_calls: int = 10, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        self._drop_expired(now)

        if len(self._timestamps) >= self.max_calls:
            sleep_for = self.period_seconds - (now - self._timestamps[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self._drop_expired(now)

        self._timestamps.append(now)

    def _drop_expired(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
            self._timestamps.popleft()
