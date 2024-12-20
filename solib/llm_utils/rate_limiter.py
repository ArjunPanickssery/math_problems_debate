import time
import threading
from typing import Dict

class RateLimiter:
    """Simple token bucket rate limiter"""
    def __init__(self, requests_per_minute: float, max_bucket_size: int = 60):
        self.rate = requests_per_minute / 60  # convert to per second
        self.max_bucket_size = max_bucket_size
        self.tokens = max_bucket_size
        self.last_update = time.time()
        self.lock = threading.Lock()

    def _add_tokens(self):
        now = time.time()
        time_passed = now - self.last_update
        new_tokens = time_passed * self.rate
        self.tokens = min(self.tokens + new_tokens, self.max_bucket_size)
        self.last_update = now

    def acquire(self):
        with self.lock:
            self._add_tokens()
            while self.tokens < 1:
                time.sleep(0.1)
                self._add_tokens()
            self.tokens -= 1

RATE_LIMITS = {
    "gpt": 500/60,  # 500 RPM for GPT-4
    "claude": 1000/60,  # 1000 RPM for Claude
    "mistral": 500/60,  # 500 RPM for Mistral
}

rate_limiters: Dict[str, RateLimiter] = {}

def get_rate_limiter(model: str) -> RateLimiter:
    """Get or create a rate limiter for the given model"""
    model_type = model.split("-")[0]
    if model_type not in rate_limiters:
        rate_limiters[model_type] = RateLimiter(RATE_LIMITS[model_type])
    return rate_limiters[model_type] 