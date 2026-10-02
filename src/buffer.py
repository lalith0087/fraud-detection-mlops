"""Buffer of recently scored transactions, used for drift monitoring.

Uses Redis when REDIS_URL is set (survives restarts and is shared across workers), otherwise an
in-memory deque. If Redis becomes unreachable the buffer falls back to memory, so monitoring can
never break /predict.
"""
import json
import os
from collections import deque

KEY = "fraud:recent"


class RecentBuffer:
    def __init__(self, maxlen: int = 1000, client=None, url: str | None = None):
        self.maxlen = maxlen
        self._mem = deque(maxlen=maxlen)
        self._redis = client
        url = url if url is not None else os.getenv("REDIS_URL")
        if self._redis is None and url:
            import redis

            self._redis = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def add(self, row: dict) -> None:
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.rpush(KEY, json.dumps(row))
                pipe.ltrim(KEY, -self.maxlen, -1)
                pipe.execute()
                return
            except Exception:
                self._redis = None  # degrade to memory rather than fail the request
        self._mem.append(row)

    def items(self) -> list[dict]:
        if self._redis is not None:
            try:
                return [json.loads(x) for x in self._redis.lrange(KEY, 0, -1)]
            except Exception:
                self._redis = None
        return list(self._mem)

    def __len__(self) -> int:
        if self._redis is not None:
            try:
                return int(self._redis.llen(KEY))
            except Exception:
                self._redis = None
        return len(self._mem)

    def clear(self) -> None:
        self._mem.clear()
        if self._redis is not None:
            try:
                self._redis.delete(KEY)
            except Exception:
                self._redis = None
