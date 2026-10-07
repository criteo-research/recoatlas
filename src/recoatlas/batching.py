"""Search batching: concurrent searches share one encoder pass."""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future


class SearchBatcher:
    """Runs `search_many` on a worker thread, batching every request that is waiting.

    A request that arrives while the encoder is idle runs immediately (no waiting window);
    requests that arrive while a batch is running are encoded together in the next batch.
    """

    def __init__(self, search_many, max_batch: int = 64):
        self._search_many = search_many
        self._max_batch = max_batch
        self._requests: queue.SimpleQueue[tuple[str, int, Future]] = queue.SimpleQueue()
        threading.Thread(target=self._serve, name="recoatlas-search-batcher", daemon=True).start()

    def search(self, query: str, top_k: int) -> str:
        future: Future = Future()
        self._requests.put((query, top_k, future))
        return future.result()

    def _serve(self) -> None:
        while True:
            batch = [self._requests.get()]
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._requests.get_nowait())
                except queue.Empty:
                    break
            try:
                outputs = self._search_many([q for q, _, _ in batch], [k for _, k, _ in batch])
            except Exception as error:  # noqa: BLE001 - the worker must survive; callers get the error
                for _, _, future in batch:
                    future.set_exception(error)
                continue
            for (_, _, future), output in zip(batch, outputs, strict=True):
                future.set_result(output)
