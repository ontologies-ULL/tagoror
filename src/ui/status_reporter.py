import asyncio
from typing import Optional


class StatusReporter:
    """
    Small utility class to track the current entity and task being processed.
    This is useful for providing real-time feedback in the UI, especially when processing large batches of entities.
    It uses an asyncio.Lock to ensure thread-safe updates and retrievals of the current status 
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_entity: Optional[str] = None
        self._current_task: Optional[str] = None

    async def set(self, entity: Optional[str], task: Optional[str]):
        async with self._lock:
            self._current_entity = entity
            self._current_task = task

    async def clear(self):
        async with self._lock:
            self._current_entity = None
            self._current_task = None

    async def get(self):
        async with self._lock:
            return self._current_entity, self._current_task
