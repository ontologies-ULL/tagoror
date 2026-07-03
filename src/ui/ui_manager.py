import asyncio
import sys
import time
from typing import Optional

from .status_reporter import StatusReporter


class UIManager:
    """
    Centralizes console output and spinner UI.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._spinner_task: Optional[asyncio.Task] = None

    def info(self, msg: str):
        if self.verbose:
            print(msg)

    def warn(self, msg: str):
        if self.verbose:
            print(f"[WARN] {msg}")

    def error(self, msg: str):
        if self.verbose:
            print(f"[ERROR] {msg}")

    async def _spinner_coroutine(self, task: asyncio.Task, filename: str, status_reporter: Optional[StatusReporter]):
        start_time = time.time()
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0

        while not task.done():
            elapsed = time.time() - start_time
            entity_label = ""
            if status_reporter is not None:
                try:
                    current_entity, current_task = await status_reporter.get()
                    if current_entity or current_task:
                        entity_label = f" | {current_entity or ''}{' - ' if (current_entity and current_task) else ''}{current_task or ''}"
                except Exception:
                    entity_label = ""

            sys.stdout.write(f"\r   {spinner_chars[i % len(spinner_chars)]} Processing '{filename}'{entity_label}... [{elapsed:.1f}s elapsed]")
            sys.stdout.flush()
            i += 1
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

        # Clear the spinner line completely once the task finishes
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()

    def start_spinner(self, task: asyncio.Task, filename: str, status_reporter: Optional[StatusReporter] = None) -> asyncio.Task:
        """Start a background spinner task and return it."""
        loop = asyncio.get_event_loop()
        self._spinner_task = loop.create_task(self._spinner_coroutine(task, filename, status_reporter))
        return self._spinner_task

    async def stop_spinner(self):
        if self._spinner_task:
            try:
                self._spinner_task.cancel()
                await self._spinner_task
            except Exception:
                pass
            self._spinner_task = None
