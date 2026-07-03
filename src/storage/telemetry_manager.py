import os
from pathlib import Path
from datetime import datetime
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter


def _default_human_readable_formatter(span: ReadableSpan) -> str:
    """
    Formats a span into a human-readable string for logging purposes.
    """
    status_name = span.status.status_code.name
    icon = "🛑" if status_name == "ERROR" else "✅"
    duration_ms = 0.0
    if span.end_time and span.start_time:
        duration_ms = (span.end_time - span.start_time) / 1e6

    output = f"[{status_name}] {icon} Span: '{span.name}' | Duration: {duration_ms:.2f}ms\n"
    if span.attributes:
        output += "  Attributes:\n"
        for key, value in span.attributes.items():
            output += f"    - {key}: {value}\n"
    if span.events:
        output += "  Events/Errors:\n"
        for event in span.events:
            if event.attributes and "exception.type" in event.attributes:
                exc_type = event.attributes.get("exception.type")
                exc_msg = event.attributes.get("exception.message")
                output += f"    -> {exc_type}: {exc_msg}\n"
            else:
                output += f"    -> Event: {event.name}\n"
    output += "-" * 60 + "\n"
    return output


class TelemetryManager:
    """
    Manages telemetry logging for the application using OpenTelemetry.
    It sets up a tracer provider and configures span processors for both human-readable
    and detailed export formats. Files are placed under `logs/` with timestamps.
    """

    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.logs_dir = self.base_path / "logs"
        self.readable_path = None
        self.detailed_path = None
        self._readable_file = None
        self._detailed_file = None

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def setup(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = self._timestamp()
        self.readable_path = self.logs_dir / f"telemetry_readable_{ts}.log"
        self.detailed_path = self.logs_dir / f"telemetry_detailed_{ts}.json"

        self._readable_file = open(self.readable_path, "w", encoding="utf-8")
        self._detailed_file = open(self.detailed_path, "w", encoding="utf-8")

        provider = TracerProvider()

        readable_exporter = ConsoleSpanExporter(out=self._readable_file, formatter=_default_human_readable_formatter)
        provider.add_span_processor(SimpleSpanProcessor(readable_exporter))

        detailed_exporter = ConsoleSpanExporter(out=self._detailed_file)
        provider.add_span_processor(SimpleSpanProcessor(detailed_exporter))

        trace.set_tracer_provider(provider)

        return {
            "readable": str(self.readable_path),
            "detailed": str(self.detailed_path),
        }

    def close(self):
        try:
            if self._readable_file and not self._readable_file.closed:
                self._readable_file.close()
        finally:
            try:
                if self._detailed_file and not self._detailed_file.closed:
                    self._detailed_file.close()
            except Exception:
                pass
