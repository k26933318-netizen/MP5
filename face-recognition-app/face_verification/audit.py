"""Small append-only audit facade; biometric data is never accepted."""

from typing import Any


class AuditLogger:
    """Write event metadata through the store's append-only audit table."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def record(self, event: str, subject_id: str | None = None, details: dict[str, Any] | None = None) -> None:
        """Append an event with only subject identifiers and non-biometric metadata."""
        forbidden = {"image", "frame", "embedding", "vector", "name", "email", "phone"}
        safe_details = {
            key: value
            for key, value in (details or {}).items()
            if key.lower() not in forbidden
        }
        self.store.audit(event, subject_id, safe_details)