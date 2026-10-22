"""Small standard-library client for the face verification service API."""

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class ServiceClient:
    """Call authenticated backend endpoints without exposing the API key to browsers."""

    def __init__(self, base_url: str, api_key: str, timeout_sec: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def create_session(
        self,
        subject_id: str,
        purpose: str,
        ttl_sec: int | None = None,
        allow_replace: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an enrollment or verification session."""
        payload: dict[str, Any] = {
            "subject_id": subject_id,
            "purpose": purpose,
            "allow_replace": allow_replace,
        }
        if ttl_sec is not None:
            payload["ttl_sec"] = ttl_sec
        if metadata is not None:
            payload["metadata"] = metadata
        return self._request("POST", "/sessions", payload)

    def get_result(
        self,
        session_id: str,
        wait: bool = False,
        timeout_sec: float = 30.0,
        poll_interval_sec: float = 0.25,
    ) -> dict[str, Any] | None:
        """Read the completed backend result, optionally polling until timeout."""
        deadline = time.monotonic() + timeout_sec
        while True:
            try:
                return self._request("GET", f"/sessions/{quote(session_id, safe='')}/result")
            except ServiceError as error:
                if error.status_code != 404 or not wait:
                    if error.status_code == 404:
                        return None
                    raise
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_interval_sec)

    def introspect_token(self, token: str, consume: bool = True) -> dict[str, Any]:
        """Ask the service to validate and optionally consume a one-time token."""
        return self._request("POST", "/tokens/introspect", {"token": token, "consume": consume})

    def get_subject(self, subject_id: str) -> dict[str, Any]:
        """Read non-biometric subject metadata."""
        return self._request("GET", f"/subjects/{quote(subject_id, safe='')}")

    def delete_subject(self, subject_id: str) -> dict[str, Any]:
        """Delete a subject and its embeddings."""
        return self._request("DELETE", f"/subjects/{quote(subject_id, safe='')}")

    def unlock_subject(self, subject_id: str, reason: str) -> dict[str, Any]:
        """Unlock a locked subject with an auditable reason."""
        return self._request("POST", f"/subjects/{quote(subject_id, safe='')}/unlock", {"reason": reason})

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{self.base_url}/api/v1{path}",
            data=data,
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            try:
                body = json.loads(error.read().decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                body = {}
            raise ServiceError(error.code, body.get("message", "Service request failed.")) from error
        except URLError as error:
            raise ServiceError(0, "Face verification service is unavailable.") from error


class ServiceError(RuntimeError):
    """An HTTP or transport error from the service."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code