"""Signed, database-backed, single-use verification tokens."""

import secrets
import time
from typing import Any, Callable

import jwt

from .config import DEFAULT_SETTINGS, Settings


class TokenManager:
    """Issue and introspect HS256 tokens with atomic single-consumption state."""

    def __init__(
        self,
        store: Any,
        settings: Settings = DEFAULT_SETTINGS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not settings.face_token_secret:
            raise ValueError("FACE_TOKEN_SECRET is required.")
        self.store = store
        self.settings = settings
        self.clock = clock

    def issue(self, subject_id: str, session_id: str) -> str:
        """Issue a signed verification token and persist its one-use identifier."""
        now = int(self.clock())
        expires_at = now + self.settings.token_ttl_sec
        jti = secrets.token_urlsafe(32)
        self.store.add_token(jti, subject_id, expires_at)
        return jwt.encode(
            {
                "iss": "face-verification",
                "sub": subject_id,
                "jti": jti,
                "iat": now,
                "exp": expires_at,
                "sid": session_id,
                "assurance": "liveness+1:1",
            },
            self.settings.face_token_secret,
            algorithm="HS256",
        )

    def introspect(self, token: str, consume: bool = False) -> dict[str, Any] | None:
        """Validate token signature, issuer and database state; optionally consume once."""
        try:
            claims = jwt.decode(
                token,
                self.settings.face_token_secret,
                algorithms=["HS256"],
                issuer="face-verification",
                options={
                    "verify_exp": False,
                    "verify_iat": False,
                    "require": ["iss", "sub", "jti", "iat", "exp", "sid", "assurance"],
                },
            )
        except jwt.InvalidTokenError:
            return None
        now = self.clock()
        if int(claims["exp"]) <= now or int(claims["iat"]) > now:
            return None
        if not self.store.token_exists(claims["jti"], now):
            return None
        if consume and not self.store.consume_token(claims["jti"], now):
            return None
        return {
            "active": True,
            "subject_id": claims["sub"],
            "session_id": claims["sid"],
            "assurance": claims["assurance"],
            "expires_at": claims["exp"],
        }