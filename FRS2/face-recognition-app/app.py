"""Convenience entry point for the standalone face verification service."""

from face_verification.api import create_app
from face_verification.config import load_settings


def main() -> None:
    """Create the service and run Flask for local development only."""
    settings = load_settings()
    app = create_app(settings)
    app.run(
        host=settings.host,
        port=settings.port,
        debug=settings.flask_debug,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()