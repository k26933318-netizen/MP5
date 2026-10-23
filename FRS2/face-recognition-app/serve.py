from waitress import serve

from face_verification.api import create_app
from face_verification.config import load_settings


def main() -> None:
    """Create the service lazily and serve one process."""
    settings = load_settings()
    app = create_app(settings)
    serve(app, host=settings.host, port=settings.port, threads=settings.waitress_threads)


if __name__ == "__main__":
    main()