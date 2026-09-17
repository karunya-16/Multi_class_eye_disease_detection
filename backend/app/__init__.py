"""4-class Eye Disease Detection API package.

Start with:

    uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from app.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
