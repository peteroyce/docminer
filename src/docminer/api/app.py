"""FastAPI application factory."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singletons (initialised by create_app / lifespan)
_pipeline = None
_storage = None


def create_app(config=None):
    """Create and return the FastAPI application.

    Parameters
    ----------
    config:
        A :class:`~docminer.config.schema.DocMinerConfig` instance.
        If *None*, the default config is used.

    Returns
    -------
    fastapi.FastAPI
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    import docminer
    from docminer.api.routes import router
    from docminer.config.schema import DocMinerConfig

    if config is None:
        config = DocMinerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        """Initialise shared resources on startup; clean up on shutdown."""
        global _pipeline, _storage

        # Initialise pipeline
        from docminer.core.pipeline import Pipeline

        _pipeline = Pipeline(config=config)
        logger.info("Pipeline initialised")

        # Initialise storage
        if config.storage.backend == "sqlite":
            try:
                from docminer.storage.backend import SQLiteBackend

                _storage = SQLiteBackend(db_path=config.storage.db_path)
                logger.info("SQLiteBackend initialised at %s", config.storage.db_path)
            except Exception as exc:
                logger.warning("Storage initialisation failed: %s", exc)
                _storage = None
        else:
            _storage = None

        yield

        # Cleanup
        if _storage is not None:
            _storage.close()
        logger.info("DocMiner API shutdown")

    app = FastAPI(
        title="DocMiner API",
        description=(
            "Document intelligence pipeline — extract, classify, and analyse "
            "PDFs, images, and scanned documents."
        ),
        version=docminer.__version__,
        lifespan=lifespan,
    )

    # CORS (permissive for development; restrict in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(router, prefix="/api/v1")

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": "DocMiner API",
            "version": docminer.__version__,
            "docs": "/docs",
        }

    return app
