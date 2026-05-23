"""
FastAPI application factory for the LLM Inference Service.
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from monitoring.metrics import setup_metrics

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Inference Service",
        description="High-throughput real-time LLM inference with token streaming and semantic caching.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics middleware
    if os.getenv("ENABLE_METRICS", "true").lower() == "true":
        setup_metrics(app)

    # Startup / shutdown
    @app.on_event("startup")
    async def startup():
        logger.info("LLM Inference Service starting up...")
        app.state.start_time = time.time()

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("LLM Inference Service shutting down...")

    # Health endpoint
    @app.get("/health", tags=["ops"])
    async def health():
        uptime = time.time() - getattr(app.state, "start_time", time.time())
        return JSONResponse({"status": "ok", "uptime_seconds": round(uptime, 1)})

    # Routes
    app.include_router(router, prefix="/v1")

    return app


app = create_app()
