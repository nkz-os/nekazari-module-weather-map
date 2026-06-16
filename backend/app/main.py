"""Weather Map — FastAPI tile server and COG generator entrypoint."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.forecast import router as forecast_router
from app.tiles import router as tiles_router

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Weather Map",
    description="Weather-derived raster overlays via COG tiles",
    version="1.0.0",
)

app.include_router(tiles_router, prefix="/api/weather-map")
app.include_router(forecast_router, prefix="/api/weather-map")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    return {"status": "ready"}
