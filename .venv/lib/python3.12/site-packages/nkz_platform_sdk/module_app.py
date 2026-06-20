"""
ModuleApp — FastAPI subclass pre-wired for Nekazari module backends.

What you write:
    from nkz_platform_sdk import ModuleApp, AuthContext

    app = ModuleApp(id="soil-health", description="Soil Health backend")

    @app.get("/parcels/{parcel_id}/analysis")
    async def analysis(parcel_id: str, ctx: AuthContext = app.auth()):
        orion = app.orion(ctx)
        return await orion.get_entity(parcel_id)

What you get for free:
- CORS configured from `ALLOWED_ORIGINS` env (comma-separated, default same-origin only).
- `/health` and `/ready` endpoints exempt from auth and rate limit.
- JSON structured logs with `tenant_id`, `user_id`, `module_id`, `trace_id` (when reachable).
- OpenAPI at `/openapi.json` (FastAPI native — keep your endpoints typed for free docs).
- `app.auth(roles=...)` shortcut → `require_auth(roles)`.
- `app.orion(ctx)` → an `OrionClient` scoped to the authenticated tenant.
- `app.timescale(ctx)` → a `TimescaleClient` scoped to the authenticated tenant.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Sequence

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from nkz_platform_sdk.auth import AuthContext, require_auth
from nkz_platform_sdk.orion import OrionClient
from nkz_platform_sdk.timescale import TimescaleClient


def _parse_origins(env_value: str | None) -> list[str]:
    if not env_value:
        return []
    return [o.strip() for o in env_value.split(",") if o.strip()]


class _JsonFormatter(logging.Formatter):
    """Single-line JSON log records — Loki / Cloud Logging / Datadog friendly."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for attr in ("module_id", "tenant_id", "user_id", "trace_id"):
            v = getattr(record, attr, None)
            if v is not None:
                payload[attr] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_json_logging(module_id: str) -> None:
    """Replace the root logger's handlers with one that emits JSON.
    Idempotent — re-applying ModuleApp() in tests won't keep stacking handlers.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # Attach module_id to every record via a filter
    module_filter = logging.Filter()
    module_filter.filter = lambda r: setattr(r, "module_id", module_id) or True  # type: ignore[assignment]
    root.addFilter(module_filter)


class ModuleApp(FastAPI):
    """FastAPI app pre-cabled with everything a Nekazari module backend needs."""

    def __init__(
        self,
        id: str,
        description: str = "",
        version: str = "0.1.0",
        allowed_origins: Sequence[str] | None = None,
        configure_logging: bool = True,
        **fastapi_kwargs: Any,
    ) -> None:
        super().__init__(
            title=f"Nekazari module: {id}",
            description=description,
            version=version,
            **fastapi_kwargs,
        )
        self.module_id = id

        if configure_logging:
            _configure_json_logging(id)

        origins = (
            list(allowed_origins)
            if allowed_origins is not None
            else _parse_origins(os.getenv("ALLOWED_ORIGINS"))
        )
        if origins:
            self.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                allow_headers=[
                    "Authorization",
                    "Content-Type",
                    "X-Tenant-ID",
                    "X-Module-Id",
                    "X-User-ID",
                    "X-User-Roles",
                    "X-Request-ID",
                    "Cookie",
                ],
            )

        # Request-scope trace id + access log
        @self.middleware("http")
        async def _request_scope(request: Request, call_next: Any) -> Any:
            trace_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
            start = time.monotonic()
            response = await call_next(request)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logging.getLogger("nkz.access").info(
                "%s %s %s in %dms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                extra={
                    "tenant_id": request.headers.get("X-Tenant-ID"),
                    "user_id": request.headers.get("X-User-ID"),
                    "trace_id": trace_id,
                },
            )
            response.headers["X-Request-ID"] = trace_id
            return response

        @self.get("/health", include_in_schema=False)
        async def _health() -> dict[str, str]:
            return {"status": "ok", "module": id}

        @self.get("/ready", include_in_schema=False)
        async def _ready() -> dict[str, str]:
            return {"status": "ready", "module": id}

        @self.exception_handler(Exception)
        async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
            logging.getLogger("nkz.error").exception("unhandled exception: %s", exc)
            return JSONResponse(
                status_code=500, content={"error": "internal", "detail": str(exc)}
            )

    def auth(self, roles: Sequence[str] | None = None) -> Any:
        """Shortcut for `require_auth(roles)`."""
        return require_auth(roles)

    def orion(self, ctx: AuthContext) -> OrionClient:
        """Create an OrionClient scoped to the authenticated tenant."""
        return OrionClient(ctx.tenant_id)

    def timescale(self, ctx: AuthContext) -> TimescaleClient:
        """Create a TimescaleClient scoped to the authenticated tenant."""
        return TimescaleClient(ctx.tenant_id)
