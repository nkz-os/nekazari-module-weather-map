"""
Authentication dependency for module backends.

The api-gateway validates the JWT and injects headers:
  X-Tenant-ID, X-User-ID, X-User-Roles, X-Request-ID

This module provides require_auth() which reads those headers.
Module backends do NOT validate JWT signatures — the gateway did that.
Defense in depth: we still validate header presence and format.
"""

import re
from dataclasses import dataclass
from typing import Sequence
from fastapi import Request, HTTPException, Depends


@dataclass(frozen=True)
class AuthContext:
    """Authenticated request context injected by the gateway."""

    tenant_id: str
    user_id: str
    roles: tuple[str, ...]
    request_id: str | None = None

    def has_role(self, role: str) -> bool:
        return role in self.roles

    def has_any_role(self, roles: Sequence[str]) -> bool:
        return any(r in self.roles for r in roles)


def require_auth(roles: Sequence[str] | None = None):
    """
    FastAPI dependency that validates gateway-injected auth headers.

    Args:
        roles: If provided, at least one role must match.

    Returns:
        FastAPI dependency yielding AuthContext.

    Raises:
        HTTPException 401 if headers are missing or invalid.
        HTTPException 403 if roles are provided and none match.
    """

    async def _require_auth(request: Request) -> AuthContext:
        tenant_id = request.headers.get("X-Tenant-ID", "").strip()
        user_id = request.headers.get("X-User-ID", "").strip()
        roles_header = request.headers.get("X-User-Roles", "").strip()
        request_id = request.headers.get("X-Request-ID", "").strip() or None

        # Validate presence
        if not tenant_id:
            raise HTTPException(
                status_code=401,
                detail="Missing X-Tenant-ID header — gateway misconfiguration?",
            )
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Missing X-User-ID header — gateway misconfiguration?",
            )

        # Validate tenant_id format (alphanumeric + underscore + hyphen, 3-63 chars)
        if not re.match(r"^[a-z0-9_-]{3,63}$", tenant_id):
            raise HTTPException(
                status_code=401,
                detail=f"Invalid X-Tenant-ID format: {tenant_id}",
            )

        user_roles = tuple(r.strip() for r in roles_header.split(",") if r.strip())

        # Role check (defense in depth — gateway also checks)
        if roles is not None:
            allowed = set(roles)
            if not any(r in allowed for r in user_roles):
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied. Required one of: {roles}",
                )

        return AuthContext(
            tenant_id=tenant_id,
            user_id=user_id,
            roles=user_roles,
            request_id=request_id,
        )

    return Depends(_require_auth)
