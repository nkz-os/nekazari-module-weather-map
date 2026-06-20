"""
Nekazari Platform SDK — Backend module development kit.

Provides:
- ModuleApp: FastAPI subclass pre-wired with CORS, /health, JSON logs, auth helpers
- require_auth: FastAPI dependency for authenticated routes
- OrionClient: Typed NGSI-LD async client with automatic tenant header injection
- SyncOrionClient: Synchronous NGSI-LD client (for sync codebases)
- TimescaleClient: Typed timeseries reader, tenant headers auto-injected
- ModuleLifecycle: Base class for install/uninstall/enable/disable hooks
- ModuleConfig: Per-tenant encrypted configuration storage

License: Apache-2.0 — modules using this SDK may use any license.
"""

from nkz_platform_sdk.auth import require_auth, AuthContext
from nkz_platform_sdk.module_app import ModuleApp
from nkz_platform_sdk.orion import OrionClient, SyncOrionClient
from nkz_platform_sdk.timescale import TimescaleClient
from nkz_platform_sdk.lifecycle import ModuleLifecycle, LifecycleResult
from nkz_platform_sdk.config import ModuleConfig
from nkz_platform_sdk.subscriptions import SubscriptionRegistrar
from nkz_platform_sdk.activation import ModuleActivation
from nkz_platform_sdk.ngsi_headers import inject_fiware_headers

__all__ = [
    "ModuleApp",
    "require_auth",
    "AuthContext",
    "OrionClient",
    "SyncOrionClient",
    "TimescaleClient",
    "ModuleLifecycle",
    "LifecycleResult",
    "ModuleConfig",
    "SubscriptionRegistrar",
    "ModuleActivation",
    "inject_fiware_headers",
]
