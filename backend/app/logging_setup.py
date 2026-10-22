"""Logging for this module's own loggers.

Uvicorn's dictConfig configures only its own loggers ("uvicorn",
"uvicorn.error", "uvicorn.access") and never the root logger or ours. Left
alone, every "app.*" logger has no level and no handler: INFO records are
dropped silently while ERROR still reaches Python's last-resort handler, so
the module looks healthy right up until it logs a traceback. Configuring the
"app" namespace (not the root logger) fixes that without touching uvicorn's
loggers or its access log, and the same applies to queue workers (arq, RQ)
that configure only their own loggers.
"""

import logging

NAMESPACE = "app"


def configure_logging(level: str) -> None:
    """Give the "app" logger a level and one formatted stream handler.

    Guarded so repeated calls (one create_app() per test, one import per
    worker job) do not stack handlers and print every line twice.
    """
    module_logger = logging.getLogger(NAMESPACE)
    module_logger.setLevel(level.upper())
    if not module_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        module_logger.addHandler(handler)
