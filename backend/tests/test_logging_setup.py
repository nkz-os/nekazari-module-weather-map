"""Module loggers must emit INFO, once, under uvicorn's logging config."""

import io
import logging
import logging.config

from uvicorn.config import LOGGING_CONFIG

from app.logging_setup import NAMESPACE, configure_logging


def test_info_is_emitted_once_under_uvicorn_config():
    logging.config.dictConfig(LOGGING_CONFIG)
    module_logger = logging.getLogger(NAMESPACE)
    for handler in list(module_logger.handlers):
        module_logger.removeHandler(handler)

    configure_logging("INFO")
    configure_logging("INFO")  # second call must not add a second handler

    stream = io.StringIO()
    module_logger.handlers[0].setStream(stream)
    logging.getLogger(NAMESPACE + ".probe").info("probe-line")

    assert len(module_logger.handlers) == 1
    assert stream.getvalue().count("probe-line") == 1
    assert "INFO" in stream.getvalue()


def test_uvicorn_loggers_are_left_alone():
    logging.config.dictConfig(LOGGING_CONFIG)
    access = logging.getLogger("uvicorn.access")
    before = list(access.handlers)
    configure_logging("INFO")
    assert access.handlers == before
    assert access.propagate is False
