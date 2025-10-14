import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable to store request ID
request_id_ctx_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
UNKNOWN_REQUEST_ID = "UNKNOWN_REQUEST_ID"


class RequestIdFilter(logging.Filter):
    """
    Logging filter that adds the request ID to log records.
    """

    def filter(self, record):
        record.request_id = request_id_ctx_var.get() or "-"
        return True


def setup_logging():
    """
    Configure logging with request ID tracking.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter with request ID
    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - (%(request_id)s) - (%(name)s) -  %(message)s"
    )
    handler.setFormatter(formatter)

    # Add request ID filter
    handler.addFilter(RequestIdFilter())

    # Add handler to logger
    logger.addHandler(handler)

    # optionally add file logging handler
    # if True:
    #     file_handler = logging.FileHandler("app.log")
    #     file_handler.setLevel(logging.INFO)
    #     file_handler.setFormatter(formatter)
    #     logger.addHandler(file_handler)

    return logger


def get_request_id() -> str:
    """
    Get the current request ID from context or return None.
    """
    return request_id_ctx_var.get() or UNKNOWN_REQUEST_ID


def set_request_id(request_id: str) -> None:
    """
    Set the request ID in the context.
    """
    request_id_ctx_var.set(request_id)


def generate_request_id() -> str:
    """
    Generate a new request ID.
    """
    return str(uuid.uuid4())
