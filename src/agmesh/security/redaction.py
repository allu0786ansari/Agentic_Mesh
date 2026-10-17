"""Remove secrets from log records before any handler writes them (ADR 0014 §6).

:func:`redact` replaces the following with ``[REDACTED]``:

* ``Authorization`` header values and ``Bearer …`` tokens;
* JWT-shaped strings;
* PEM private-key blocks;
* Hugging Face tokens;
* the literal values of the secrets named in :data:`SECRET_ENV` when they are set.

:func:`install` makes every log record of the process redacted when it is created,
by rewriting ``record.msg`` and ``record.args``. Every handler, including uvicorn's
access log and httpx's request log, then sees only the redacted text.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

MARK = "[REDACTED]"
SECRET_ENV = ("MESH_APPROVAL_KEY", "HF_TOKEN")
PATTERNS = [
    re.compile(
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z]+ )?PRIVATE KEY-----",
        re.S,
    ),
    re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+(?:\s+[^'\"\s,}]+)?"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]+"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
]


def redact(text: str) -> str:
    out = text
    for pattern in PATTERNS:
        if pattern.groups:
            out = pattern.sub(lambda m: m.group(1) + MARK, out)
        else:
            out = pattern.sub(MARK, out)
    for name in SECRET_ENV:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            out = out.replace(value, MARK)
    return out


class RedactingFilter(logging.Filter):
    """For handlers outside this process's record factory (kept for explicit use)."""

    def filter(self, record: logging.LogRecord) -> bool:
        _scrub(record)
        return True


def _scrub(record: logging.LogRecord) -> None:
    # Arguments are redacted one by one first, so formatters that read record.args as a
    # tuple (uvicorn's access log) keep working; only a secret that appears after
    # formatting collapses the record to its redacted message.
    if isinstance(record.args, tuple):
        record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
    elif isinstance(record.args, dict):
        record.args = {k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()}
    if isinstance(record.msg, str) and not record.args:
        record.msg = redact(record.msg)  # a template with placeholders is left to the check below
    try:
        message = record.getMessage()
    except Exception:  # a malformed record is left to logging's own error handling
        return
    cleaned = redact(message)
    if cleaned != message:
        record.msg, record.args = cleaned, None


_installed = False


def install() -> None:
    """Redact every log record of this process at creation (idempotent).

    A record factory is used rather than a filter on some loggers: records that
    propagate to ancestor loggers bypass those loggers' filters, and handlers added
    later would miss a handler filter. The factory sees every record once.
    """
    global _installed
    if _installed:
        return
    base = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = base(*args, **kwargs)
        _scrub(record)
        return record

    logging.setLogRecordFactory(factory)
    _installed = True
