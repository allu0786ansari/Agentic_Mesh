"""Local secrets from ``.env`` (see ``.env.example``).

``.env`` holds machine-local secrets:

* ``HF_TOKEN``: authenticated Hugging Face Hub requests for the encoder;
* ``MESH_APPROVAL_KEY``: the HMAC key of the approval gate.

The file is git-ignored. :func:`load_env` reads it at the command-line entry points.

* **It never overrides a variable that is already set,** so a value exported in the
  shell, or set by CI, wins.
* **Values are never printed or written to an artifact.** Provenance records no
  environment variables.

The parser is deliberately small, so it needs no extra dependency. It accepts:

* ``KEY=value`` lines;
* an optional ``export`` prefix;
* single- or double-quoted values;
* blank lines and ``#`` comments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        out[key] = value
    return out


def load_env(path: Path | str = ".env") -> list[str]:
    """Set the variables in ``path`` that are not already set; returns their names."""
    p = Path(path)
    if not p.is_file():
        return []
    loaded = []
    for key, value in parse(p.read_text(encoding="utf-8")).items():
        if value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded
