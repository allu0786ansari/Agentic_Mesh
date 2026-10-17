"""Memory readings for long LLM runs (ADR 0013 §6).

* :func:`available_mb` is ``MemAvailable`` from ``/proc/meminfo``.
* :func:`llm_server_rss_mb` is the summed resident memory of the local LLM server
  processes (``llama-server``, the runner that Ollama starts).

Both are Linux-only and return ``None`` where ``/proc`` cannot be read. They are only
observed and logged; nothing decides on them.
"""

from __future__ import annotations

from pathlib import Path

SERVER_NAMES = ("llama-server", "ollama_llama_se")  # /proc comm is cut at 15 characters


def _kb(text: str, key: str) -> int | None:
    for line in text.splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return None


def available_mb(proc: Path = Path("/proc")) -> int | None:
    try:
        kb = _kb((proc / "meminfo").read_text(), "MemAvailable")
    except OSError:
        return None
    return None if kb is None else kb // 1024


def llm_server_rss_mb(
    proc: Path = Path("/proc"), names: tuple[str, ...] = SERVER_NAMES
) -> int | None:
    total, found = 0, False
    try:
        pids = [p for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        return None
    for pid in pids:
        try:
            if (pid / "comm").read_text().strip() not in names:
                continue
            kb = _kb((pid / "status").read_text(), "VmRSS")
        except OSError:
            continue
        if kb is not None:
            total, found = total + kb, True
    return total // 1024 if found else None


def snapshot() -> dict[str, int | None]:
    return {"mem_available_mb": available_mb(), "llm_server_rss_mb": llm_server_rss_mb()}


def llm_server_info(
    proc: Path = Path("/proc"), names: tuple[str, ...] = SERVER_NAMES
) -> dict[str, object] | None:
    """The running LLM server's command line and prompt-cache setting (ADR 0013 §7).

    ``cache_ram`` comes from ``--cache-ram`` on the command line, else from
    ``LLAMA_ARG_CACHE_RAM`` in the server's environment. That environment is readable
    only when the server runs as the same user, as it does when started by hand. The
    value is ``None`` when neither is visible.
    """
    try:
        pids = sorted((p for p in proc.iterdir() if p.name.isdigit()), key=lambda p: p.name)
    except OSError:
        return None
    for pid in pids:
        try:
            if (pid / "comm").read_text().strip() not in names:
                continue
            argv = (pid / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        argv = [a for a in argv if a]
        cache_ram: str | None = None
        if "--cache-ram" in argv[:-1]:
            cache_ram = argv[argv.index("--cache-ram") + 1]
        env_readable = False
        try:
            env = (pid / "environ").read_bytes().decode(errors="replace").split("\0")
            env_readable = True
            for item in env:
                if item.startswith("LLAMA_ARG_CACHE_RAM=") and cache_ram is None:
                    cache_ram = item.split("=", 1)[1]
        except OSError:
            pass
        return {
            "pid": int(pid.name),
            "cmdline": argv,
            "cache_ram": cache_ram,
            "env_readable": env_readable,
        }
    return None
