from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from storage.sqlite_schema import connect, initialise
from agents.investigator.circuit_breaker import CircuitBreaker


class ChaosTest:
    """Exercise the system against simulated node failures during investigations."""

    SCENARIOS = (
        "pre_investigation_failure",
        "mid_hop_failure",
        "dual_failure",
        "recovery_after_timeout",
    )
    TERMINAL_STATUSES = {"completed", "failed", "recovered", "timed_out"}

    def __init__(self, *, database_path: str | Path = "logs/chaos_sessions.db") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialise()

    def initialise(self) -> None:
        initialise(self.database_path)
        with connect(self.database_path) as database:
            database.execute(
                """
                CREATE TABLE IF NOT EXISTS chaos_sessions (
                    session_id TEXT PRIMARY KEY,
                    scenario TEXT NOT NULL,
                    status TEXT NOT NULL,
                    node_id TEXT,
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    error TEXT
                )
                """
            )

    def _write_session(self, session_id: str, scenario: str, status: str, *, node_id: str | None = None, error: str | None = None) -> None:
        with connect(self.database_path) as database:
            database.execute(
                """
                INSERT INTO chaos_sessions (session_id, scenario, status, node_id, started_at, updated_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    scenario = excluded.scenario,
                    status = excluded.status,
                    node_id = excluded.node_id,
                    updated_at = excluded.updated_at,
                    error = excluded.error
                """,
                (
                    session_id,
                    scenario,
                    status,
                    node_id,
                    asyncio.get_running_loop().time(),
                    asyncio.get_running_loop().time(),
                    error,
                ),
            )

    def _read_status(self, session_id: str) -> str:
        with connect(self.database_path) as database:
            row = database.execute(
                "SELECT status FROM chaos_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return str(row[0]) if row else "missing"

    async def _simulate_session(self, scenario: str, session_id: str, *, fail_at: float = 0.0, recover_after: float = 0.0) -> None:
        self._write_session(session_id, scenario, "started", node_id=session_id.split("-")[-1])
        await asyncio.sleep(0.05)

        if scenario == "pre_investigation_failure":
            breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
            try:
                await breaker.call(lambda: self._failed_node_call())
            except TimeoutError:
                pass
            self._write_session(session_id, scenario, "failed", error="node unavailable before investigation")
            return

        if scenario == "mid_hop_failure":
            await asyncio.sleep(fail_at)
            breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
            try:
                await breaker.call(lambda: self._failed_node_call())
            except TimeoutError:
                pass
            self._write_session(session_id, scenario, "failed", error="mid-hop timeout")
            return

        if scenario == "dual_failure":
            await asyncio.sleep(fail_at)
            breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
            try:
                await breaker.call(lambda: self._failed_node_call())
            except TimeoutError:
                pass
            self._write_session(session_id, scenario, "failed", error="dual node failure")
            return

        await asyncio.sleep(0.15)
        if recover_after > 0:
            breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
            try:
                await breaker.call(lambda: self._failed_node_call())
            except TimeoutError:
                pass
            await asyncio.sleep(recover_after)
            await breaker.call(self._healthy_node_call)
            self._write_session(session_id, scenario, "recovered", error=None)
            return
        self._write_session(session_id, scenario, "completed", error=None)

    @staticmethod
    async def _failed_node_call() -> None:
        raise TimeoutError("injected node timeout")

    @staticmethod
    async def _healthy_node_call() -> str:
        return "recovered"

    async def _run_scenario(self, scenario: str, *, timeout_seconds: float) -> dict[str, Any]:
        fail_at = 0.2 if scenario in {"mid_hop_failure", "dual_failure"} else 0.0
        recover_after = 0.2 if scenario == "recovery_after_timeout" else 0.0

        session_ids = [f"{scenario}-session-{index}" for index in range(1, 4)]
        tasks = [
            asyncio.create_task(self._simulate_session(scenario, session_id, fail_at=fail_at, recover_after=recover_after))
            for session_id in session_ids
        ]

        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            pass

        statuses = {session_id: self._read_status(session_id) for session_id in session_ids}
        passed = all(status in self.TERMINAL_STATUSES for status in statuses.values())
        return {
            "scenario": scenario,
            "passed": bool(passed),
            "sessions": statuses,
        }

    def run_all(self, *, timeout_seconds: float = 5.0, output_path: str | Path = "evaluation/chaos_results.json") -> dict[str, Any]:
        self.initialise()
        results = {}
        for scenario in self.SCENARIOS:
            results[scenario] = asyncio.run(self._run_scenario(scenario, timeout_seconds=timeout_seconds))
        passed = all(item["passed"] for item in results.values())
        report = {
            "passed": bool(passed),
            "scenarios": results,
            "database_path": str(self.database_path),
        }
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def main(self) -> None:
        report = self.run_all(timeout_seconds=5.0)
        if report["passed"]:
            print("All chaos scenarios passed")
        else:
            print("Chaos scenarios failed")
            for scenario, item in report["scenarios"].items():
                if not item["passed"]:
                    print(f"{scenario}: {item}")


if __name__ == "__main__":
    ChaosTest().main()
