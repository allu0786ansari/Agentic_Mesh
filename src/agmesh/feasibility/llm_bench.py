"""Week 1 feasibility spike: local LLM latency, JSON validity and memory (Gate G0).

The benchmark sends representative attribution prompts — rendered evidence text plus
five candidate MITRE ATT&CK for ICS techniques — to each local Ollama model and
asks for schema-constrained JSON, exactly as the Attribution Agent will in Week 10.

It measures what Gate G0 needs to decide:

* latency per call (p50, p95) → the L_LLM target for ``full`` mode (Proposal §8.6);
* decode and prompt tokens per second;
* JSON validity, schema validity and grounding (ID among the candidates);
* resident memory of the Ollama runner while the model is loaded.

The technique list below is a small fixed sample for benchmarking only. The real
knowledge base is built from the pinned STIX bundle in Week 9. "Indicative
agreement" with the scenario's intended technique is reported for interest only;
it is not an accuracy result.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from agmesh.common.artifacts import write_artifact
from agmesh.common.config import DEFAULT_PARAMS_PATH, load_params
from agmesh.common.memory import MemorySampler, processes_matching
from agmesh.security.egress import EgressDeniedError, require_allowlisted

PROVISIONAL_L_LLM_S = 20.0


@dataclass(frozen=True)
class Technique:
    technique_id: str
    name: str
    tactic: str


TECHNIQUES: tuple[Technique, ...] = (
    Technique("T0801", "Monitor Process State", "Collection"),
    Technique("T0806", "Brute Force I/O", "Impair Process Control"),
    Technique("T0813", "Denial of Control", "Impact"),
    Technique("T0814", "Denial of Service", "Inhibit Response Function"),
    Technique("T0815", "Denial of View", "Impact"),
    Technique("T0827", "Loss of Control", "Impact"),
    Technique("T0830", "Adversary-in-the-Middle", "Collection"),
    Technique("T0831", "Manipulation of Control", "Impact"),
    Technique("T0832", "Manipulation of View", "Impact"),
    Technique("T0836", "Modify Parameter", "Impair Process Control"),
    Technique("T0842", "Network Sniffing", "Discovery"),
    Technique("T0846", "Remote System Discovery", "Discovery"),
    Technique("T0855", "Unauthorized Command Message", "Impair Process Control"),
    Technique("T0856", "Spoof Reporting Message", "Evasion"),
    Technique("T0861", "Point & Tag Identification", "Collection"),
    Technique("T0888", "Remote System Information Discovery", "Discovery"),
)
_BY_ID = {t.technique_id: t for t in TECHNIQUES}


@dataclass(frozen=True)
class Scenario:
    site: str
    evidence: str
    intended: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "site_b",
        "node b1 (IED1A): write_ops >100; read_ops 1-10; request_rate high; "
        "exception_responses 0; new_peer true",
        "T0855",
    ),
    Scenario(
        "site_b",
        "node b3 (SCADA HMI): requests to many unit identifiers; read_ops 11-100; "
        "exception_responses 11-100; request_rate high; new_peer true",
        "T0846",
    ),
    Scenario(
        "site_b",
        "nodes b1, b2 (IED1A, IED1B): read_ops >100; request_rate high; "
        "exception_responses >100; response latency up high",
        "T0814",
    ),
    Scenario(
        "site_b",
        "node b2 (IED1B): write_ops 11-100 to holding registers used as setpoints; "
        "request_rate normal; new_peer false",
        "T0836",
    ),
    Scenario(
        "site_b",
        "node b3 (SCADA HMI): reported register values constant while write_ops 1-10 "
        "continue; diagnostic_ops 0",
        "T0856",
    ),
    Scenario(
        "site_a",
        "node a2: top deviating features P1_B2016 (setpoint) up high, P1_LIT01 (level) "
        "down medium, P1_FCV01D (valve demand) up medium",
        "T0836",
    ),
    Scenario(
        "site_a",
        "node a1: top deviating features P1_PP01AD (pump command) up high, "
        "P1_FT01 (flow) down low; no setpoint change",
        "T0831",
    ),
    Scenario(
        "site_a",
        "node a4: top deviating features P2_SIT01 (speed sensor) variance near zero "
        "while P2_SD01 (speed demand) up medium",
        "T0832",
    ),
)

SYSTEM_PROMPT = (
    "You are an industrial control system incident attribution assistant. "
    "Choose the single MITRE ATT&CK for ICS technique from the candidate list that "
    "best explains the evidence. Respond with JSON only."
)


@dataclass(frozen=True)
class Prompt:
    index: int
    scenario: Scenario
    candidates: tuple[str, ...]


def build_prompts(n: int, seed: int = 7, n_candidates: int = 5) -> list[Prompt]:
    """Deterministic prompts: each scenario's intended technique plus random distractors."""
    rng = random.Random(seed)  # noqa: S311 - reproducible benchmark prompts, not security
    prompts: list[Prompt] = []
    for i in range(n):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        distractors = [t.technique_id for t in TECHNIQUES if t.technique_id != scenario.intended]
        candidates = [scenario.intended, *rng.sample(distractors, n_candidates - 1)]
        rng.shuffle(candidates)
        prompts.append(Prompt(i, scenario, tuple(candidates)))
    return prompts


def response_schema(candidates: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "technique_id": {"type": "string", "enum": list(candidates)},
            "tactic": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
        },
        "required": ["technique_id", "tactic", "confidence", "rationale"],
    }


def render_user_message(prompt: Prompt) -> str:
    lines = [f"Evidence ({prompt.scenario.site}):", prompt.scenario.evidence, "", "Candidates:"]
    for technique_id in prompt.candidates:
        t = _BY_ID[technique_id]
        lines.append(f"- {t.technique_id} {t.name} (tactic: {t.tactic})")
    lines += [
        "",
        "Return technique_id (one of the candidate IDs), tactic, confidence between 0 and 1, "
        "and a rationale of at most 40 words.",
    ]
    return "\n".join(lines)


@dataclass
class CallResult:
    index: int
    ok: bool
    latency_s: float
    json_valid: bool = False
    schema_valid: bool = False
    grounded: bool = False
    agrees_with_intended: bool = False
    confidence: float | None = None
    eval_count: int | None = None
    eval_duration_s: float | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_s: float | None = None
    load_duration_s: float | None = None
    error: str | None = None


def validate_output(content: str, candidates: tuple[str, ...]) -> dict[str, Any]:
    """Check the model's JSON the way the Attribution Agent will (Proposal §8.4)."""
    result: dict[str, Any] = {
        "json_valid": False,
        "schema_valid": False,
        "grounded": False,
        "technique_id": None,
        "confidence": None,
    }
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(data, dict):
        return result
    result["json_valid"] = True
    technique_id = data.get("technique_id")
    confidence = data.get("confidence")
    schema_ok = (
        isinstance(technique_id, str)
        and isinstance(data.get("tactic"), str)
        and isinstance(data.get("rationale"), str)
        and isinstance(confidence, int | float)
        and not isinstance(confidence, bool)
        and 0.0 <= float(confidence) <= 1.0
    )
    result["schema_valid"] = schema_ok
    result["technique_id"] = technique_id if isinstance(technique_id, str) else None
    if isinstance(confidence, int | float) and not isinstance(confidence, bool):
        result["confidence"] = float(confidence)
    result["grounded"] = isinstance(technique_id, str) and technique_id in candidates
    return result


class OllamaClient:
    """Minimal synchronous client for the endpoints the benchmark needs."""

    def __init__(self, base_url: str, timeout_s: float, http: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=timeout_s)

    def version(self) -> str | None:
        try:
            response = self.http.get(f"{self.base_url}/api/version")
            response.raise_for_status()
            return str(response.json().get("version"))
        except (httpx.HTTPError, ValueError):
            return None

    def installed_models(self) -> list[str]:
        response = self.http.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        return [str(m.get("name")) for m in response.json().get("models", [])]

    def loaded_models(self) -> list[dict[str, Any]]:
        try:
            response = self.http.get(f"{self.base_url}/api/ps")
            response.raise_for_status()
            models: list[dict[str, Any]] = response.json().get("models", [])
            return models
        except (httpx.HTTPError, ValueError):
            return []

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.http.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "format": schema,
                "stream": False,
                "options": options,
                "keep_alive": "30m",
            },
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body


def is_installed(model: str, installed: list[str]) -> bool:
    if model in installed:
        return True
    return ":" not in model and f"{model}:latest" in installed


def _ns_to_s(value: Any) -> float | None:
    return round(float(value) / 1e9, 4) if isinstance(value, int | float) else None


def run_call(
    client: OllamaClient, model: str, prompt: Prompt, options: dict[str, Any]
) -> CallResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_message(prompt)},
    ]
    start = time.perf_counter()
    try:
        body = client.chat(model, messages, response_schema(prompt.candidates), options)
    except httpx.HTTPError as exc:
        return CallResult(
            prompt.index, ok=False, latency_s=time.perf_counter() - start, error=str(exc)
        )
    latency = time.perf_counter() - start
    content = str(body.get("message", {}).get("content", ""))
    checked = validate_output(content, prompt.candidates)
    return CallResult(
        index=prompt.index,
        ok=True,
        latency_s=round(latency, 4),
        json_valid=checked["json_valid"],
        schema_valid=checked["schema_valid"],
        grounded=checked["grounded"],
        agrees_with_intended=checked["technique_id"] == prompt.scenario.intended,
        confidence=checked["confidence"],
        eval_count=body.get("eval_count"),
        eval_duration_s=_ns_to_s(body.get("eval_duration")),
        prompt_eval_count=body.get("prompt_eval_count"),
        prompt_eval_duration_s=_ns_to_s(body.get("prompt_eval_duration")),
        load_duration_s=_ns_to_s(body.get("load_duration")),
    )


def _rate(numerators: list[int | None], durations: list[float | None]) -> float | None:
    pairs = [(n, d) for n, d in zip(numerators, durations, strict=True) if n and d]
    if not pairs:
        return None
    return round(sum(n for n, _ in pairs) / sum(d for _, d in pairs), 2)


def summarise(results: list[CallResult], min_confidence: float) -> dict[str, Any]:
    done = [r for r in results if r.ok]
    if not done:
        return {"n_calls": len(results), "n_ok": 0}
    latencies = np.array([r.latency_s for r in done])
    p95 = float(np.percentile(latencies, 95))
    n = len(done)
    return {
        "n_calls": len(results),
        "n_ok": n,
        "latency_s": {
            "p50": round(float(np.percentile(latencies, 50)), 3),
            "p95": round(p95, 3),
            "mean": round(float(latencies.mean()), 3),
            "max": round(float(latencies.max()), 3),
        },
        "decode_tokens_per_s": _rate(
            [r.eval_count for r in done], [r.eval_duration_s for r in done]
        ),
        "prompt_tokens_per_s": _rate(
            [r.prompt_eval_count for r in done], [r.prompt_eval_duration_s for r in done]
        ),
        "mean_output_tokens": round(float(np.mean([r.eval_count or 0 for r in done])), 1),
        "json_valid_rate": round(sum(r.json_valid for r in done) / n, 3),
        "schema_valid_rate": round(sum(r.schema_valid for r in done) / n, 3),
        "grounded_rate": round(sum(r.grounded for r in done) / n, 3),
        "above_min_confidence_rate": round(
            sum((r.confidence or 0.0) >= min_confidence for r in done) / n, 3
        ),
        "indicative_agreement_rate": round(sum(r.agrees_with_intended for r in done) / n, 3),
        "suggested_l_llm_s": max(5, math.ceil(p95 * 1.5)),
        "meets_provisional_20s": p95 <= PROVISIONAL_L_LLM_S,
    }


def benchmark_model(
    client: OllamaClient,
    model: str,
    prompts: list[Prompt],
    options: dict[str, Any],
    min_confidence: float,
    sample_memory: bool = True,
) -> dict[str, Any]:
    # Warm-up loads the model so that load time is reported separately from steady state.
    warmup = run_call(client, model, prompts[0], options)
    sampler = MemorySampler(processes_matching("ollama"), interval_s=0.5) if sample_memory else None
    if sampler:
        sampler.start()
    try:
        results = []
        for prompt in prompts:
            result = run_call(client, model, prompt, options)
            results.append(result)
            status = "ok" if result.ok else f"error: {result.error}"
            position = f"{prompt.index + 1}/{len(prompts)}"
            print(f"  [{model}] call {position} {result.latency_s:.2f}s {status}")
    finally:
        if sampler:
            sampler.stop()
    loaded = [
        m
        for m in client.loaded_models()
        if is_installed(model, [str(m.get("name")), str(m.get("model"))])
    ]
    reported = loaded[0] if loaded else {}
    return {
        "model": model,
        "status": "benchmarked",
        "warmup": {
            "ok": warmup.ok,
            "latency_s": round(warmup.latency_s, 3),
            "load_duration_s": warmup.load_duration_s,
            "error": warmup.error,
        },
        "summary": summarise(results, min_confidence),
        "memory": {
            "ollama_processes": sampler.summary() if sampler else None,
            "ollama_reported_size_mb": round(reported["size"] / 2**20)
            if reported.get("size")
            else None,
            "ollama_reported_vram_mb": round(reported.get("size_vram", 0) / 2**20)
            if reported
            else None,
        },
        "calls": [asdict(r) for r in results],
    }


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--base-url", default=None, help="Ollama URL (default: params llm.base_url)"
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="default: llm.model and llm.alternative_model"
    )
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--num-predict", type=int, default=256, help="output token cap per call")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/feasibility/llm_bench.json"))
    parser.add_argument(
        "--no-memory", action="store_true", help="skip Ollama process memory sampling"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, http: httpx.Client | None = None) -> int:
    args = parse_args(argv)
    params = load_params(args.params)
    base_url = args.base_url or params.llm.base_url
    try:
        require_allowlisted(base_url, params.security.egress_allowlist)
    except EgressDeniedError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    models = args.models or [params.llm.model, params.llm.alternative_model]
    client = OllamaClient(base_url, args.timeout_s, http=http)
    version = client.version()
    try:
        installed = client.installed_models()
    except httpx.HTTPError as exc:
        print(f"ERROR: cannot reach Ollama at {base_url}: {exc}", file=sys.stderr)
        print(
            "Start it with `ollama serve` (or check that the service is running).", file=sys.stderr
        )
        return 1

    prompts = build_prompts(
        args.n_prompts, seed=args.seed, n_candidates=params.mesh.attribution.top_k
    )
    options = {
        "temperature": params.llm.temperature,
        "num_ctx": params.llm.num_ctx,
        "num_predict": args.num_predict,
        "seed": 0,
    }
    reports: list[dict[str, Any]] = []
    for model in models:
        if not is_installed(model, installed):
            print(f"SKIP {model}: not installed. Run `ollama pull {model}` and re-run.")
            reports.append({"model": model, "status": "not_installed"})
            continue
        print(f"Benchmarking {model} with {len(prompts)} prompts ...")
        reports.append(
            benchmark_model(
                client,
                model,
                prompts,
                options,
                params.mesh.attribution.min_confidence,
                not args.no_memory,
            )
        )

    benchmarked = [r for r in reports if r["status"] == "benchmarked"]
    path = write_artifact(
        args.output,
        kind="feasibility.llm_bench",
        data_source="runtime",
        payload={
            "ollama_version": version,
            "base_url": base_url,
            "options": options,
            "n_prompts": len(prompts),
            "provisional_l_llm_s": PROVISIONAL_L_LLM_S,
            "models": reports,
        },
        params_path=args.params,
    )
    print(f"\nWrote {path}")
    for report in benchmarked:
        s = report["summary"]
        if not s.get("n_ok"):
            print(f"  {report['model']}: no successful calls")
            continue
        print(
            f"  {report['model']}: p50 {s['latency_s']['p50']}s, p95 {s['latency_s']['p95']}s, "
            f"{s['decode_tokens_per_s']} tok/s, schema-valid {s['schema_valid_rate']:.0%}, "
            f"grounded {s['grounded_rate']:.0%}, suggested L_LLM {s['suggested_l_llm_s']}s"
        )
    return 0 if benchmarked else 1


if __name__ == "__main__":
    raise SystemExit(main())
