"""Evaluation utilities for the Agentic Mesh project."""

__all__ = ["ChaosTest", "build_ragas_report", "generate_charts"]


def __getattr__(name: str):
	if name == "ChaosTest":
		from .chaos_test import ChaosTest

		return ChaosTest
	if name == "build_ragas_report":
		from .ragas_eval import build_ragas_report

		return build_ragas_report
	if name == "generate_charts":
		from .plot_metrics import generate_charts

		return generate_charts
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
