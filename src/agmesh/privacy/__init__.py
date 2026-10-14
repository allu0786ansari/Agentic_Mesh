"""Differential privacy for the federated detectors (Proposal §7.3; Plan Week 5).

``calibrate``        RDP accounting with a wide order grid; noise calibration up front.
``accountant_store`` per-client, append-only accountant history on disk.
``budget_guard``     refuses any release that would exceed the client's total budget.
``noisy_stats``      DP histogram release for normalisation statistics.
``bounds``           public per-feature value ranges (no data involved).
"""
