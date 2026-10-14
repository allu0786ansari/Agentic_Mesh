"""Federated training and evaluation (Proposal §7; Plan Week 6).

``protocol``    the messages exchanged between server and clients (plain data only).
``strategies``  FedAvg and FedProx aggregation.
``fed_eval``    aggregation of client evaluations; pooled AUROC from binned histograms.
``registry``    the node registry (active / degraded) in SQLite.
``server``      the round loop. It imports no data loader and never reads a partition.
``client``      a client: loads only its own partition and owns its privacy budget.
``backends``    in-process orchestration (deterministic) and the Flower adapter.
``run``         config -> federation -> result artifact.
"""
