"""Reproducible analysis of public frontier AI capability, price and disclosure signals.

The package is organised in three layers that are deliberately kept separate:

``aicap.sources``
    Fetch and validate raw public data. Every source module declares a schema contract
    and fails loudly when upstream drifts, instead of silently emitting empty columns.

``aicap.analysis``
    Estimators. Each module answers one question, returns tidy tables, and either
    reports an uncertainty interval or records a refusal explaining why the available
    data cannot support the claim.

``aicap.report``
    Rendering. Contains no analysis. Every number it prints is read from an analysis
    table, so the prose cannot drift away from the computation.
"""

__version__ = "2.0.0"
