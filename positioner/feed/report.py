"""Builds a summary ValidationReport from a list of ValidationResult objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from positioner.feed.validator import ValidationResult


@dataclass
class ValidationReport:
    """Summary of a full-feed validation run."""
    total: int
    valid_count: int
    invalid_count: int
    invalid_rows: List[ValidationResult] = field(default_factory=list)


def build_report(results: List[ValidationResult]) -> ValidationReport:
    """Aggregate a list of ValidationResult objects into a ValidationReport.

    Invariant guaranteed: valid_count + invalid_count == total == len(results).

    Args:
        results: list of ValidationResult produced by validate_row().

    Returns:
        ValidationReport summarising the validation run.
    """
    invalid_rows = [r for r in results if not r.valid]
    valid_count = len(results) - len(invalid_rows)
    return ValidationReport(
        total=len(results),
        valid_count=valid_count,
        invalid_count=len(invalid_rows),
        invalid_rows=invalid_rows,
    )
