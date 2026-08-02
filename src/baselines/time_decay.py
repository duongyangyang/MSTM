"""
Baseline 2 — Time-Decay Forgetting.

Removes memory records based on age past a configurable threshold.
Older records are dropped; newer records are retained.
See TODOLIST.md Phase 3, Task 8.

Assumes memory records are formatted as bullet points (one per line)
and each record may optionally carry a timestamp annotation like
"(since 2024)", "(until 2023)", "(Mar 2024)", etc.
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional


# Simple regex patterns for extracting dates from text
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_DATE_PATTERN = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+(\d{4})\b"
)
_RELATIVE_PATTERN = re.compile(
    r"\b(\d+)\s+(day|week|month|year)s?\s+(ago|old)\b"
)
# "recently", "last month", "this year", "currently", "now"
_RECENT_PATTERN = re.compile(
    r"\b(recently|currently|now|this\s+(year|month|week)|"
    r"last\s+(month|week)|past\s+(\d+)\s+(day|week|month|year)s?)\b",
    re.IGNORECASE,
)


def _estimate_age_years(record: str) -> Optional[float]:
    """
    Estimate the age of a memory record in years.
    Returns None if no date information is found (assumed recent).
    Returns a higher number for older records.
    """
    record_lower = record.lower()

    # If the record contains "recently", "currently", "now", etc., it's recent
    if _RECENT_PATTERN.search(record_lower):
        return 0.0

    # Look for explicit year mentions
    years = _YEAR_PATTERN.findall(record)
    if years:
        current_year = datetime.now().year
        ages = [current_year - int(y) for y in years]
        # Return the most recent year mentioned
        return min(ages)

    # Look for month+year patterns
    date_matches = _DATE_PATTERN.findall(record)
    if date_matches:
        month_str, year_str = date_matches[0]
        try:
            month_num = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                "may": 5, "jun": 6, "jul": 7, "aug": 8,
                "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            m = month_num.get(month_str.lower()[:3], 1)
            record_date = datetime(int(year_str), m, 1)
            now = datetime.now()
            return (now - record_date).days / 365.25
        except (ValueError, KeyError):
            pass

    # Look for relative time expressions
    rel_matches = _RELATIVE_PATTERN.findall(record_lower)
    if rel_matches:
        num, unit, _ = rel_matches[0]
        num = int(num)
        if unit.startswith("year"):
            return num
        elif unit.startswith("month"):
            return num / 12
        elif unit.startswith("week"):
            return num / 52
        elif unit.startswith("day"):
            return num / 365

    # No date information found — assume recent
    return 0.0


def transition(
    M: str,
    delta_M: str,
    age_threshold: float = 2.0,
    **kwargs,
) -> str:
    """
    Age-based memory pruning: forget records older than the threshold,
    then append new information.

    Args:
        M: Current memory state as a string (records separated by newlines).
        delta_M: New information to incorporate.
        age_threshold: Maximum age in years before a record is forgotten.
                       Default 2.0 years.
        **kwargs: Additional kwargs (ignored).

    Returns:
        M_prime: The evolved memory state with old records removed.
    """
    records = [r.strip() for r in M.split("\n") if r.strip()]

    # Filter out records older than the threshold
    kept_records = []
    for record in records:
        # Skip records that are clearly header/abstraction records
        # (these don't have individual timestamps — keep them)
        stripped = record.lstrip("- *•")
        age = _estimate_age_years(stripped)
        if age is None or age <= age_threshold:
            kept_records.append(record)

    # Build new memory state
    kept_memory = "\n".join(kept_records)

    # Append new information
    if delta_M.strip():
        if kept_memory:
            kept_memory += "\n" + delta_M.strip()
        else:
            kept_memory = delta_M.strip()

    return kept_memory


def transition_batch(
    memories: List[str],
    deltas: List[str],
    age_threshold: float = 2.0,
    **kwargs,
) -> List[str]:
    """Batch version of transition."""
    assert len(memories) == len(deltas)
    return [
        transition(m, d, age_threshold=age_threshold)
        for m, d in zip(memories, deltas)
    ]