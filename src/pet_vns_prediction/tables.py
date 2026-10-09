"""Data-only descriptive summaries ported from the original cohort script.

No P values, feature selection, real cohort rows, or manuscript text templates.
"""
from __future__ import annotations
import numpy as np


def cohort_descriptives(rows, y, *, continuous, categorical):
    """Summarize caller-owned aligned rows in all/responder/nonresponder groups.

    Missing or nonfinite requested fields fail explicitly, as in the original
    complete-case source. Category levels are shared across groups. SD uses
    ddof=1 and is None for a one-row group; quantiles use NumPy's linear method.
    Frequency reconstruction, if needed, must be explicit before calling.
    """
    y = np.asarray(y)
    if y.ndim != 1 or len(rows) != len(y) or not len(rows) or not np.isin(y, [0, 1]).all():
        raise ValueError("nonempty aligned rows and binary outcomes required")
    if not all(np.any(y == group) for group in (0, 1)):
        raise ValueError("both outcome groups are required")
    if len(set(continuous) | set(categorical)) != len(continuous) + len(categorical):
        raise ValueError("field names must be unique across continuous and categorical fields")
    for row in rows:
        for field in [*continuous, *categorical]:
            if field not in row or row[field] is None:
                raise ValueError(f"missing value: {field}")
        for field in categorical:
            if isinstance(row[field], (float, np.floating)) and not np.isfinite(row[field]):
                raise ValueError(f"nonfinite category: {field}")
    arrays = {field: np.asarray([r[field] for r in rows], dtype=float) for field in continuous}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("continuous fields must be finite")
    levels = {field: sorted(set(r[field] for r in rows), key=str) for field in categorical}
    summary = {}
    for group, mask in {"all": np.ones(len(rows), dtype=bool), "responder": y == 1, "nonresponder": y == 0}.items():
        selected = [r for r, included in zip(rows, mask) if included]
        item = {"n": len(selected), "continuous": {}, "categorical": {}}
        for field, array in arrays.items():
            v = array[mask]
            item["continuous"][field] = {"n": len(v), "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "median": float(np.median(v)),
                "q1": float(np.quantile(v, .25)), "q3": float(np.quantile(v, .75)),
                "min": float(v.min()), "max": float(v.max())}
        for field in categorical:
            item["categorical"][field] = {level: {"count": sum(r[field] == level for r in selected),
                "percent": 100 * sum(r[field] == level for r in selected) / len(selected)} for level in levels[field]}
        summary[group] = item
    return {"n": len(rows), "responders": int(sum(y == 1)), "nonresponders": int(sum(y == 0)),
            "groups": summary, "inference": "Descriptive only; no P values or feature selection."}
