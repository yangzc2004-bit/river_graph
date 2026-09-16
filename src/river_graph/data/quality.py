"""Pre-registered covariate quality rules (R1).

A value is rejected here for exactly one kind of reason: it cannot be a real
measurement of that variable, given its unit and its physical range.  The
rules never use model scores, model residuals, or outer-test labels, so a
value that a model merely finds hard to predict is never removed.  That
distinction is the whole point: DOC concentrations above 400 mg/L stay in the
data because they are real measurements, while a water temperature of
1310 deg C does not stay because it cannot be one.

Every rejected row keeps its raw value and a reason code so the decision can
be traced back to the provider record.  Rejection means "missing", never
"repaired": we do not guess a decimal point or convert a unit we cannot
verify.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RULE_VERSION = "covariate_quality_v1"

# Repository root, used by callers that record where a rule set was applied.
REPO_ROOT = Path(__file__).resolve().parents[3]

# Units are compared after strip().lower().  None means "any unit is accepted"
# (used only where the reader already normalises the unit, e.g. NWIS cfs).
RULES: dict[str, dict] = {
    "temperature": {
        "units": ("deg c",),
        "min": -5.0,
        "max": 40.0,
        "rationale": (
            "Water temperature is accepted only when the provider states "
            "deg C and the value lies in [-5, 40] deg C. -5 C covers "
            "near-freezing under-ice readings; 40 C is above the highest "
            "reliably reported stream temperature in this basin and far below "
            "the mislabelled values found in the cache (up to 1310 deg C). "
            "Rows reported in deg F are rejected rather than converted: the "
            "deg F group contains values that are impossible even after "
            "conversion (for example 25.5 deg F in July, 4 deg F in October), "
            "so the unit label itself is not trustworthy for those rows."
        ),
    },
    "ph": {
        "units": ("standard units",),
        "min": 0.0,
        "max": 14.0,
        "rationale": "pH is dimensionless and cannot leave [0, 14].",
    },
    "spec_conductance": {
        "units": ("us/cm",),
        "min": 0.0,
        "max": 100000.0,
        "rationale": (
            "Specific conductance is non-negative; 100000 uS/cm is far above "
            "any freshwater value in this basin and only catches unit errors."
        ),
    },
    "discharge": {
        "units": None,
        "min": None,
        "max": 3.0e6,
        "signed": True,
        "rationale": (
            "Discharge is accepted as a signed quantity: negative daily values "
            "come from tidal and backwater reaches where the flow reverses and "
            "are real measurements. The bound only catches magnitude or unit "
            "errors (the largest observed value is about 1.4e6 cfs)."
        ),
    },
    "doc": {
        "units": ("mg/l",),
        "min": None,
        "max": None,
        "rationale": (
            "DOC is the prediction target, not a covariate. Only the "
            "already-frozen unit/detection filters apply; no value-range rule "
            "is added, so high concentrations are never removed for being "
            "hard to predict."
        ),
    },
}

REASONS = (
    "missing_value",
    "missing_unit",
    "unsupported_unit",
    "below_physical_min",
    "above_physical_max",
)


def normalize_unit(unit) -> str:
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return ""
    return str(unit).strip().lower()


def rules_digest() -> str:
    """Stable digest of the rule set, recorded with every build."""
    payload = {
        "version": RULE_VERSION,
        "rules": {
            name: {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in rule.items()
            }
            for name, rule in sorted(RULES.items())
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def apply_rules(frame: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Annotate a covariate frame with qc_status / qc_reason; never drops rows.

    The input frame must expose 'value' and (optionally) 'unit'.  Rows are
    classified in a fixed order so the reason is deterministic.
    """
    if variable not in RULES:
        raise KeyError("no quality rule for variable " + str(variable))
    rule = RULES[variable]
    out = frame.copy()
    if "unit" not in out.columns:
        out["unit"] = None
    value = pd.to_numeric(out["value"], errors="coerce")
    unit = out["unit"].map(normalize_unit)
    reason = pd.Series("", index=out.index, dtype=object)

    def mark(mask: pd.Series, code: str) -> None:
        todo = mask & (reason == "")
        reason.loc[todo] = code

    mark(value.isna(), "missing_value")
    allowed = rule.get("units")
    if allowed is not None:
        mark(unit == "", "missing_unit")
        mark(~unit.isin(list(allowed)), "unsupported_unit")
    # A signed variable is bounded on its MAGNITUDE: negative discharge is a
    # real reversing flow in tidal and backwater reaches, but -4e6 cfs is not
    # a measurement either. Without this the rule accepted any negative value
    # while the documentation claimed an absolute bound.
    checked = value.abs() if rule.get("signed") else value
    if rule.get("min") is not None:
        mark(checked < rule["min"], "below_physical_min")
    if rule.get("max") is not None:
        mark(checked > rule["max"], "above_physical_max")

    out["variable"] = variable
    out["qc_reason"] = reason
    out["qc_status"] = np.where(reason == "", "accepted", "rejected")
    out["qc_value"] = value
    out["qc_unit"] = unit
    return out


def summarize(audited: pd.DataFrame) -> dict:
    """Counts per status and reason for one or more audited variables."""
    if audited.empty:
        return {"rows": 0, "accepted": 0, "rejected": 0, "by_reason": {}}
    counts = audited[audited.qc_status == "rejected"].groupby(
        ["variable", "qc_reason"]
    ).size()
    return {
        "rows": len(audited),
        "accepted": int((audited.qc_status == "accepted").sum()),
        "rejected": int((audited.qc_status == "rejected").sum()),
        "by_reason": {
            str(variable) + ":" + str(reason): int(count)
            for (variable, reason), count in counts.items()
        },
    }


def rejection_sample(audited: pd.DataFrame, per_reason: int = 5) -> dict:
    """A few raw examples per (variable, reason) so the decision is traceable."""
    rejected = audited[audited.qc_status == "rejected"]
    sample: dict[str, list] = {}
    for (variable, reason), part in rejected.groupby(["variable", "qc_reason"]):
        key = str(variable) + ":" + str(reason)
        rows = []
        for row in part.head(per_reason).itertuples(index=False):
            rows.append(
                {
                    "site_no": str(getattr(row, "site_no", "")),
                    "date": str(getattr(row, "date", "")),
                    "value": None if pd.isna(row.value) else float(row.value),
                    "unit": None if row.unit is None or pd.isna(row.unit)
                    else str(row.unit),
                }
            )
        sample[key] = rows
    return sample


def write_report(path: str | Path, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temp.replace(path)
    return path
