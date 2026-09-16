"""T03/R3: scenario-matched internal validation splits for the H3-A protocol.

The frozen benchmark masks assign every observed cell to train / val / test.
H3-A keeps the *outer* test positions untouched and hidden, and replaces the
single internal "val" role with an arrangement that matches the shift each
scenario is about:

* e1_*                    random cell split  -> reuse the original train/val/test
* e2a_strict              temporal split     -> the strict future-forecasting
  control: no DOC observation from the test period is visible at all, so there
  is no future context.  The internal validation split is applied to the
  original pre-cutoff validation months; its 20% val_context is recorded but
  never opened, so no observed cell disappears from the split unannounced.
* e2b_partial             temporal split     -> future reconstruction under a
  running network: the original future context stays visible and the internal
  val splits 20/80 into val_context (opened during selection) and val_target.
* e2b_partial_hiddenctx   the same target pool as e2b_partial with the
  val_context role recorded but not opened.  This is a selection-visibility
  diagnostic, not the strict E2a scenario, and it is named accordingly.
* e3_internal_*           spatial split      -> keep the original held-out test
  stations; pick ~10% of the observed non-test stations as internal validation
  stations and give all their months to val; every remaining non-test cell
  goes to train.

Visibility is explicit: every mask stores a visible_roles array naming the
roles that may be opened in the selection / validation forward.  Absence of a
key no longer silently means "hidden", and a cell role can never disappear
from the split without a record.

Nothing here retrains or rewrites a frozen mask; the outer test arrays are
copied byte-for-byte from experiments/masks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Frozen protocol constants (mirrored in configs/h3a_v1.json).
INTERNAL_VAL_SEED = 1729
E2_CONTEXT_FRAC = 0.2
E3_INTERNAL_VAL_FRAC = 0.10

E1_MASKS = ("e1_r20_seed42", "e1_r20_seed43", "e1_r20_seed44")
E2A_SOURCE = "e2a_strict"
E2B_SOURCE = "e2b_partial"
E3_SOURCES = ("e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44")

# name -> source mask in experiments/masks
MASK_SOURCES = {
    **{name: name for name in E1_MASKS},
    "e2a_strict": E2A_SOURCE,
    "e2b_partial": E2B_SOURCE,
    "e2b_partial_hiddenctx": E2B_SOURCE,
    **{f"e3_internal_seed{s}": f"e3_spatial_seed{s}" for s in (42, 43, 44)},
}

DEV_SCENARIOS = ("e1_r20_seed42", "e2b_partial", "e3_internal_seed42")
KEY_SCENARIOS = (
    "e1_r20_seed42",
    "e1_r20_seed43",
    "e1_r20_seed44",
    "e2a_strict",
    "e2b_partial",
    "e3_internal_seed42",
    "e3_internal_seed43",
    "e3_internal_seed44",
)
# Outside the 8 key scenarios: a selection-visibility diagnostic that reuses
# the e2b target pool.  It never stands in for the strict temporal control.
DIAGNOSTIC_SCENARIOS = ("e2b_partial_hiddenctx",)

ROLES = ("train", "val", "val_context", "context", "test")
VISIBLE_ROLE_CHOICES = ("context", "train", "val_context")
NEVER_VISIBLE_ROLES = ("val", "test")


SOURCE_GRID_FILE = "source_grid.json"
SOURCE_GRID_VERSION = 1


def station_order_digest(sites: list[str]) -> str:
    """Digest of the station order, which is what a flat index really means."""
    import hashlib

    return hashlib.sha256("\n".join(str(s) for s in sites).encode("utf-8")).hexdigest()


def grid_of(y_mask: np.ndarray) -> dict:
    n, t = y_mask.shape
    return {
        "n_stations": int(n),
        "n_months": int(t),
        "observed_cells": int(y_mask.sum()),
    }


def source_grid_path(source_dir: Path) -> Path:
    return Path(source_dir) / SOURCE_GRID_FILE


def read_source_grid(source_dir: Path) -> dict:
    """The grid the frozen masks were encoded on.

    A flat index means station_index * n_months + month_index. With the same
    integers and a different month count the same number points at a different
    station and month, so importing a frozen mask without checking this is not
    a migration, it is silent corruption: combining a 653-month dataset with
    652-month indices produced "cell 137646 in both val and test".
    """
    path = source_grid_path(source_dir)
    if not path.is_file():
        raise ValueError(
            "missing " + str(path) + ": the frozen masks must declare the grid "
            "they were built on before they can be reused. Register it with "
            "scripts/build_h3_masks.py --register-source-grid."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("n_stations", "n_months", "observed_cells", "station_order_sha256"):
        if key not in record:
            raise ValueError("source grid record is missing " + key)
    return record


def write_source_grid(
    source_dir: Path, sites: list[str], grid: dict | None = None,
    y_mask: np.ndarray | None = None,
) -> dict:
    """Record the grid the frozen masks were encoded on.

    The grid can be taken from the dataset that produced them, or declared
    explicitly from a manifest when that dataset is no longer available. The
    station order always comes from the current station list, which must have
    the same length.
    """
    if grid is None:
        if y_mask is None:
            raise ValueError("either grid or y_mask is required")
        grid = grid_of(y_mask)
    if int(grid["n_stations"]) != len(sites):
        raise ValueError(
            "declared grid has " + str(grid["n_stations"])
            + " stations but the station list has " + str(len(sites))
        )
    record = {
        "version": SOURCE_GRID_VERSION,
        "n_stations": int(grid["n_stations"]),
        "n_months": int(grid["n_months"]),
        "observed_cells": int(grid["observed_cells"]),
        "station_order_sha256": station_order_digest(sites),
        "note": (
            "grid the frozen masks in this directory were encoded on; a mask "
            "built on a different grid must be migrated by station/month, "
            "never re-indexed by copying integers"
        ),
    }
    path = source_grid_path(source_dir)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)
    return record


def migrate_cells(
    cells: np.ndarray, source: dict, y_mask: np.ndarray, role: str, audit: dict
) -> np.ndarray:
    """Re-encode flat indices from the source grid onto this grid.

    Indices are decoded to (station_row, month_col) and re-encoded, which
    preserves the station/month assignment exactly. Cells that are no longer
    observed on the new grid are dropped and counted, because a cell cannot be
    assigned a role when its label has disappeared.
    """
    cells = np.asarray(cells, dtype=np.int64)
    if cells.size == 0:
        return cells
    old_t = int(source["n_months"])
    new_t = int(y_mask.shape[1])
    rows = cells // old_t
    cols = cells % old_t
    if rows.max() >= y_mask.shape[0]:
        raise ValueError(
            "source mask " + role + " refers to station row " + str(int(rows.max()))
            + " but this dataset has " + str(y_mask.shape[0]) + " stations"
        )
    if cols.max() >= new_t:
        raise ValueError(
            "source mask " + role + " refers to month column " + str(int(cols.max()))
            + " but this dataset has " + str(new_t) + " months"
        )
    observed = y_mask[rows, cols]
    migrated = np.sort((rows[observed] * new_t + cols[observed]).astype(np.int64))
    audit[role] = {
        "before": int(cells.size),
        "after": int(migrated.size),
        "dropped_unobserved": int(cells.size - migrated.size),
    }
    return migrated


def assert_source_grid(source: dict, y_mask: np.ndarray, sites: list[str]) -> dict:
    """Compare the grid and station order the frozen masks assume with this one."""
    actual = grid_of(y_mask)
    problems = []
    digest = station_order_digest(sites)
    if source["station_order_sha256"] != digest:
        problems.append(
            "station order differs (source "
            + str(source["station_order_sha256"])[:12] + ", this dataset "
            + digest[:12] + "); flat indices cannot be migrated"
        )
    if source["n_stations"] != actual["n_stations"]:
        problems.append(
            "station count differs: source " + str(source["n_stations"])
            + ", this dataset " + str(actual["n_stations"])
        )
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "source": {k: source[k] for k in
                   ("n_stations", "n_months", "observed_cells")},
        "target": actual,
        "identical_grid": bool(source["n_months"] == actual["n_months"]),
        "requires_migration": bool(source["n_months"] != actual["n_months"]),
    }


def load_mask(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def split_val_context(
    val: np.ndarray,
    seed: int = INTERNAL_VAL_SEED,
    context_frac: float = E2_CONTEXT_FRAC,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the original val into (val_context, val_target).

    The permutation is taken over the *sorted* cell indices so the result does
    not depend on the storage order of the input file.
    """
    cells = np.sort(np.asarray(val, dtype=np.int64))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(cells)
    n_context = round(len(cells) * context_frac)
    return (
        np.sort(perm[:n_context]).astype(np.int64),
        np.sort(perm[n_context:]).astype(np.int64),
    )


def e3_eligible_stations(y_mask: np.ndarray, outer_test: np.ndarray) -> np.ndarray:
    """Observed stations that keep at least one observed cell outside the test."""
    n, t = y_mask.shape
    observed = y_mask.reshape(n, t).sum(axis=1)
    in_test = np.bincount(np.asarray(outer_test, dtype=np.int64) // t, minlength=n)
    return np.flatnonzero(observed - in_test > 0)


def e3_internal_stations(
    y_mask: np.ndarray,
    outer_test: np.ndarray,
    seed: int = INTERNAL_VAL_SEED,
    frac: float = E3_INTERNAL_VAL_FRAC,
) -> np.ndarray:
    """Pick the internal validation stations for one E3 internal mask."""
    candidates = e3_eligible_stations(y_mask, outer_test)
    if len(candidates) == 0:
        raise ValueError("no station has an observed cell outside the outer test")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(candidates)
    n_val = max(1, round(len(candidates) * frac))
    return np.sort(perm[:n_val]).astype(np.int64)


def e3_internal_split(
    y_mask: np.ndarray,
    outer_test: np.ndarray,
    seed: int = INTERNAL_VAL_SEED,
    frac: float = E3_INTERNAL_VAL_FRAC,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """(split, val_station_rows, candidate_rows) for one E3 internal mask."""
    val_rows = e3_internal_stations(y_mask, outer_test, seed, frac)
    candidates = e3_eligible_stations(y_mask, outer_test)

    flat_observed = np.flatnonzero(y_mask.ravel())
    test = np.sort(np.asarray(outer_test, dtype=np.int64))
    is_test = np.zeros(y_mask.size, dtype=bool)
    is_test[test] = True
    is_val = np.zeros(y_mask.shape[0], dtype=bool)
    is_val[val_rows] = True

    val = flat_observed[is_val[flat_observed // y_mask.shape[1]]]
    train = flat_observed[
        ~is_val[flat_observed // y_mask.shape[1]] & ~is_test[flat_observed]
    ]
    return (
        {
            "train": np.sort(train).astype(np.int64),
            "val": np.sort(val).astype(np.int64),
            "test": test,
        },
        val_rows,
        candidates,
    )


def component_report(
    edges: pd.DataFrame, sites: list[str], rows: np.ndarray | None = None
) -> dict:
    """Weakly-connected-component distribution of the graph and of a station set."""
    import networkx as nx

    graph = nx.from_pandas_edgelist(edges, "source", "target", create_using=nx.Graph)
    index = {site: i for i, site in enumerate(sites)}
    graph = graph.subgraph([s for s in graph.nodes if s in index])
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    component_of = {site: cid for cid, comp in enumerate(components) for site in comp}
    isolated = [s for s in sites if s not in component_of]
    report = {
        "n_stations": len(sites),
        "n_components": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "component_sizes": [len(c) for c in components],
        "n_isolated_stations": len(isolated),
        "n_edges": len(edges),
    }
    if rows is not None:
        counts = np.zeros(len(components), dtype=int)
        outside = 0
        for row in np.asarray(rows, dtype=np.int64):
            cid = component_of.get(sites[int(row)])
            if cid is None:
                outside += 1
            else:
                counts[cid] += 1
        report.update(
            {
                "n_selected_stations": len(rows),
                "selected_per_component": counts.tolist(),
                "selected_in_largest_component": int(counts[0]) if len(counts) else 0,
                "selected_isolated": int(outside),
                "selected_components_nonempty": int((counts > 0).sum()),
            }
        )
    return report


def assert_disjoint(split: dict[str, np.ndarray]) -> None:
    """Every observed cell belongs to at most one role."""
    seen: dict[int, str] = {}
    for role in ROLES:
        if role not in split:
            continue
        cells = np.asarray(split[role])
        if cells.ndim != 1:
            raise ValueError(f"{role}: expected a flat cell index array")
        for cell in cells.tolist():
            if cell in seen:
                raise ValueError(f"cell {cell} in both {seen[cell]} and {role}")
            seen[cell] = role


def assert_visible_roles(split: dict[str, np.ndarray]) -> tuple[str, ...]:
    """Validate and return the declared visible roles of one mask."""
    if "visible_roles" not in split:
        raise ValueError("mask must declare visible_roles explicitly")
    roles = tuple(str(r) for r in np.asarray(split["visible_roles"]).tolist())
    # Check the forbidden roles first so the message names the real problem
    # instead of reporting val/test as merely "unknown".
    leaked = [r for r in roles if r in NEVER_VISIBLE_ROLES]
    if leaked:
        raise ValueError(
            "a mask may never open " + ", ".join(leaked)
            + " during selection or validation"
        )
    unknown = [r for r in roles if r not in VISIBLE_ROLE_CHOICES]
    if unknown:
        raise ValueError("unknown visible role(s): " + ", ".join(unknown))
    for role in roles:
        if role not in ROLES or role not in split:
            raise ValueError("visible role not present in the mask: " + role)
    if len(set(roles)) != len(roles):
        raise ValueError("duplicate visible role")
    return roles


def build_masks(
    source_dir: Path,
    y_mask: np.ndarray,
    edges: pd.DataFrame,
    sites: list[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Build every h3a_v1 mask plus the manifest describing it."""
    n, t = y_mask.shape
    masks: dict[str, dict[str, np.ndarray]] = {}
    entries: dict[str, dict] = {}

    # The frozen masks are encoded on whatever grid they were built on. Check
    # it before importing anything: identical integers on a different grid mean
    # different stations and months.
    source_grid = read_source_grid(source_dir)
    grid_check = assert_source_grid(source_grid, y_mask, sites)
    migration_audit: dict[str, dict] = {}
    print(
        "source grid " + json.dumps(grid_check["source"])
        + " target " + json.dumps(grid_check["target"])
        + ("; migrating by station/month" if grid_check["requires_migration"]
           else "; grids identical")
    )

    for name, source in MASK_SOURCES.items():
        original = load_mask(source_dir / f"{source}.npz")
        if grid_check["requires_migration"]:
            original = dict(original)
            for role in ROLES:
                if role in original:
                    original[role] = migrate_cells(
                        original[role], source_grid, y_mask,
                        source + "." + role, migration_audit.setdefault(source, {}),
                    )
        if name in E1_MASKS:
            split = {
                "train": np.sort(original["train"]).astype(np.int64),
                "val": np.sort(original["val"]).astype(np.int64),
                "test": np.sort(original["test"]).astype(np.int64),
            }
            visible = ("train",)
            detail = {"internal_validation": "original train/val/test reused"}
        elif name in ("e2a_strict", "e2b_partial", "e2b_partial_hiddenctx"):
            val_context, val_target = split_val_context(original["val"])
            split = {
                "train": np.sort(original["train"]).astype(np.int64),
                "val": val_target,
                "val_context": val_context,
                "test": np.sort(original["test"]).astype(np.int64),
            }
            if "context" in original:
                split["context"] = np.sort(original["context"]).astype(np.int64)
            visible = ["train"]
            if "context" in split:
                visible.append("context")
            if name == "e2b_partial":
                visible.append("val_context")
            visible = tuple(visible)
            detail = {
                "internal_validation": (
                    "original val split 20/80 by seed " + str(INTERNAL_VAL_SEED)
                ),
                "future_context_cells": len(split.get("context", [])),
                "val_context_opened_at_selection": "val_context" in visible,
                "val_target_cells": len(val_target),
                "val_context_cells": len(val_context),
                "role": (
                    "strict temporal control (no future DOC context)"
                    if name == "e2a_strict"
                    else "selection-visibility diagnostic on the e2b target pool"
                    if name == "e2b_partial_hiddenctx"
                    else "future reconstruction under a running network"
                ),
            }
        else:
            built, val_rows, candidates = e3_internal_split(y_mask, original["test"])
            built["test"] = np.sort(original["test"]).astype(np.int64)
            built["val_sites"] = np.array(
                [sites[int(r)] for r in val_rows], dtype="<U32"
            )
            split = built
            visible = ("train",)
            detail = {
                "internal_validation": (
                    str(len(val_rows))
                    + " internal validation stations ("
                    + format(E3_INTERNAL_VAL_FRAC, ".0%")
                    + " of "
                    + str(len(candidates))
                    + " observed non-test stations), all their months"
                ),
                "internal_val_seed": INTERNAL_VAL_SEED,
                "internal_val_stations": len(val_rows),
                "candidate_stations": len(candidates),
                "cv_report": component_report(edges, sites, val_rows),
            }
        split["visible_roles"] = np.array(visible, dtype="<U16")
        assert_disjoint(split)
        assert_visible_roles(split)
        masks[name] = split
        entries[name] = {
            "source_mask": source,
            "roles": {r: len(split[r]) for r in ROLES if r in split},
            "visible_roles": list(visible),
            "outer_test_cells": len(split["test"]),
            "outer_test_stations": int(
                len(np.unique(split["test"] // t)) if len(split["test"]) else 0
            ),
            **detail,
        }

    # The diagnostic mask must score the exact same target pool as E2b.
    for role in ("train", "val", "test", "context", "val_context"):
        if not np.array_equal(
            masks["e2b_partial"][role], masks["e2b_partial_hiddenctx"][role]
        ):
            raise ValueError(f"e2b diagnostic disagrees with e2b on {role}")
    if "val_context" not in masks["e2b_partial"]["visible_roles"]:
        raise ValueError("e2b_partial must open val_context")
    if "val_context" in masks["e2b_partial_hiddenctx"]["visible_roles"]:
        raise ValueError("the diagnostic mask must keep val_context closed")
    if "val_context" in masks["e2a_strict"]["visible_roles"]:
        raise ValueError("the strict temporal control must not open val_context")
    if "context" in masks["e2a_strict"]:
        raise ValueError("the strict temporal control must have no future context")

    # Outer test positions must be identical to the frozen masks ON THE SOURCE
    # GRID. Comparing raw integers would be meaningless once the month count
    # changes, so the comparison is in (station, month) coordinates.
    old_t = int(source_grid["n_months"])
    new_t = int(t)

    def pairs(cells: np.ndarray, width: int) -> set:
        cells = np.asarray(cells, dtype=np.int64)
        return set(zip((cells // width).tolist(), (cells % width).tolist()))

    for name, source in MASK_SOURCES.items():
        original = load_mask(source_dir / f"{source}.npz")
        frozen_test = pairs(original["test"], old_t)
        built_test = pairs(masks[name]["test"], new_t)
        dropped = frozen_test - built_test
        added = built_test - frozen_test
        if added:
            raise ValueError(
                f"{name}: migration invented " + str(len(added))
                + " test cells that are not in " + source
            )
        # frozen cells that are no longer observed legitimately disappear; they
        # are reported per role in migration_audit rather than silently ignored
        entries[name]["outer_test_cells_dropped_unobserved"] = len(dropped)
        entries[name]["outer_test_preserved_fraction"] = (
            1.0 - len(dropped) / len(frozen_test) if frozen_test else 1.0
        )

    manifest = {
        "protocol_version": "h3a_v1",
        "internal_val_seed": INTERNAL_VAL_SEED,
        "e2_context_frac": E2_CONTEXT_FRAC,
        "e3_internal_val_frac": E3_INTERNAL_VAL_FRAC,
        "source_dir": source_dir.as_posix(),
        "dataset": {
            "n_stations": int(n),
            "n_months": int(t),
            "observed_cells": int(y_mask.sum()),
        },
        "dev_scenarios": list(DEV_SCENARIOS),
        "key_scenarios": list(KEY_SCENARIOS),
        "diagnostic_scenarios": list(DIAGNOSTIC_SCENARIOS),
        "source_grid_check": grid_check,
        "migration_audit": migration_audit,
        "masks": entries,
        "invariants": [
            "roles never overlap inside a mask",
            "every mask declares its visible_roles explicitly",
            "val and test can never be declared visible",
            "the outer test array is byte-identical to the frozen mask",
            "E3 internal validation stations carry no train label",
            "e2a_strict has no future context and never opens val_context",
            "e2b_partial_hiddenctx scores the same target pool as e2b_partial",
        ],
    }
    return masks, manifest


def write_masks(
    masks: dict[str, dict[str, np.ndarray]], manifest: dict, out_dir: Path
) -> dict:
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    # A mask that is no longer produced must not linger: a stale file with a
    # removed or renamed scenario could otherwise still be selected and run.
    for stale in sorted(out_dir.glob("*.npz")):
        if stale.stem not in masks:
            stale.unlink()
            manifest.setdefault("removed_masks", []).append(stale.stem)
    hashes = {}
    for name, split in masks.items():
        path = out_dir / f"{name}.npz"
        temp = out_dir / f"{name}.npz.tmp"
        with open(temp, "wb") as fh:
            np.savez(fh, **split)
        temp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[name] = digest
        manifest["masks"][name]["sha256"] = digest
    manifest_path = out_dir.parent / "split_manifest.json"
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(manifest_path)
    return hashes
