"""T03: scenario-matched internal validation splits for the H3-A protocol.

The frozen benchmark masks assign every observed cell to train / val / test.
H3-A keeps the *outer* test positions untouched and hidden, and replaces the
single internal "val" role with an arrangement that matches the shift each
scenario is about:

* e1_*             random cell split  -> reuse the original train/val/test
* e2a/e2b_partial  temporal split     -> keep the original train; split the
  original val into val_context (20%) and val_target (80%).  E2a hides both
  during selection, E2b opens only val_context; both are scored on the same
  val_target.  E2a and E2b may therefore select *different* best weights,
  unlike the old "weights must be identical" assumption.
* e3_internal_*    spatial split      -> keep the original held-out test
  stations; pick ~10% of the observed non-test stations as internal validation
  stations and give all their months to val; every remaining non-test cell
  goes to train.

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
E2_SOURCE = "e2b_partial"
E3_SOURCES = ("e3_spatial_seed42", "e3_spatial_seed43", "e3_spatial_seed44")

# name -> source mask in experiments/masks
MASK_SOURCES = {
    **{name: name for name in E1_MASKS},
    "e2a_partial": E2_SOURCE,
    "e2b_partial": E2_SOURCE,
    **{f"e3_internal_seed{s}": f"e3_spatial_seed{s}" for s in (42, 43, 44)},
}

DEV_SCENARIOS = ("e1_r20_seed42", "e2b_partial", "e3_internal_seed42")
KEY_SCENARIOS = (
    "e1_r20_seed42",
    "e1_r20_seed43",
    "e1_r20_seed44",
    "e2a_partial",
    "e2b_partial",
    "e3_internal_seed42",
    "e3_internal_seed43",
    "e3_internal_seed44",
)

ROLES = ("train", "val", "val_context", "context", "test")


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

    for name, source in MASK_SOURCES.items():
        original = load_mask(source_dir / f"{source}.npz")
        if name in E1_MASKS:
            split = {
                "train": np.sort(original["train"]).astype(np.int64),
                "val": np.sort(original["val"]).astype(np.int64),
                "test": np.sort(original["test"]).astype(np.int64),
            }
            detail = {"internal_validation": "original train/val/test reused"}
        elif name in ("e2a_partial", "e2b_partial"):
            if "context" not in original:
                raise ValueError(f"{source}: E2 mask has no future context")
            val_context, val_target = split_val_context(original["val"])
            split = {
                "train": np.sort(original["train"]).astype(np.int64),
                "val": val_target,
                "test": np.sort(original["test"]).astype(np.int64),
                "context": np.sort(original["context"]).astype(np.int64),
            }
            if name == "e2b_partial":
                split["val_context"] = val_context
            detail = {
                "internal_validation": (
                    "original val split 20/80 by seed " + str(INTERNAL_VAL_SEED)
                ),
                "val_context_visible_at_selection": name == "e2b_partial",
                "val_target_cells": len(val_target),
                "val_context_cells": len(val_context),
            }
        else:
            built, val_rows, candidates = e3_internal_split(y_mask, original["test"])
            built["test"] = np.sort(original["test"]).astype(np.int64)
            built["val_sites"] = np.array(
                [sites[int(r)] for r in val_rows], dtype="<U32"
            )
            split = built
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
        assert_disjoint(split)
        masks[name] = split
        entries[name] = {
            "source_mask": source,
            "roles": {r: len(split[r]) for r in ROLES if r in split},
            "outer_test_cells": len(split["test"]),
            "outer_test_stations": int(
                len(np.unique(split["test"] // t)) if len(split["test"]) else 0
            ),
            **detail,
        }

    # E2a and E2b must be scored on the same val_target and share every role.
    for role in ("train", "val", "test", "context"):
        if not np.array_equal(masks["e2a_partial"][role], masks["e2b_partial"][role]):
            raise ValueError(f"E2a/E2b disagree on {role}")
    if "val_context" in masks["e2a_partial"]:
        raise ValueError("E2a must not expose val_context")

    # Outer test positions must be identical to the frozen masks.
    for name, source in MASK_SOURCES.items():
        original = load_mask(source_dir / f"{source}.npz")
        if not np.array_equal(np.sort(masks[name]["test"]), np.sort(original["test"])):
            raise ValueError(f"{name}: outer test changed relative to {source}")

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
        "masks": entries,
        "invariants": [
            "roles never overlap inside a mask",
            "the outer test array is byte-identical to the frozen mask",
            "E3 internal validation stations carry no train label",
            "E2a and E2b share one val_target",
        ],
    }
    return masks, manifest


def write_masks(
    masks: dict[str, dict[str, np.ndarray]], manifest: dict, out_dir: Path
) -> dict:
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
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
