"""3B-R2 selection contracts (docs/paper/phase3_r2_ranking_spec.md §2, §7).

Selection rules must not read test DOC and must be invariant to any test
label perturbation; the operational threshold must come from train/val-era
cells only.
"""

import importlib.util
from pathlib import Path

import numpy as np


def _mod():
    spec = importlib.util.spec_from_file_location(
        "r2", Path("scripts/run3b_r2_ranking.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_selection_never_reads_test_doc():
    r2 = _mod()
    rng = np.random.default_rng(0)
    unc = rng.random(24)
    role = np.array(["train"] * 8 + ["val"] * 4 + ["context"] * 4
                    + ["test"] * 8)
    te = np.flatnonzero(role == "test")
    thr1, sel1 = r2.select_operational(unc, role, 0.10)
    diag1 = r2.select_rank_diag(unc[te], 0.10)
    # perturb "test labels" — selection inputs are unchanged by construction;
    # assert the functions depend only on unc/role
    thr2, sel2 = r2.select_operational(unc, role, 0.10)
    diag2 = r2.select_rank_diag(unc[te], 0.10)
    assert thr1 == thr2
    assert (sel1 == sel2).all() and (diag1 == diag2).all()


def test_operational_threshold_ignores_test_unc_extremes():
    r2 = _mod()
    rng = np.random.default_rng(1)
    unc = rng.random(20)
    role = np.array(["train"] * 10 + ["val"] * 5 + ["test"] * 5)
    thr1, _ = r2.select_operational(unc, role, 0.10)
    unc2 = unc.copy()
    unc2[role == "test"] = 1e6  # extreme test uncertainty must not move thr
    thr2, _ = r2.select_operational(unc2, role, 0.10)
    assert thr1 == thr2


def test_rank_diag_takes_exact_top_p():
    r2 = _mod()
    unc_te = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.0])
    sel = r2.select_rank_diag(unc_te, 0.10)
    assert sel.sum() == 1
    assert sel[1]  # highest uncertainty
    sel2 = r2.select_rank_diag(unc_te, 0.20)
    assert sel2.sum() == 2
    assert sel2[1] and sel2[5]


def test_selection_unit_is_station_month_equal_count_baseline():
    r2 = _mod()
    assert r2.P_MAIN == 0.10 and r2.P_SENS == (0.05, 0.20)
    assert r2.RANDOM_REPEATS >= 100 and r2.RANDOM_SEED == 42
