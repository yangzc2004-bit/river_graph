"""Temporary recovery helper: warm the per-station WQP cache in parallel.

The dataset build fetches stations sequentially, which is fine for a few new
stations but not for rebuilding 571 of them. This uses the same resumable
fetch_station_results function from river_graph.data.wqp, only with a small
thread pool, and skips anything already cached.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from river_graph.data.wqp import fetch_station_results

CACHE = ROOT / "data/raw/wqp_results"
NODES = ROOT / "data/processed/graph_nodes.csv"


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sites = list(pd.read_csv(NODES, dtype={"site_no": str})["site_no"])
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [s for s in sites if not (CACHE / (s + ".csv")).exists()]
    print(f"sites={len(sites)} cached={len(sites) - len(todo)} todo={len(todo)}")
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_station_results, s, CACHE): s for s in todo}
        for future in as_completed(futures):
            site = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                print("FAILED " + site + ": " + str(exc), flush=True)
                failed += 1
                continue
            if result is None:
                print("FAILED " + site, flush=True)
                failed += 1
            else:
                done += 1
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)} fetched", flush=True)
    print(f"warmed {done}, failed {failed}, cached now "
          f"{len(list(CACHE.glob('*.csv')))}/{len(sites)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
