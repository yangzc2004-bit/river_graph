"""Build the unchanged public Mobius INCA-C model and run its complete example."""
from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from river_graph.experiments.provenance import sha256_file

BASE = Path("cache/doc_process_model_v1").resolve()
SOURCE = BASE/"Mobius"
OUTPUT = Path("experiments/phase4_transfer/doc_process_model_audit_v2")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    application = SOURCE/"Applications/IncaC"
    library = BASE/"incac.dylib"
    command = ["clang++", "-std=c++11", "-O2", "-fPIC", "-shared", "-D__unix__",
               "-Wno-deprecated-declarations", "incac_dll.cpp", "-o", str(library)]
    build = subprocess.run(command, cwd=application, capture_output=True, text=True, check=False)
    (OUTPUT/"build.log").write_text(build.stdout+build.stderr)
    record = {"model": "INCA-C, public Mobius/PERSiST continuation", "source_url": "https://github.com/NIVANorge/Mobius",
        "source_commit": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "build_command": command, "build_returncode": build.returncode,
        "equations_modified": False, "portability": "POSIX export macro enabled through compiler flag on macOS",
        "original_2007_paper_reproduction": False, "mississippi_reproduction": False,
        "readme_caveat": "Public examples are explicitly not correctly calibrated; Mobius uses integrated PERSiST and additional later INCA-C processes",
        "files": {str(path.relative_to(SOURCE)): sha256_file(path) for path in SOURCE.rglob("*")
                  if path.is_file() and ".git" not in path.parts and
                  (path.suffix in (".h", ".cpp") or path.name in ("mobius.py", "inputs_Storgama.dat", "params_Storgama_optimized02.dat"))},
        "probe_script_sha256": sha256_file(Path(__file__)),
        "previous_attempt": "doc_process_model_audit_v1/audit.json records an incompatible unmodified Langtjern parameter file"}
    if build.returncode:
        record["status"] = "build_failed"
    else:
        specification = importlib.util.spec_from_file_location("mobius_doc_audit", SOURCE/"PythonWrapper/mobius.py")
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        module.initialize(str(library))
        try:
            dataset = module.DataSet.setup_from_parameter_and_input_files(
                str(application/"Storgama/params_Storgama_optimized02.dat"), str(application/"Storgama/inputs_Storgama.dat"))
            started = time.monotonic()
            finished = dataset.run_model(ms_timeout=60000)
            duration = time.monotonic()-started
            doc = dataset.get_result_series("Reach DOC concentration", ["Storgama"])
            start = dataset.get_parameter_time("Start date", [])
            dates = pd.date_range(start, periods=len(doc), freq="D")
            prediction_path = OUTPUT/"storgama_example.parquet"
            pd.DataFrame({"date": dates, "doc_mg_l": doc}).to_parquet(prediction_path, index=False)
            record.update({"status": "example_completed" if finished else "example_timeout", "finished": bool(finished),
                "timesteps": len(doc), "start": str(dates[0].date()), "end": str(dates[-1].date()),
                "simulation_seconds": duration, "finite_outputs": int(np.isfinite(doc).sum()),
                "negative_outputs": int(np.count_nonzero(doc < 0)), "doc_unit": dataset.get_result_unit("Reach DOC concentration"),
                "doc_min": float(np.nanmin(doc)), "doc_max": float(np.nanmax(doc)),
                "output_sha256": sha256_file(prediction_path),
                "native_input_list": dataset.get_input_list(), "parameter_count": len(dataset.get_parameter_list())})
            (OUTPUT/"storgama_example.meta.json").write_text(json.dumps({
                "model": record["model"], "source_commit": record["source_commit"],
                "prediction_sha256": record["output_sha256"], "probe_script_sha256": record["probe_script_sha256"],
                "parameter_sha256": sha256_file(application/"Storgama/params_Storgama_optimized02.dat"),
                "inputs_sha256": sha256_file(application/"Storgama/inputs_Storgama.dat"),
                "scope": "native public example execution only; no Mississippi score or paper reproduction"},indent=2)+"\n")
            dataset.delete()
        except Exception as error:  # noqa: BLE001 - record upstream build/runtime feasibility failures
            record.update({"status": "example_failed", "error": str(error)})
    (OUTPUT/"audit.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k not in ("files", "native_input_list")}))


if __name__ == "__main__":
    main()
