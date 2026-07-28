from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every synthetic pipeline stage")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "smoke_synthetic.json"),
    )
    args = parser.parse_args()
    scripts = [
        "01_collect_traces.py",
        "02_fourier_audit.py",
        "03_fit_shared_subspace.py",
        "04_build_counterfactual_prefixes.py",
        "05_run_observational_cross.py",
        "06_run_causal_interventions.py",
        "07_analyze_results.py",
    ]
    root = Path(__file__).resolve().parents[1]
    for script in scripts:
        print(f"==> {script}", flush=True)
        subprocess.run(
            [sys.executable, str(root / "scripts" / script), "--config", args.config],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()

