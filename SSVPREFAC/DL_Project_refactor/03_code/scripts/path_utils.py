from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPTS_DIR.parent
REPO_ROOT = CODE_DIR.parent

SRC_DIR = CODE_DIR / "src"
CONFIG_DIR = CODE_DIR / "configs"

DATA_ROOT_DIR = REPO_ROOT / "04_data"
DATASETS_DIR = DATA_ROOT_DIR / "datasets"
SAMPLE_INPUTS_DIR = DATA_ROOT_DIR / "sample_inputs"

RESULTS_DIR = REPO_ROOT / "05_results"
LOGS_DIR = RESULTS_DIR / "ablations"
ABLATIONS_DIR = RESULTS_DIR / "ablations"
ABLATION_JSON_DIR = RESULTS_DIR / "ablation_json_results"
FIGURES_DIR = RESULTS_DIR / "figures"

DEMO_DIR = REPO_ROOT / "06_demo"


def ensure_import_paths(include_scripts: bool = False) -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    if include_scripts and str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


def default_config_path(config_name: str = "default.yaml") -> str:
    return str(CONFIG_DIR / config_name)
