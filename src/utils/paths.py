from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
AUDIT_DIR = REPORTS_DIR / "data_audit"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def ensure_project_directories() -> None:
    """Create project directories if they do not already exist."""

    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        AUDIT_DIR,
        CHECKPOINTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)