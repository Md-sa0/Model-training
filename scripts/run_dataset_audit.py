from __future__ import annotations

from pathlib import Path

from qwen_sepsis.dataset_audit import DEFAULT_OUTPUT_DIR, DEFAULT_RAW_PATH, run_audit


def main() -> None:
    run_audit(DEFAULT_RAW_PATH, DEFAULT_OUTPUT_DIR)


if __name__ == "__main__":
    main()
