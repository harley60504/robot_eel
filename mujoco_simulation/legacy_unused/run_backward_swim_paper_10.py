from __future__ import annotations

import sys
from pathlib import Path

import run_free_swim_paper_10 as free_swim_batch


DEFAULT_PREFIX = "backward_negfreq_w100_ew08_a20_targetv11_ppo_free_swim_freq_phase"
DEFAULT_ORGANIZED_DIR = Path("outputs/paper_backward_swim_10_targetv11_w100_ew08_a20_extracted_figures")
DEFAULT_TARGET_SPEED = -0.11
DEFAULT_START_X = 2.40
DEFAULT_START_Y = 0.0
DEFAULT_FREQ_LOW = -1.25
DEFAULT_FREQ_HIGH = -1.0
DEFAULT_PHASE_LAG_LOW = 0.5
DEFAULT_PHASE_LAG_HIGH = 0.8


def add_default_flag(argv: list[str], flag: str, value: object) -> None:
    if flag not in argv[1:]:
        argv.extend([flag, str(value)])


def main() -> None:
    argv = [sys.argv[0]]
    user_args = sys.argv[1:]
    argv.extend(user_args)

    add_default_flag(argv, "--runs", 10)
    add_default_flag(argv, "--prefix", DEFAULT_PREFIX)
    add_default_flag(argv, "--organized-dir", DEFAULT_ORGANIZED_DIR)
    add_default_flag(argv, "--target-speed", DEFAULT_TARGET_SPEED)
    add_default_flag(argv, "--start-x", DEFAULT_START_X)
    add_default_flag(argv, "--start-y", DEFAULT_START_Y)
    add_default_flag(argv, "--freq-low", DEFAULT_FREQ_LOW)
    add_default_flag(argv, "--freq-high", DEFAULT_FREQ_HIGH)
    add_default_flag(argv, "--phase-lag-low", DEFAULT_PHASE_LAG_LOW)
    add_default_flag(argv, "--phase-lag-high", DEFAULT_PHASE_LAG_HIGH)

    sys.argv = argv
    free_swim_batch.main()


if __name__ == "__main__":
    main()
