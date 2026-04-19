"""
V1 Smoke Test
=============

Runs ONE iteration of the V1 SW sweep at the lowest ramp height to verify
end-to-end plumbing before launching the full sweep.

Tests in order: SW connect, assembly open, cooler component lookup, equation
manager access, equation index lookup, set/rebuild/readback, .x_t export.

Output: a single ramp_smoke.x_t in output/. Does not write JSONL or summary.

Usage:
    python smoke_test.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from run import (
    OUTPUT_DIR,
    connect_sw,
    find_equation_index,
    load_config,
    open_assembly,
    open_cooler_part,
    run_iteration,
    setup_logging,
)

logger = logging.getLogger("v1_smoke_test")


SMOKE_VALUE_MM = 1.0


def main():
    setup_logging()
    cfg = load_config()
    logger.info(f"Smoke test: one iteration at {SMOKE_VALUE_MM} mm "
                f"(picked to differ from any default — confirms the setter actually writes).")

    sw = connect_sw()
    asm = open_assembly(sw, cfg["assembly_path"])
    part_model, eq_mgr = open_cooler_part(sw, cfg["cooler_part_filename"])
    eq_index = find_equation_index(eq_mgr, cfg["equation_var_name"])

    output_path = OUTPUT_DIR / "ramp_smoke.x_t"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        result = run_iteration(
            sw, asm, part_model, eq_mgr, eq_index,
            cfg["equation_var_name"], SMOKE_VALUE_MM, output_path,
        )
    except Exception as e:
        logger.error(f"Smoke test FAILED: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Smoke test PASSED.")
    logger.info(
        f"  ramp={result['ramp_value_mm']:.3f} mm | "
        f"rebuild {result['rebuild_part_seconds']+result['rebuild_asm_seconds']:.2f}s | "
        f"export {result['export_seconds']:.2f}s | "
        f"{result['file_size_bytes']/1024:.1f} KB"
    )
    logger.info(f"Output: {output_path}")
    logger.info("If this passed, run.py should work end-to-end.")


if __name__ == "__main__":
    main()
