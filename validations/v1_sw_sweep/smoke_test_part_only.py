"""
V1 Smoke Test — Diagnostic
==========================

Bisects what the script does differently from manual editing. Run sequence:
  1. Open part (no assembly), DO NOT edit, just rebuild — does EditRebuild3 alone trip?
  2. If (1) succeeded, edit InletRampHeight and rebuild — does the edit trip it?

This isolates whether the bug is in OpenDoc6+EditRebuild3 or in set_equation.
"""

import logging
import sys

from run import (
    MM_PER_INCH,
    connect_sw,
    find_equation_index,
    load_config,
    open_cooler_part,
    parse_equation_rhs_inches,
    set_equation,
    setup_logging,
)

logger = logging.getLogger("v1_smoke_part_only")

SMOKE_VALUE_MM = 1.0


def main():
    setup_logging()
    cfg = load_config()

    sw = connect_sw()
    part_model, eq_mgr = open_cooler_part(sw, cfg["cooler_part_filename"])
    eq_index = find_equation_index(eq_mgr, cfg["equation_var_name"])

    # Stage 1: rebuild without editing anything
    logger.info("STAGE 1 — rebuild WITHOUT editing the equation")
    rebuild_ok = bool(part_model.EditRebuild3)
    logger.info(f"  EditRebuild3 returned: {rebuild_ok}")
    if not rebuild_ok:
        logger.error(
            "Part rebuild FAILED with no equation edit. "
            "OpenDoc6+EditRebuild3 alone is breaking the part. "
            "The set_equation step is innocent."
        )
        sys.exit(1)
    logger.info("  STAGE 1 PASSED — open+rebuild is clean.")

    # Stage 2: edit, then rebuild
    logger.info(f"STAGE 2 — edit InletRampHeight to {SMOKE_VALUE_MM} mm and rebuild")
    eq_before = eq_mgr.Equation(eq_index)
    logger.info(f"  Before: {eq_before!r}")

    value_in = SMOKE_VALUE_MM / MM_PER_INCH
    new_eq = f'"{cfg["equation_var_name"]}" = {value_in:.6f}'
    set_equation(eq_mgr, eq_index, new_eq)

    eq_after_set = eq_mgr.Equation(eq_index)
    logger.info(f"  After set: {eq_after_set!r}")

    rebuild_ok = bool(part_model.EditRebuild3)
    logger.info(f"  EditRebuild3 returned: {rebuild_ok}")

    eq_after_rebuild = eq_mgr.Equation(eq_index)
    readback_in = parse_equation_rhs_inches(eq_after_rebuild)
    readback_mm = readback_in * MM_PER_INCH
    logger.info(f"  After rebuild: {eq_after_rebuild!r} ({readback_mm:.4f} mm)")

    if not rebuild_ok:
        logger.error(
            "STAGE 2 FAILED — set_equation broke the part rebuild "
            "(stage 1 was clean, so the edit is the culprit)."
        )
        sys.exit(1)

    if abs(readback_mm - SMOKE_VALUE_MM) > 1e-3:
        logger.error(f"Readback mismatch: wanted {SMOKE_VALUE_MM} mm, got {readback_mm} mm.")
        sys.exit(1)

    logger.info("STAGE 2 PASSED — edit + rebuild is clean too. Bug is elsewhere.")


if __name__ == "__main__":
    main()
