"""
V1 SolidWorks Sweep Validation
==============================

Sweeps inlet manifold ramp height across a parametric SolidWorks assembly,
exporting Parasolid (.x_t) at each step.

Per-iteration pass requires:
  1. Rebuild returns no error
  2. Read-back equation value matches the value written
  3. Output .x_t file exists and is non-zero size

Aggregate pass requires all iterations pass. Fails fast on first error.
SolidWorks is left open at exit regardless of outcome.

Usage:
    python run.py
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger("v1_sw_sweep")

# SolidWorks API constants (swconst.h)
SW_DOC_ASSEMBLY = 2
SW_DOC_PART = 1
SW_OPEN_DOC_OPTIONS_SILENT = 1
SW_SAVE_AS_CURRENT_VERSION = 0
SW_SAVE_AS_OPTIONS_SILENT = 1
SW_SAVE_AS_OPTIONS_COPY = 2
SW_COMPONENT_RESOLVED = 2

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "config.json"
OUTPUT_DIR = HERE / "output"
JSONL_PATH = OUTPUT_DIR / "runs.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.txt"
LOG_PATH = OUTPUT_DIR / "run.log"


# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    required = [
        "assembly_path", "cooler_part_filename", "equation_var_name",
        "manifold_pipe_height_mm", "ramp_min_mm", "ramp_max_fraction", "n_samples",
    ]
    missing = [k for k in required if cfg.get(k) is None]
    if missing:
        raise ValueError(
            f"config.json missing required fields: {missing}. "
            f"Fill these in before running."
        )

    ramp_max = cfg["manifold_pipe_height_mm"] * cfg["ramp_max_fraction"]
    if ramp_max <= cfg["ramp_min_mm"]:
        raise ValueError(
            f"ramp_max ({ramp_max:.3f} mm) <= ramp_min ({cfg['ramp_min_mm']} mm). "
            f"Check pipe height and fraction."
        )

    return cfg


def ramp_heights_mm(cfg):
    """Linear sweep of ramp heights from ramp_min_mm to ramp_max_fraction*pipe_height."""
    lo = cfg["ramp_min_mm"]
    hi = cfg["manifold_pipe_height_mm"] * cfg["ramp_max_fraction"]
    return np.linspace(lo, hi, cfg["n_samples"])


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


# ------------------------------------------------------------------------------
# SolidWorks COM
# ------------------------------------------------------------------------------

def _byref_int():
    """VARIANT byref int32 for COM out-params in late binding."""
    import pythoncom
    import win32com.client
    return win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def connect_sw():
    """Attach to running SolidWorks or launch a new instance. Always leaves SW visible.
    Late binding — the equation setter uses raw IDispatch Invoke so we don't need
    early-bound type info."""
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()
    sw = win32com.client.Dispatch("SldWorks.Application")
    sw.Visible = True
    logger.info(f"Connected to SolidWorks (version {sw.RevisionNumber})")
    return sw


def open_assembly(sw, asm_path):
    """Open the assembly. If already open, OpenDoc6 returns the existing handle."""
    asm_path = Path(asm_path)
    if not asm_path.is_absolute():
        asm_path = (HERE / asm_path).resolve()
    if not asm_path.exists():
        raise FileNotFoundError(f"Assembly not found: {asm_path}")

    errors = _byref_int()
    warnings = _byref_int()
    asm = sw.OpenDoc6(
        str(asm_path),
        SW_DOC_ASSEMBLY,
        SW_OPEN_DOC_OPTIONS_SILENT,
        "",
        errors, warnings,
    )
    if asm is None:
        raise RuntimeError(f"OpenDoc6 returned None for {asm_path} (errors={errors.value}, warnings={warnings.value})")
    logger.info(f"Opened assembly: {asm_path.name}")
    return asm


def open_cooler_part(sw, part_filename):
    """Open the cooler part as its own document. Necessary because EquationMgr obtained
    via assembly→component→GetModelDoc2 is read-only — the PROPPUT setter silently no-ops.
    Opening the part directly gives a writable EquationMgr.
    Both the assembly and this part doc stay open simultaneously."""
    part_path = (HERE / part_filename).resolve()
    if not part_path.exists():
        raise FileNotFoundError(f"Cooler part not found: {part_path}")
    errors = _byref_int()
    warnings = _byref_int()
    part_doc = sw.OpenDoc6(
        str(part_path),
        SW_DOC_PART,
        SW_OPEN_DOC_OPTIONS_SILENT,
        "", errors, warnings,
    )
    if part_doc is None:
        raise RuntimeError(f"OpenDoc6 returned None for {part_path}")
    sw.ActivateDoc3(part_path.name, True, 0, errors)
    eq_mgr = part_doc.GetEquationMgr
    if eq_mgr is None:
        raise RuntimeError("Part's GetEquationMgr returned None")
    logger.info(f"Opened cooler part: {part_path.name} ({eq_mgr.GetCount} equations)")
    return part_doc, eq_mgr


def find_equation_index(eq_mgr, var_name):
    """Find equation index by global variable name (case-insensitive)."""
    n = eq_mgr.GetCount
    for i in range(n):
        eq = eq_mgr.Equation(i)
        # Equation format: '"VarName"= 0.005' (quotes around name, value in SI)
        lhs = eq.split("=", 1)[0].strip().strip('"')
        if lhs.lower() == var_name.lower():
            return i
    raise ValueError(f"Global variable '{var_name}' not found in equation manager")


MM_PER_INCH = 25.4


def parse_equation_rhs_inches(eq_str):
    """Extract numeric RHS as a bare number. Document units = inches in this part,
    so unitless equation values are inches. Format: '"Name"= 0.0196850'."""
    import re
    rhs = eq_str.split("=", 1)[1].strip()
    m = re.match(r"\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", rhs)
    if not m:
        raise ValueError(f"Could not parse equation RHS: {rhs!r}")
    return float(m.group(1))


def set_equation(eq_mgr, index, equation_string):
    """Set equation at index via raw IDispatch Invoke. SW exposes .Equation as an
    indexed COM PROPPUT — pywin32 late binding can't generate the setter directly
    (no DISPID_PROPERTYPUT named-arg), so we Invoke it ourselves with param order
    (index, value). EvaluateAll forces equation re-evaluation."""
    import pythoncom
    dispid = eq_mgr._oleobj_.GetIDsOfNames(0, "Equation")
    eq_mgr._oleobj_.Invoke(
        dispid, 0, pythoncom.DISPATCH_PROPERTYPUT, 0, index, equation_string
    )
    _ = eq_mgr.EvaluateAll


def save_doc_copy(doc, output_path):
    """Generic SW SaveAs2 with Copy flag — format determined by output extension
    (.x_t, .SLDPRT, .SLDASM, .step, .iges, etc.). Copy flag keeps the working
    doc pointed at the original file. Returns True on success."""
    import pythoncom
    import win32com.client
    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = _byref_int()
    warnings = _byref_int()
    null_obj = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)
    empty_str = win32com.client.VARIANT(pythoncom.VT_BSTR, "")
    ok = doc.Extension.SaveAs2(
        str(output_path),
        SW_SAVE_AS_CURRENT_VERSION,
        SW_SAVE_AS_OPTIONS_SILENT | SW_SAVE_AS_OPTIONS_COPY,
        null_obj, empty_str, False,
        errors, warnings,
    )
    if not ok:
        logger.error(f"  SaveAs2 failed for {output_path.name}: "
                     f"errors={errors.value}, warnings={warnings.value}")
    return bool(ok)


# ------------------------------------------------------------------------------
# Sweep iteration
# ------------------------------------------------------------------------------

def run_iteration(sw, asm, part_model, eq_mgr, eq_index, var_name, ramp_value_mm, output_path):
    """Run one parameter iteration. Returns dict of metrics. Raises on any failure."""
    result = {
        "ramp_value_mm": float(ramp_value_mm),
        "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # Activate the cooler part — eq_mgr writes only succeed when the part is the
    # active doc (previous iteration ended with the asm active for export).
    activate_part_errs = _byref_int()
    sw.ActivateDoc3(part_model.GetTitle, True, 0, activate_part_errs)

    # Document units are inches. Convert mm → inches for the equation value.
    # Match SW's stored format spacing exactly ('"name" = value' with spaces both
    # sides of =) — SW seems to silently reject writes whose format differs from
    # how the user typed the equation in the UI.
    value_inches = ramp_value_mm / MM_PER_INCH
    eq_string = f'"{var_name}" = {value_inches:.6f}'

    # 1. Set equation
    t0 = time.time()
    set_equation(eq_mgr, eq_index, eq_string)
    result["equation_value_set_mm"] = float(ramp_value_mm)
    result["equation_value_set_in"] = value_inches

    # 2. Rebuild part, save it to disk, then rebuild assembly.
    # - Part: EditRebuild3 (incremental). ForceRebuild3 trips intermediate features
    #   ("intended cut does not intersect the model").
    # - Save part to disk: forces the assembly's component reference to see a
    #   timestamp change. Without this, the assembly's EditRebuild3 doesn't pick up
    #   in-memory edits from a sibling open doc, and ForceRebuild3 trips errors too.
    # - Assembly: EditRebuild3 picks up the updated cooler from disk.
    # EditRebuild3 is a property in pywin32 late binding — access without parens.
    if not bool(part_model.EditRebuild3):
        raise RuntimeError(
            f"Part rebuild failed at ramp={ramp_value_mm:.3f} mm — "
            f"check the cooler feature tree for red error indicators"
        )
    # Save3(options, errors, warnings) — silent flag = 1
    save_errors = _byref_int()
    save_warnings = _byref_int()
    if not bool(part_model.Save3(SW_SAVE_AS_OPTIONS_SILENT, save_errors, save_warnings)):
        raise RuntimeError(
            f"Cooler save-to-disk failed at ramp={ramp_value_mm:.3f} mm "
            f"(errors={save_errors.value}, warnings={save_warnings.value})"
        )
    result["rebuild_part_seconds"] = round(time.time() - t0, 3)

    t1 = time.time()
    # Activate the assembly first — switching the active doc tells SW to detect
    # modified component files and refresh references before rebuild. Without this,
    # asm.EditRebuild3 doesn't pick up the just-saved cooler.
    activate_errors = _byref_int()
    sw.ActivateDoc3(asm.GetTitle, True, 0, activate_errors)
    if not bool(asm.EditRebuild3):
        raise RuntimeError(
            f"Assembly rebuild failed at ramp={ramp_value_mm:.3f} mm — "
            f"check the assembly feature tree for red error indicators"
        )
    result["rebuild_asm_seconds"] = round(time.time() - t1, 3)

    # 3. Read back equation, verify match (catches silent COM write failures)
    eq_after = eq_mgr.Equation(eq_index)
    value_readback_in = parse_equation_rhs_inches(eq_after)
    value_readback_mm = value_readback_in * MM_PER_INCH
    result["equation_value_readback_mm"] = value_readback_mm
    if abs(value_readback_mm - ramp_value_mm) > 1e-4:
        raise RuntimeError(
            f"Dimension read-back mismatch: wrote {ramp_value_mm} mm "
            f"({value_inches:.6f} in), read {value_readback_mm} mm "
            f"({value_readback_in:.6f} in) (raw eq: {eq_after!r})"
        )

    # 4. Export .x_t
    t2 = time.time()
    ok = save_doc_copy(asm, output_path)
    result["export_seconds"] = round(time.time() - t2, 3)
    if not ok:
        raise RuntimeError(f"SaveAs2 returned False at ramp={ramp_value_mm:.3f} mm")
    if not output_path.exists():
        raise RuntimeError(f"Output file missing after SaveAs2: {output_path}")
    size = output_path.stat().st_size
    result["file_size_bytes"] = size
    if size == 0:
        raise RuntimeError(f"Output file is zero size: {output_path}")

    result["pass"] = True
    return result


def write_jsonl_row(row):
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_summary(cfg, results, started_at, ended_at, status):
    elapsed = ended_at - started_at
    minutes, seconds = divmod(int(elapsed.total_seconds()), 60)

    lines = [
        "V1 SolidWorks Sweep — Summary",
        "=" * 30,
        f"Started: {started_at.isoformat(timespec='seconds')}",
        f"Ended:   {ended_at.isoformat(timespec='seconds')}",
        f"Total:   {minutes}m {seconds}s",
        "",
        f"Iterations: {sum(1 for r in results if r.get('pass'))}/{cfg['n_samples']} passed",
        "",
        f"Ramp range: {cfg['ramp_min_mm']:.3f} mm -> "
        f"{cfg['manifold_pipe_height_mm'] * cfg['ramp_max_fraction']:.3f} mm "
        f"({cfg['manifold_pipe_height_mm']} mm pipe × {cfg['ramp_max_fraction']:.2f})",
        f"Sample spacing: linear, n={cfg['n_samples']}",
        "",
    ]

    if results and all(r.get("pass") for r in results):
        rebuild_times = [r["rebuild_part_seconds"] + r["rebuild_asm_seconds"] for r in results]
        export_times = [r["export_seconds"] for r in results]
        sizes = [r["file_size_bytes"] for r in results]
        lines.extend([
            f"Rebuild time (s): mean={np.mean(rebuild_times):.2f}, "
            f"median={np.median(rebuild_times):.2f}, "
            f"range=[{min(rebuild_times):.2f}, {max(rebuild_times):.2f}]",
            f"Export time (s):  mean={np.mean(export_times):.2f}, "
            f"median={np.median(export_times):.2f}, "
            f"range=[{min(export_times):.2f}, {max(export_times):.2f}]",
            f"File size (bytes): min={min(sizes)}, max={max(sizes)}",
            "",
        ])

    lines.append(f"Status: {status}")

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    setup_logging()
    cfg = load_config()
    logger.info(f"Config loaded. n_samples={cfg['n_samples']}, "
                f"pipe_height={cfg['manifold_pipe_height_mm']} mm, "
                f"max_fraction={cfg['ramp_max_fraction']}")

    # Reset JSONL
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()

    sw = connect_sw()
    asm = open_assembly(sw, cfg["assembly_path"])
    part_model, eq_mgr = open_cooler_part(sw, cfg["cooler_part_filename"])
    eq_index = find_equation_index(eq_mgr, cfg["equation_var_name"])
    logger.info(f"Found global variable '{cfg['equation_var_name']}' at index {eq_index}")

    heights = ramp_heights_mm(cfg)
    started_at = datetime.now()
    results = []
    status = "FAIL"

    try:
        for i, h in enumerate(heights):
            output_path = OUTPUT_DIR / f"ramp_{i:02d}.x_t"
            logger.info(f"[{i+1}/{len(heights)}] ramp={h:.3f} mm -> {output_path.name}")
            row = run_iteration(sw, asm, part_model, eq_mgr, eq_index,
                                cfg["equation_var_name"], h, output_path)
            row["iteration"] = i
            results.append(row)
            write_jsonl_row(row)
            logger.info(
                f"  OK  | rebuild {row['rebuild_part_seconds']+row['rebuild_asm_seconds']:.2f}s "
                f"| export {row['export_seconds']:.2f}s "
                f"| {row['file_size_bytes']/1024:.1f} KB"
            )
        status = "PASS"
        logger.info(f"All {len(heights)} iterations passed.")
    except Exception as e:
        logger.error(f"Sweep halted: {e}", exc_info=True)
        write_jsonl_row({
            "iteration": len(results),
            "pass": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        raise
    finally:
        ended_at = datetime.now()
        write_summary(cfg, results, started_at, ended_at, status)
        logger.info(f"Summary written to {SUMMARY_PATH}")
        logger.info("SolidWorks left open per config.")


if __name__ == "__main__":
    main()
