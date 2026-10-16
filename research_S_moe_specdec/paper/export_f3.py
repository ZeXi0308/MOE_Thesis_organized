#!/usr/bin/env python3
"""Export F3 for the paper, removing exactly its duplicate provenance footer.

Run from the project root: python3 -B paper/export_f3.py
The original figure and every non-footer PGF line remain unchanged.
"""
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sys.path.insert(0, str(ROOT / "code"))
from plot_energy_progress import render_pgf

SOURCE = ROOT / "figures/F3F4F6_olmoe_compiled_controller_r01/F3_effective_tpot_92cb309cdb53.tex"
OUTPUT = PAPER / "figures"
NAME = "F3_executed_policies"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source_bytes = SOURCE.read_bytes()
    source_pdf = SOURCE.with_suffix(".pdf")
    original_pdf_sha = sha(source_pdf)
    lines = source_bytes.decode("utf-8").splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines)
               if line.startswith(r"\path ") and "south west" in line and "south east" in line]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one bottom footer path, found {len(matches)}")
    index = matches[0]
    footer = lines[index]
    if "Open dots:" not in footer or not footer.rstrip().endswith(r"(group c1r1.south east);"):
        raise ValueError("The unique bottom path is not the expected F3 provenance footer")
    exported_lines = lines[:index] + lines[index + 1:]
    changes = [line for line in difflib.ndiff(lines, exported_lines) if not line.startswith("  ")]
    if changes != ["- " + footer]:
        raise ValueError("Export must delete only the one footer line")
    tex = "".join(exported_lines)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = render_pgf(tex, OUTPUT, NAME, ["pdf", "png"])
    if (OUTPUT / (NAME + ".tex")).read_bytes() != tex.encode("utf-8"):
        raise ValueError("Rendered PGF differs from the verified single-line deletion")
    pdf = OUTPUT / (NAME + ".pdf")
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    if re.search(r"Open\s+dots\s*:", text):
        raise ValueError("Removed footer text remains extractable from the exported PDF")
    for label in ("Effective TPOT", "Native table", "Native threshold", "Adaptive", "Fixed gamma"):
        if label not in text:
            raise ValueError(f"Expected visible plot label missing from PDF text: {label}")
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    size = re.search(r"Page size:\s*([0-9.]+) x ([0-9.]+) pts", info)
    if not size or not re.search(r"Pages:\s*1\s", info):
        raise ValueError("Expected a one-page PDF with explicit page dimensions")
    width, height = map(float, size.groups())
    if SOURCE.read_bytes() != source_bytes or sha(source_pdf) != original_pdf_sha:
        raise ValueError("Original figure changed during export")
    manifest = {"status": "exported_footer_removed", "source_tex": str(SOURCE),
        "source_tex_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_pdf": str(source_pdf), "source_pdf_sha256": original_pdf_sha,
        "removed_source_line_number": index + 1, "removed_line": footer.rstrip("\n"),
        "deleted_lines": 1, "added_lines": 0, "all_other_tex_lines_unchanged": True,
        "export_tex_sha256": sha(OUTPUT / (NAME + ".tex")), "export_pdf_sha256": sha(pdf),
        "script_sha256": sha(Path(__file__)), "page_count": 1,
        "page_size_pt": [width, height], "page_size_cm": [width * 2.54 / 72, height * 2.54 / 72],
        "pdf_text_contains_open_dots": False, "required_plot_labels_present": True,
        "outputs": files,
        "boundary": "Deletes only the duplicate provenance footer; no numerical data, plot commands, title, legend or axis is changed. Original complete figure is retained."}
    (OUTPUT / (NAME + "_export.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf), "deleted_source_line": index + 1,
        "page_size_pt": [width, height], "page_size_cm": manifest["page_size_cm"],
        "pdf_text_contains_open_dots": False}))


if __name__ == "__main__":
    main()
