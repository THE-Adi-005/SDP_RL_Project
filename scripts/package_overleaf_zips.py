"""
Packaging script for CascadeRoute-VL standalone Overleaf submission projects.
Creates flattened, self-contained, lightweight ZIP archives for Main, Supp, and Combined projects,
including .latexmkrc configuration and default main.tex entry points for instant Overleaf free-tier compilation.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CVPR_DIR = REPO_ROOT / "cvpr2027"
FIG_DIR = REPO_ROOT / "reports" / "figures"
QUAL_DIR = FIG_DIR / "qualitative_raw"

MAIN_TEX = CVPR_DIR / "CascadeRoute_VL_CVPR2027.tex"
SUPP_TEX = CVPR_DIR / "Supplementary.tex"
BIB_FILE = CVPR_DIR / "references.bib"
BST_FILE = CVPR_DIR / "ieeenat_fullname.bst"
STY_FILE = CVPR_DIR / "cvpr.sty"
LATEXMKRC_FILE = CVPR_DIR / ".latexmkrc"

MAIN_ZIP = CVPR_DIR / "CascadeRoute_VL_Main_Overleaf.zip"
SUPP_ZIP = CVPR_DIR / "CascadeRoute_VL_Supp_Overleaf.zip"
COMB_ZIP = CVPR_DIR / "CascadeRoute_VL_Overleaf.zip"

def flatten_tex_main(content: str) -> str:
    return re.sub(r"\.\./reports/figures/", "", content)

def flatten_tex_supp(content: str) -> str:
    return re.sub(r"\.\./reports/figures/qualitative_raw/", "", content)

def build_zips():
    print("[1/3] Preparing flattened and optimized source files...")
    main_tex_str = flatten_tex_main(MAIN_TEX.read_text(encoding="utf-8"))
    supp_tex_str = flatten_tex_supp(SUPP_TEX.read_text(encoding="utf-8"))

    main_figs = [
        "fig1_reasoning_tax_teaser.png",
        "fig2_architecture.png",
        "fig3_pareto_frontier.png",
        "fig4_ood_shift_adaptation.png",
        "fig5_ablations_and_tokens.png",
        "fig6_budgets_and_gates.png",
        "fig7_live_configs_and_drift.png",
    ]

    supp_figs = [
        "tax_inversion_OCRBench_314.png",
        "tax_inversion_OCRBench_979.png",
        "reasoning_gain_MathVista_MINI_897.png",
        "reasoning_gain_MathVista_MINI_874.png",
    ]

    for f in main_figs:
        p = FIG_DIR / f
        assert p.exists(), f"Missing figure: {p}"
    for f in supp_figs:
        p = QUAL_DIR / f
        assert p.exists(), f"Missing qualitative figure: {p}"

    print("[2/3] Writing optimized ZIP archives (with .latexmkrc and main.tex)...")
    
    # 1. Main ZIP
    with zipfile.ZipFile(MAIN_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", main_tex_str)
        zf.writestr("CascadeRoute_VL_CVPR2027.tex", main_tex_str)
        zf.write(BIB_FILE, "references.bib")
        zf.write(BST_FILE, "ieeenat_fullname.bst")
        zf.write(STY_FILE, "cvpr.sty")
        if LATEXMKRC_FILE.exists():
            zf.write(LATEXMKRC_FILE, ".latexmkrc")
        for f in main_figs:
            zf.write(FIG_DIR / f, f)
    print(f"  Created: {MAIN_ZIP} ({MAIN_ZIP.stat().st_size / 1024:.1f} KB)")

    # 2. Supp ZIP
    with zipfile.ZipFile(SUPP_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", supp_tex_str)
        zf.writestr("Supplementary.tex", supp_tex_str)
        zf.write(BIB_FILE, "references.bib")
        zf.write(BST_FILE, "ieeenat_fullname.bst")
        zf.write(STY_FILE, "cvpr.sty")
        if LATEXMKRC_FILE.exists():
            zf.write(LATEXMKRC_FILE, ".latexmkrc")
        for f in supp_figs:
            zf.write(QUAL_DIR / f, f)
    print(f"  Created: {SUPP_ZIP} ({SUPP_ZIP.stat().st_size / 1024:.1f} KB)")

    # 3. Combined ZIP
    with zipfile.ZipFile(COMB_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", main_tex_str)
        zf.writestr("CascadeRoute_VL_CVPR2027.tex", main_tex_str)
        zf.writestr("Supplementary.tex", supp_tex_str)
        zf.write(BIB_FILE, "references.bib")
        zf.write(BST_FILE, "ieeenat_fullname.bst")
        zf.write(STY_FILE, "cvpr.sty")
        if LATEXMKRC_FILE.exists():
            zf.write(LATEXMKRC_FILE, ".latexmkrc")
        for f in main_figs:
            zf.write(FIG_DIR / f, f)
        for f in supp_figs:
            zf.write(QUAL_DIR / f, f)
    print(f"  Created: {COMB_ZIP} ({COMB_ZIP.stat().st_size / 1024:.1f} KB)")

def verify_zip_compilation():
    print("[3/3] Verifying compilation speed and layout of ZIPs in fresh temporary directories...")
    tectonic_exe = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "tectonic" / "tectonic.exe"
    assert tectonic_exe.exists(), f"Tectonic not found at {tectonic_exe}"

    import fitz

    # Test Main ZIP (testing main.tex entry point)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(MAIN_ZIP, "r") as zf:
            zf.extractall(tmp_path)
        
        print("  Compiling extracted Main ZIP (main.tex) with Tectonic...")
        t0 = time.time()
        res = subprocess.run(
            [str(tectonic_exe), "main.tex"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True
        )
        elapsed = time.time() - t0
        assert res.returncode == 0, f"Main ZIP compilation failed:\n{res.stderr}\n{res.stdout}"
        
        pdf_path = tmp_path / "main.pdf"
        assert pdf_path.exists(), "Main PDF not generated from ZIP"
        with fitz.open(pdf_path) as doc:
            print(f"  Main ZIP PDF verified: {len(doc)} pages in {elapsed:.2f}s.")
            assert len(doc) == 10, f"Expected 10 pages for Main, got {len(doc)}"
            for i, page in enumerate(doc):
                txt = page.get_text()
                assert "??" not in txt, f"Undefined reference (??) on page {i+1} of Main ZIP"
                if i == 7:
                    assert "Conclusion" in txt or "Limitations" in txt, "Conclusion/Limitations not ending on Page 8"
                if i == 8:
                    assert "References" in txt, "References not starting on Page 9"
        print(f"  Main ZIP compilation: 100% PASS ({elapsed:.2f}s, 10 pages, body ends p.8, refs start p.9, 0 ??)")

    # Test Supp ZIP (testing main.tex entry point)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(SUPP_ZIP, "r") as zf:
            zf.extractall(tmp_path)
        
        print("  Compiling extracted Supp ZIP (main.tex) with Tectonic...")
        t0 = time.time()
        res = subprocess.run(
            [str(tectonic_exe), "main.tex"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True
        )
        elapsed = time.time() - t0
        assert res.returncode == 0, f"Supp ZIP compilation failed:\n{res.stderr}\n{res.stdout}"
        
        pdf_path = tmp_path / "main.pdf"
        assert pdf_path.exists(), "Supp PDF not generated from ZIP"
        with fitz.open(pdf_path) as doc:
            print(f"  Supp ZIP PDF verified: {len(doc)} pages in {elapsed:.2f}s.")
            assert len(doc) == 8, f"Expected 8 pages for Supp, got {len(doc)}"
            for i, page in enumerate(doc):
                txt = page.get_text()
                assert "??" not in txt, f"Undefined reference (??) on page {i+1} of Supp ZIP"
        print(f"  Supp ZIP compilation: 100% PASS ({elapsed:.2f}s, 8 pages, 0 ??)")

if __name__ == "__main__":
    build_zips()
    verify_zip_compilation()
    print("\nALL OVERLEAF ZIP PACKAGING AND COMPILATION CHECKS PASSED SUCCESSFULLY!")
