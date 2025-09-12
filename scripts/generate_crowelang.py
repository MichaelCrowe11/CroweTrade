"""CroweLang → Python generation shim.

Invoked in CI or locally to regenerate Python strategy code from
`.crowe` sources placed under `crowelang_integration/`.

This is a placeholder; it assumes the standalone `crowe-lang` repo
is cloned as a sibling directory and that `npm install && npm run build`
has been executed there so the compiler CLI is available.
"""
from __future__ import annotations

import subprocess, sys, pathlib, json, shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "crowelang_integration"
OUT_DIR = ROOT / "src" / "crowetrade" / "strategies" / "generated"
COMPILER_REPO = ROOT.parent / "crowe-lang"

def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def compile_file(crowe_path: pathlib.Path) -> None:
    rel = crowe_path.relative_to(SRC_DIR)
    stem = rel.stem
    out_file = OUT_DIR / f"{stem}.py"
    cmd = [
        "node",
        str(COMPILER_REPO / "packages" / "crowe-compiler" / "dist" / "cli.cjs"),
        "--target", "python",
        str(crowe_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[crowelang] ERROR compiling {crowe_path}: {e}\n{getattr(e, 'stderr', '')}", file=sys.stderr)
        raise SystemExit(1)
    out_file.write_text(result.stdout, encoding="utf-8")
    print(f"[crowelang] wrote {out_file.relative_to(ROOT)}")

def main():
    if not COMPILER_REPO.exists():
        print("Compiler repo 'crowe-lang' not found adjacent to CroweTrade root.", file=sys.stderr)
        sys.exit(1)
    ensure_out_dir()
    sources = list(SRC_DIR.glob("*.crowe"))
    if not sources:
        print("No .crowe sources found.")
        return
    for f in sources:
        compile_file(f)

if __name__ == "__main__":
    main()
