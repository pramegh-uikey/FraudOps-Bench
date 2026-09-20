#!/usr/bin/env bash
# Rebuild the IEEE Access submission source archive from paper/latex/.
#
# Run this after ANY edit to fraudops_bench.tex or its images, otherwise
# fraudops_bench_latex_source.zip silently goes stale against the PDF you
# upload -- and IEEE requires the source and the PDF to match exactly.
#
# The three files to upload are then:
#   paper/latex/fraudops_bench.pdf        the manuscript
#   paper/fraudops_bench_latex_source.zip the LaTeX source
#   paper/supplementary_holdout_v2.zip    per-arm run outputs
set -euo pipefail

cd "$(dirname "$0")"
LATEX_DIR=latex
OUT=fraudops_bench_latex_source.zip

# Logo.png, notaglineLogo.png and bullet.png are loaded by ieeeaccess.cls,
# not by the .tex -- easy to miss, and the build fails without them.
FILES=(
  fraudops_bench.tex
  ieeeaccess.cls
  images/risk_coverage_curves.pdf
  images/ayushi_ambilkar.png
  images/pramegh_uikey.png
  images/Logo.png
  images/notaglineLogo.png
  images/bullet.png
)

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
for f in "${FILES[@]}"; do
  [ -f "$LATEX_DIR/$f" ] || { echo "missing: $LATEX_DIR/$f" >&2; exit 1; }
  mkdir -p "$STAGE/$(dirname "$f")"
  cp "$LATEX_DIR/$f" "$STAGE/$f"
done

rm -f "$OUT"
( cd "$STAGE" && zip -qr "$OLDPWD/$OUT" . )
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"

# Prove the archive is self-sufficient: compile it in an empty directory.
if command -v tectonic >/dev/null 2>&1; then
  V=$(mktemp -d); trap 'rm -rf "$STAGE" "$V"' EXIT
  unzip -q "$OUT" -d "$V"
  ( cd "$V" && tectonic -X compile fraudops_bench.tex --outdir "$V" >/dev/null 2>&1 ) \
    && echo "verified: compiles standalone ($(pdfinfo "$V/fraudops_bench.pdf" | awk '/Pages/{print $2}') pages)" \
    || { echo "FAILED to compile from the archive -- a file is missing" >&2; exit 1; }
else
  echo "note: tectonic not found, skipped the standalone-compile check" >&2
fi
