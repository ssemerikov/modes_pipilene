#!/bin/sh
# usage: ocr_page.sh <pdf> <outdir> <page>
# One page: render at 300 dpi grey, recognise with Tesseract (English), keep only the text.
pdf="$1"; out="$2"; p="$3"
n=$(printf '%03d' "$p")
[ -s "$out/p$n.txt" ] && exit 0
pdftoppm -f "$p" -l "$p" -r 300 -gray -singlefile "$pdf" "$out/p$n" && \
tesseract "$out/p$n.pgm" "$out/p$n" -l eng --psm 3 >/dev/null 2>&1
rm -f "$out/p$n.pgm"
