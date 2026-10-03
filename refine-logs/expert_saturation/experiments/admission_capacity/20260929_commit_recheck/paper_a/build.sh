#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p tmp/pdfs output/pdf
for pass in 1 2; do
    pdflatex -interaction=batchmode -halt-on-error -output-directory=tmp/pdfs draft.tex
done
if rg -n 'Overfull|undefined references|undefined on input line' tmp/pdfs/draft.log; then
    echo 'Resolve the reported layout or reference issue before delivery.' >&2
    exit 1
fi
cp tmp/pdfs/draft.pdf output/pdf/A_recovery_working_draft_20261001.pdf
