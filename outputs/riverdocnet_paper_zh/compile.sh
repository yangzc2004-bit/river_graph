#!/bin/sh
# Build the editable manuscript with the user's local TeX Live.
set -eu
cd "$(dirname "$0")"
riverdocnet_texbin="${RIVERDOCNET_TEXBIN:-/Users/yzccc/local_texlive/texlive/2026/bin/universal-darwin}"
export PATH="$riverdocnet_texbin:$PATH"
mkdir -p latex_build
for riverdocnet_pass in 1 2; do
    xelatex -interaction=nonstopmode -halt-on-error -file-line-error \
        -output-directory=latex_build riverdocnet_mississippi.tex \
        > "latex_build/pass_${riverdocnet_pass}.txt" 2>&1
done
cp latex_build/riverdocnet_mississippi.pdf riverdocnet_mississippi.pdf
