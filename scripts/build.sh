#!/bin/bash

cd report
pandoc --filter pandoc-citeproc report.md -o ../dist/report.pdf -f markdown+implicit_figures && open ../dist/report.pdf