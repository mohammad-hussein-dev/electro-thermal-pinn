#!/bin/bash
# ============================================================
# Build Script for Electro-Thermal PINN Technical Report
# ============================================================

set -e  # Exit on error

echo "========================================"
echo "  Building Electro-Thermal PINN PDF"
echo "========================================"

# Check if xelatex is available
if command -v xelatex &> /dev/null; then
    ENGINE="xelatex"
    echo "Using XeLaTeX engine..."
elif command -v pdflatex &> /dev/null; then
    ENGINE="pdflatex"
    echo "Using pdfLaTeX engine..."
else
    echo "ERROR: No LaTeX engine found. Please install TeX Live."
    exit 1
fi

# Build the document (run twice for TOC)
echo "Compiling LaTeX document (pass 1/2)..."
$ENGINE -interaction=nonstopmode -halt-on-error Electro_Thermal_PINN_Technical_Report.tex

echo "Compiling LaTeX document (pass 2/2)..."
$ENGINE -interaction=nonstopmode -halt-on-error Electro_Thermal_PINN_Technical_Report.tex

echo ""
echo "========================================"
echo "  PDF generated successfully!"
echo "  Output: Electro_Thermal_PINN_Technical_Report.pdf"
echo "========================================"
