# IEEE Format LaTeX Paper - Satellite Anomaly Detection

This directory contains the IEEE format LaTeX paper for the Stats4520 satellite anomaly detection project.

## Files

- `paper.tex` - Main LaTeX document in IEEE conference format
- `references.bib` - BibTeX bibliography file with all references
- `Makefile` - Makefile for easy compilation
- `LATEX_README.md` - This file

## Requirements

To compile the LaTeX document, you need:
- A LaTeX distribution (e.g., TeX Live, MiKTeX, or MacTeX)
- `pdflatex` compiler
- `bibtex` for bibliography processing
- `IEEEtran` document class (usually included in LaTeX distributions)

## Compilation Instructions

### Using Make (Recommended)

```bash
# Compile the full paper with bibliography
make

# Quick compile (without updating bibliography)
make quick

# Clean auxiliary files
make clean

# Clean everything including PDF
make cleanall
```

### Manual Compilation

If you prefer to compile manually or don't have Make:

```bash
# First pass
pdflatex paper.tex

# Process bibliography
bibtex paper

# Second pass (resolve citations)
pdflatex paper.tex

# Third pass (resolve cross-references)
pdflatex paper.tex
```

### Using Overleaf

You can also use Overleaf (online LaTeX editor):
1. Create a new project on Overleaf
2. Upload `paper.tex` and `references.bib`
3. Set the compiler to pdfLaTeX
4. Compile the document

## Document Structure

The paper includes the following sections:

1. **Title and Abstract** - Overview of the project
2. **Introduction** - Problem motivation and objectives
3. **Related Work** - Background on anomaly detection methods
4. **Methodology** - Detailed description of:
   - Data preprocessing pipeline
   - LSTM-based classification approach
   - ARIMA-based outlier detection
5. **Experimental Setup** - Datasets and evaluation metrics
6. **Results** - Performance analysis and comparisons
7. **Discussion** - Practical implications and limitations
8. **Conclusion** - Summary and future work
9. **References** - Bibliography

## IEEE Format

The document uses the `IEEEtran` document class in conference mode, which provides:
- Standard IEEE two-column layout
- Proper formatting for titles, authors, and sections
- IEEE-style citations and bibliography
- Professional typesetting for equations and algorithms

## Customization

To customize the paper:

1. **Author Information**: Edit the `\author` block in `paper.tex`
2. **Title**: Modify the `\title` command
3. **Abstract**: Update the content in the `\begin{abstract}...\end{abstract}` block
4. **Sections**: Add or modify sections as needed
5. **References**: Add new entries to `references.bib` in BibTeX format

## Output

The compilation produces `paper.pdf`, a professionally formatted IEEE conference paper ready for submission or presentation.

## Troubleshooting

**Missing IEEEtran class**: Install the IEEE template package:
```bash
# On Ubuntu/Debian
sudo apt-get install texlive-publishers

# On macOS with MacTeX, it should be included
```

**Bibliography not showing**: Make sure to run the full compilation sequence (pdflatex → bibtex → pdflatex → pdflatex)

**Compilation errors**: Check the `.log` file for detailed error messages

## License

This LaTeX document is part of the Stats4520 project and follows the same license as the repository.
