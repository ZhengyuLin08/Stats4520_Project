# Makefile for compiling LaTeX IEEE paper

# Main tex file (without .tex extension)
MAIN = paper

# Compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Default target
all: $(MAIN).pdf

# Compile the PDF (with bibliography)
$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Quick compile (no bibliography update)
quick: $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot

# Clean everything including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# Phony targets
.PHONY: all quick clean cleanall
