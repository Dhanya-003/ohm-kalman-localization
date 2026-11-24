# Build the handout: make (latexmk -pdf). TikZ figures in figures/, no external graphics.
LATEXMK := latexmk

kalman.pdf: kalman.tex $(wildcard figures/*.tikz figures/*.txt)
	$(LATEXMK) -pdf -interaction=nonstopmode kalman.tex

# Solution repository only: the handout with the solutions of docs/solutions/ (kalman.tex reads them only here)
solution: kalman_solution.pdf

kalman_solution.pdf: kalman.tex $(wildcard solutions/*.tex figures/*.tikz figures/*.txt)
	$(LATEXMK) -pdf -interaction=nonstopmode -jobname=kalman_solution -usepretex='\def\withsolution{}' kalman.tex

.PHONY: log clean solution

log:
	@grep -nE "^!|Overfull \\\\hbox \(([1-9][0-9]|[0-9]+[0-9])pt|Undefined control|LaTeX Warning: (Reference|Citation)" kalman.log || echo "log clean"

clean:
	-$(LATEXMK) -c kalman.tex >/dev/null 2>&1
	-$(LATEXMK) -c -jobname=kalman_solution kalman.tex >/dev/null 2>&1
