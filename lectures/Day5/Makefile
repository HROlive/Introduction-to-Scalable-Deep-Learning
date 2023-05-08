all: Lecture1 Lecture2
Lecture1: Lecture1.md Lecture1.tex
	pandoc --slide-level 1 \
		--filter pandoc-citeproc \
		--bibliography=references.bib \
		--from markdown+table_captions+escaped_line_breaks+tex_math_double_backslash+fenced_code_blocks \
		--to beamer \
		--listings \
		--output Lecture1_content.tex \
		Lecture1.md
	latexmk -pdflatex="pdflatex --shell-escape %O %S" -pdf Lecture1.tex

Lecture2: Lecture2.md Lecture2.tex
	pandoc --slide-level 1 \
		--filter pandoc-citeproc \
		--bibliography=references.bib \
		--from markdown+table_captions+escaped_line_breaks+tex_math_double_backslash+fenced_code_blocks \
		--to beamer \
		--listings \
		--output Lecture2_content.tex \
		Lecture2.md
	latexmk -pdflatex="pdflatex --shell-escape %O %S" -pdf Lecture2.tex
clean:
	rm -f Lecture1.pdf Lecture2.pdf
