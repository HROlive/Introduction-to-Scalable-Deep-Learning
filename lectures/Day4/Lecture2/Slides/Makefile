# Define input files
LATEXFILE=Lecture_Slides.tex
PANDOCTEX=Slides.tex
PANDOCMD=Slides.md

default: slides

clean:
	latexmk -C

slides:
	pandoc \
		--slide-level 2 \
		--from markdown+table_captions+escaped_line_breaks+tex_math_double_backslash+fenced_code_blocks \
		--to beamer \
		--listings \
		--output $(PANDOCTEX) \
		$(PANDOCMD)
		latexmk -pdflatex="pdflatex --shell-escape %O %S" -pdf $(LATEXFILE)
	#	latexmk -pdflatex="xelatex --shell-escape %O %S" -pdf $(LATEXFILE)

pandoc:
	pandoc \
		--slide-level 2 \
		--from markdown+table_captions+escaped_line_breaks+tex_math_double_backslash+fenced_code_blocks \
		--to beamer \
		--listings \
		--output $(PANDOCTEX) \
		$(PANDOCMD)

tex:
	# latexmk -pdflatex="pdflatex --shell-escape %O %S" -pdf $(LATEXFILE)

#
