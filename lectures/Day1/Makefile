# The idea is to convert every Markdown file here into a HTML presentation using reveal.js

SOURCE_DOCS := $(wildcard *.md)

EXPORTED_DOCS=\
  $(addprefix public/,$(SOURCE_DOCS:.md=.html))

RM=/bin/rm

PANDOC=pandoc

PANDOC_OPTIONS=-t revealjs -s \
	-V revealjs-url=. \
	--include-in-header=slides.css \
	-V hlss=zenburn \
	-V theme=sky \
	-V transition=fade  \
	-i 
# --embed-resources   # This make a single file, good for distribution
#	-A footer.html # The footer is just too big

public/%.html : %.md *.css
	$(PANDOC) $(PANDOC_OPTIONS) -o $@ $<


.PHONY: all clean

# I am not proud of this. One day has a different naming scheme and it messes up for everyone. Plus, make on MacOS is different.
PDF_DESTINATIONS = \
	public/day2_lecture1.pdf \
	public/day2_lecture2.pdf \
	public/day3_lecture1.pdf \
	public/day3_lecture2.pdf \
	public/day4_lecture1.pdf \
	public/day4_lecture2.pdf \
	public/day5_lecture1.pdf \
	public/day5_lecture1.pdf \
	public/day5_lecture2.pdf

all : $(PDF_DESTINATIONS) $(EXPORTED_DOCS)

# REALLY not proud of this.
public/day2_lecture1.pdf: ../Day2/Lecture1/Slides/Lecture_slides.pdf
	cp $< $@

public/day2_lecture2.pdf: ../Day2/Lecture2/Slides/Lecture_slides.pdf
	cp $< $@

public/day3_lecture1.pdf: ../Day3/Lecture1/Slides/Lecture_Slides.pdf
	cp $< $@

public/day3_lecture2.pdf: ../Day3/Lecture2/Slides/slides.pdf
	cp $< $@

public/day4_lecture1.pdf: ../Day4/Lecture1/Slides/Lecture_Slides.pdf
	cp $< $@

public/day4_lecture2.pdf: ../Day4/Lecture2/Slides/Lecture_Slides.pdf
	cp $< $@

public/day5_lecture1.pdf: ../Day5/Lecture1.pdf
	cp $< $@

public/day5_lecture2.pdf: ../Day5/Lecture2.pdf
	cp $< $@


clean:
	- $(RM) $(EXPORTED_DOCS) $(PDF_DESTINATIONS)