ALUMNOS := $(notdir $(wildcard alumnos/*))
# Los PDF finales se copian a reports/privacidad-datos/ en la raíz del repositorio
SALIDA := ../../../reports/privacidad-datos
PDFS := $(ALUMNOS:%=$(SALIDA)/%.pdf)
# Se puede cambiar si biber del sistema no funciona: make BIBER=/ruta/a/biber
BIBER ?= biber
COMUN := $(wildcard comun/* compartido/*)

.PHONY: all clean $(ALUMNOS)

# make compila todos; make facundo-barbera compila solo ese reporte
all: $(PDFS)

$(ALUMNOS): %: $(SALIDA)/%.pdf

$(SALIDA)/%.pdf: alumnos/%/main.tex alumnos/%/investigacion.tex alumnos/%/references.bib $(COMUN)
	cd alumnos/$* && latexmk -pdf -e '$$biber=q($(BIBER) %O %S)' -interaction=nonstopmode -halt-on-error main.tex
	cp alumnos/$*/main.pdf $@

clean:
	for d in alumnos/*/; do (cd $$d && latexmk -C); done
