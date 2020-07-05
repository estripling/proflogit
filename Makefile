# Global parameters
SHELL := /bin/bash
PYPATH := /usr/bin/python3
PYEXE := $(PWD)/venv/bin/python
PGKNAME := proflogit

# Main
.PHONY: all help program
all: program

help: Makefile
	@sed -n 's/^##//p' $<

program:
	@echo "Commands for package management"
	@echo "make help"


## venv ::  Create virtual Python enviroment
.PHONY: venv
venv:
	$(PYPATH) -m venv $(PWD)/venv
	$(PYEXE) -m pip install -r requirements.txt --no-cache-dir


## rmvenv ::  Remove venv/ directory
.PHONY: rmvenv
rmvenv:
	rm -rf venv/


## install ::  Install package
.PHONY: install
install:
	$(PYEXE) -m pip install .


## develop ::  Install development package
.PHONY: develop
develop:
	$(PYEXE) -m pip install -e .


## uninstall ::  Uninstall (development) package
.PHONY: uninstall
uninstall:
	$(PYEXE) -m pip uninstall --yes $(PGKNAME)


## dev-uninstall ::  Uninstall (development) package
.PHONY: dev-uninstall
dev-uninstall:
	$(PYEXE) -m pip uninstall --yes $(PGKNAME)
	rm -r $(PGKNAME).egg-info


## tests ::  Run tests
.PHONY: tests
tests:
	cd tests/ && make tests && cd -


## rmdir ::  Remove __pychache__ directories
.PHONY: rmdir
rmdir:
	find . -name __pycache__ -type d -exec rm -rf {} +


## vars :: Echo variables
.PHONY: vars
vars:
	@echo SHELL: $(SHELL)
	@echo PYPATH: $(PYPATH)
	@echo PYEXE: $(PYEXE)
	@echo PGKNAME: $(PGKNAME)
