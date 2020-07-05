# Global parameters
SHELL := /bin/bash
PYEXE := python3

# Function to run Python in virtual environment,
# needed for testing CLI functions.
define pyvenv
source ../venv/bin/activate && $(PYEXE) -m unittest ${1} && deactivate;
endef

# Main
.PHONY: all help program
all: program

help: Makefile
	@sed -n 's/^##//p' $<

program:
	@echo "Commands for testing package"
	@echo "make help"


## tests ::  Run all tests
.PHONY: tests
tests: test-empc test-rga test-proflogit


## test-empc ::  Run EMPC tests
.PHONY: test-empc
test-empc:
	$(call pyvenv, test_empc)


## test-rga ::  Run RGA tests
.PHONY: test-rga
test-rga:
	$(call pyvenv, test_rga)


## test-proflogit ::  Run ProfLogit tests
.PHONY: test-proflogit
test-proflogit:
	$(call pyvenv, test_proflogit)
