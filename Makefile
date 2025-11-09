# Default paths
VENV ?= .venv
PYTHON ?= python3
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install run ingest clean-store

help:
	@echo "Targets:"
	@echo "  make install             # create venv and install dependencies"
	@echo "  make run ARGS='...'      # run main.py with optional CLI args"
	@echo "  make ingest SOURCE=path  # ingest docs before quitting"
	@echo "  make clean-store         # wipe the local Chroma store"

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

install: venv
	@$(PIP) install -U pip
	@$(PIP) install -r requirements.txt

run: install
	@$(PYTHON_BIN) main.py $(ARGS)

ingest: install
ifndef SOURCE
	$(error SOURCE is required, e.g. make ingest SOURCE=docs RESET=1)
endif
	@$(PYTHON_BIN) main.py --source $(SOURCE) $(if $(RESET),--reset,) $(if $(filter 0,$(RECURSIVE)),--no-recursive,) $(ARGS)

clean-store: install
	@$(PYTHON_BIN) -c "from utils.embeddings import reset_store, store_size; reset_store(); print(f'Cleared store. Remaining vectors: {store_size()}')"
