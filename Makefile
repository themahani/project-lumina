SHELL = /usr/bin/env bash

# --- OS-Agnostic Python Interpreter ---
# Try to find python3, otherwise fall back to python.
PYTHON_CMD ?= $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
ifeq ($(PYTHON_CMD),)
    $(error "Could not find 'python3' or 'python' in your PATH. Please install Python.")
endif

.PHONY: help train prepdata installdeps mlflow_serve clean

REQS_FILE = ./requirements.txt
VENV_DIR = ./venv

PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
MLFLOW = $(VENV_DIR)/bin/mlflow

# The stamp file to indicate a successful dependency installation
VENV_STAMP = $(VENV_DIR)/.venv_installed

help:
	@echo "Using Python interpreter: $(PYTHON_CMD)"
	@echo "---"
	@echo "installdeps:"
	@echo "	Install project dependencies into a virtual environment."
	@echo "prepdata:"
	@echo "	Download and preprocess data."
	@echo "train:"
	@echo "	Train all the models."
	@echo "mlflow_serve:"
	@echo "	Serve the MLflow model."
	@echo "clean:"
	@echo "	Remove the virtual environment and temporary files."

$(VENV_STAMP): $(REQS_FILE)
	test -d $(VENV_DIR) || $(PYTHON_CMD) -m venv $(VENV_DIR)
	@echo "Installing dependencies from $(REQS_FILE)..."
	$(PIP) install -r $(REQS_FILE)
	# Create the stamp file to mark installation as complete.
	touch $(VENV_STAMP)

# A convenience target to allow forcing re-installation.
installdeps:
	$(MAKE) clean
	$(MAKE) $(VENV_STAMP)

prepdata: $(VENV_STAMP)
	$(PYTHON) ./download_data.py
	$(PYTHON) ./src/build_graph.py

mlflow_serve: $(VENV_STAMP)
	$(MLFLOW) server &

train: prepdata mlflow_serve
	$(PYTHON) ./src/train.py

clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_DIR)
