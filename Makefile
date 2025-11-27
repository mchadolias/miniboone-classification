#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = miniboone-classification
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
TEST_DIR = tests
REPORT_DIR = test_reports

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: requirements clean lint format test test-dev test-html test-cov test-full
.PHONY: test-smoke test-unit test-integration open-report open-cov create_environment help
.PHONY: test-cov-ci test-ci lint-ci

## Install Python dependencies
requirements:
	uv sync

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(REPORT_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

## Lint using flake8, black, and isort (use `make format` to do formatting)
lint:
	flake8 src
	isort --check --diff src
	black --check src

## Format source code with black
format:
	isort src
	black src

## Run tests for CI (uses uv run)
test-ci:
	uv run pytest $(TEST_DIR) -v

## Run linting for CI (uses uv run)  
lint-ci:
	uv run flake8 src
	uv run isort --check --diff src
	uv run black --check src

## Run tests with coverage for CI
test-cov-ci:
	mkdir -p $(REPORT_DIR)
	uv run pytest $(TEST_DIR) --cov=src --cov-report=term-missing --cov-report=html:$(REPORT_DIR)/coverage
	@echo "📈 Coverage report: $(REPORT_DIR)/coverage/index.html"

## Run tests
test:
	python -m pytest $(TEST_DIR)

## Run tests with verbose output and stop on first failure
test-dev:
	pytest $(TEST_DIR) -v --tb=short -x

## Run tests with HTML report (no coverage)
test-html:
	mkdir -p $(REPORT_DIR)
	pytest $(TEST_DIR) --html=$(REPORT_DIR)/report.html --self-contained-html
	@echo "📊 HTML report generated: $(REPORT_DIR)/report.html"

## Run tests with coverage report
test-cov:
	mkdir -p $(REPORT_DIR)
	pytest $(TEST_DIR) --cov=src --cov-report=term-missing --cov-report=html:$(REPORT_DIR)/coverage
	@echo "📈 Coverage report: $(REPORT_DIR)/coverage/index.html"

## Run tests with both HTML and coverage reports
test-full:
	mkdir -p $(REPORT_DIR)
	pytest $(TEST_DIR) --cov=src --cov-report=term-missing --cov-report=html:$(REPORT_DIR)/coverage --html=$(REPORT_DIR)/report.html --self-contained-html
	@echo "📊 HTML report: $(REPORT_DIR)/report.html"
	@echo "📈 Coverage report: $(REPORT_DIR)/coverage/index.html"

## Run quick smoke tests only
test-smoke:
	pytest $(TEST_DIR)/smoke/ -v

## Run only unit tests
test-unit:
	pytest $(TEST_DIR)/unit/ -v

## Run only integration tests  
test-integration:
	pytest $(TEST_DIR)/integration/ -v

## Open the latest test report in browser (macOS)
open-report:
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(REPORT_DIR)/report.html; \
	elif command -v open > /dev/null; then \
		open $(REPORT_DIR)/report.html; \
	else \
		echo "Please open manually: $(REPORT_DIR)/report.html"; \
	fi

## Open the latest coverage report in browser (macOS)
open-cov:
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(REPORT_DIR)/coverage/index.html; \
	elif command -v open > /dev/null; then \
		open $(REPORT_DIR)/coverage/index.html; \
	else \
		echo "Please open manually: $(REPORT_DIR)/coverage/index.html"; \
	fi

## Set up Python interpreter environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\.venv\Scripts\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)