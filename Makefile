#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = miniboone-classification
PYTHON_VERSION = 3.14
PYTHON_INTERPRETER = python
TEST_DIR = tests
REPORT_DIR = test_reports

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: clean lint lint-fix format format-ci lint-ci lint-ci-nonblocking \
        test test-ci test-unit test-integration test-dev test-cov test-cov-ci \
        test-html test-full test-smoke open-report open-cov requirements \
		create_environment help

## Install Python dependencies
requirements:
	uv sync

## Remove cached files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf $(REPORT_DIR) .pytest_cache .coverage htmlcov

## Lint check (flake8 + isort + black)
lint:
	uv run flake8 src
	uv run isort --check --diff src
	uv run black --check src

## Auto-fix linting (recommended locally)
lint-fix:
	uv run autoflake -r --remove-all-unused-imports --in-place src
	uv run isort src
	uv run black src

## Format code automatically
format:
	uv run autoflake -r --remove-unused-variables --remove-all-unused-imports --in-place src
	uv run isort src
	uv run black src

## CI auto-format (quiet output)
format-ci:
	uv run autoflake -r --remove-all-unused-imports --in-place src
	uv run isort src
	uv run black src

## Strict CI lint
lint-ci:
	uv run flake8 src
	uv run isort --check --diff src
	uv run black --check src

## Lint for feature branches (non-blocking)
lint-ci-nonblocking:
	-uv run flake8 src || echo "⚠️  flake8 issues (non-blocking)"
	-uv run isort --check --diff src || echo "⚠️  isort issues (non-blocking)"
	-uv run black --check src || echo "⚠️  black issues (non-blocking)"

## Run all tests
test:
	uv run pytest $(TEST_DIR)

## Unit tests only
test-unit:
	uv run pytest $(TEST_DIR)/unit -v

## Integration tests only
test-integration:
	uv run pytest $(TEST_DIR)/integration -v

## Development tests
test-dev:
	uv run pytest $(TEST_DIR) -v --tb=short -x

## Coverage CI
test-cov-ci:
	mkdir -p $(REPORT_DIR)
	uv run pytest $(TEST_DIR) \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:$(REPORT_DIR)/coverage \
		--cov-report=xml:$(REPORT_DIR)/coverage.xml
	@echo "📈 Coverage report written to $(REPORT_DIR)/coverage/"

## Quick smoke tests
test-smoke:
	uv run pytest $(TEST_DIR)/smoke -v

## HTML-only report
test-html:
	mkdir -p $(REPORT_DIR)
	uv run pytest $(TEST_DIR) --html=$(REPORT_DIR)/report.html --self-contained-html

## Full test suite
test-full:
	mkdir -p $(REPORT_DIR)
	uv run pytest $(TEST_DIR) \
		--cov=src \
		--cov-report=html:$(REPORT_DIR)/coverage \
		--html=$(REPORT_DIR)/report.html --self-contained-html

## Open coverage report
open-cov:
	xdg-open $(REPORT_DIR)/coverage/index.html || open $(REPORT_DIR)/coverage/index.html

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