# APGI Validation Framework Makefile
# ==================================

.PHONY: help clean test test-fast test-full lint install dev-install venv docs protocols-check

help:
	@echo "Available targets:"
	@echo "  venv         - Create virtual environment"
	@echo "  clean        - Remove temporary files and cache directories"
	@echo "  test         - Run test suite"
	@echo "  test-fast    - Run fast test suite (< 5 minutes)"
	@echo "  test-full    - Run full test suite with 100% coverage"
	@echo "  lint         - Run linting checks"
	@echo "  install      - Install package and dependencies"
	@echo "  dev-install  - Install in development mode"
	@echo "  docs         - Generate documentation"
	@echo "  protocols-check - Validate protocol specification 1.1.0 (protocols/)"

venv:
	python3 -m venv .venv

clean:
	@echo "Cleaning temporary files and cache directories..."
	python3 utils/delete_pycache.py --yes

test:
	python3 -m pytest

test-fast:
	@echo "Running fast test suite (< 5 minutes)..."
	python3 -m pytest -c pytest-fast.ini --ignore=tests/comprehensive

test-full:
	@echo "Running full test suite with 100% coverage..."
	python3 -m pytest tests/ --ignore=tests/comprehensive \
		--cov=. --cov-report=html --cov-report=xml --cov-report=term-missing \
		--cov-fail-under=100

protocols-check:
	python3 scripts/validate_protocols.py

lint:
	@echo "Running black formatting (skipping files with syntax errors)..."
	@python3 -m black . 2>/dev/null || true
	@echo "Running flake8 for syntax errors..."
	@python3 -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics 
	@echo "Running flake8 for style issues..."
	@python3 -m flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "Running mypy..."
	@python3 -m mypy . || true

install: venv
	.venv/bin/pip install .

dev-install: venv
	.venv/bin/pip install -e .

docs:
	@echo "Generating documentation..."
	@echo "Converting Markdown to HTML..."
	@command -v pandoc >/dev/null 2>&1 || { echo "pandoc not installed. Install with: brew install pandoc (macOS) or apt install pandoc (Ubuntu)"; exit 1; }
	@pandoc docs/APGI_Equations.md -o docs/APGI_Equations.html --metadata title="APGI Equations" --css=github-markdown.css
	@pandoc docs/APGI-Parameter-Specifications.md -o docs/APGI-Parameter-Specifications.html --metadata title="APGI Parameter Specifications" --css=github-markdown.css
	@pandoc docs/APGI-Empirical-Credibility-Roadmap.md -o docs/APGI-Empirical-Credibility-Roadmap.html --metadata title="APGI Empirical Credibility Roadmap" --css=github-markdown.css
	@pandoc docs/APGI-Falsification-Criteria.md -o docs/APGI-Falsification-Criteria.html --metadata title="APGI Falsification Criteria" --css=github-markdown.css
	@pandoc docs/Tutorial.md -o docs/Tutorial.html --metadata title="APGI Tutorial" --css=github-markdown.css
	@pandoc docs/Innovations.md -o docs/Innovations.html --metadata title="APGI Innovations" --css=github-markdown.css
	@pandoc docs/Innovations-Software-Implementation.md -o docs/Innovations-Software-Implementation.html --metadata title="APGI Software Implementation" --css=github-markdown.css
	@pandoc docs/Multimodal-Integration.md -o docs/Multimodal-Integration.html --metadata title="APGI Multimodal Integration" --css=github-markdown.css
	@echo "Documentation generated in docs/ directory"
	@echo "Open HTML files in a web browser to view"
