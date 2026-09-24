# Spoiler Shield: one entry point for the whole monorepo.
# Run `make help` to list targets.

ML  := cd ml &&
EXT := cd extension &&

.PHONY: help setup check check-ml check-ext spike-model build-ext e2e bench

help:  ## List targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup:  ## Install Python (with training extras) and extension dependencies
	$(ML) uv sync --extra train
	$(EXT) pnpm install

check: check-ml check-ext  ## Lint, typecheck and unit-test everything

check-ml:  ## Python: ruff, pyright, pytest
	$(ML) uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest

check-ext:  ## Extension: typecheck and unit tests
	$(EXT) pnpm run typecheck && pnpm run test

spike-model:  ## Build the Phase 0 stand-in model into extension/public/models/spike
	$(ML) uv run spoiler-shield spike-model --out ../extension/public/models/spike

build-ext:  ## Build the Chrome extension
	$(EXT) pnpm run build

e2e: spike-model  ## Build, then end-to-end test the real extension in headless Chromium
	$(EXT) pnpm run e2e

bench: spike-model  ## Build, then run the in-browser latency benchmark (writes docs/results/)
	$(EXT) pnpm run bench
