.PHONY: setup test lint demo-data demo demo-yahoo status image cluster-up cluster-demo cluster-down

WORKERS ?= 2

setup:            ## one-command dev environment (uv installs Python + locked deps)
	uv sync --extra ray

test:
	uv run pytest

lint:
	uv run ruff check src tests

demo-data:
	uv run quantlab data synth --store data/prices

demo: demo-data   ## offline: synthetic data → pairs strategy → sweep → validate → register
	uv run quantlab run jobs/demo.yaml

demo-yahoo:       ## free Yahoo Finance data: built-in momentum + trend, and a user plugin (strategies/low_vol.py)
	uv run quantlab run jobs/momentum_yahoo.yaml
	uv run quantlab run jobs/trend_yahoo.yaml
	uv run quantlab run jobs/low_vol_yahoo.yaml

status:
	uv run quantlab status

image:
	docker build --build-arg CODE_VERSION=$$(git rev-parse --short HEAD) -t quantlab:latest .

cluster-up: image
	docker compose up -d --scale ray-worker=$(WORKERS) ray-head ray-worker

cluster-demo:
	docker compose run --rm quantlab data synth --store /workspace/data/prices
	docker compose run --rm quantlab run jobs/demo.yaml --backend ray

cluster-down:
	docker compose down
