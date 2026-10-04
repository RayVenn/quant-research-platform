.PHONY: setup test lint demo-data demo status image cluster-up cluster-demo cluster-down

WORKERS ?= 2

setup:            ## one-command dev environment (uv installs Python + locked deps)
	uv sync --extra ray

test:
	uv run pytest

lint:
	uv run ruff check src tests

demo-data:
	uv run pairlab data synth --store data/prices

demo: demo-data   ## full pipeline on synthetic data: screen → sweep → validate → register
	uv run pairlab run jobs/demo.yaml

status:
	uv run pairlab status

image:
	docker build --build-arg CODE_VERSION=$$(git rev-parse --short HEAD) -t pairlab:latest .

cluster-up: image
	docker compose up -d --scale ray-worker=$(WORKERS) ray-head ray-worker

cluster-demo:
	docker compose run --rm pairlab data synth --store /workspace/data/prices
	docker compose run --rm pairlab run jobs/demo.yaml --backend ray

cluster-down:
	docker compose down
