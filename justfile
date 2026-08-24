run := "configs/run.yaml"

defaults:
	@just --list

test:
	uv run pytest

lint:
	uv run pre-commit run --all-files
