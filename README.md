# promptcachedb

Cache LLM prompts to a persistent database

## Usage

Currently there's a demo that stores prompt caches to disk locally. You
can test this by:

- Cloning this repo
- [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)
- `uv sync`

## Running the server

To run locally, do `uv run fastapi dev`.

To run with docker, do `docker compose up`.

## Running the demo

`uv run --package promptcachedb-client demo`

The benchmark can be ran with

`uv run --package promptcachedb-client benchmark`
