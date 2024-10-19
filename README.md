# promptcachedb

Cache LLM prompts to a persistent database

## Usage

Currently there's a demo that stores prompt caches to disk locally. You
can test this by:

- Cloning this repo
- [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)
- `uv sync`

## Running the server

- `uv run fastapi dev`

## Running the demo

- `uv run --package promptcachedb-client demo`

I'm new to `uv` so not 100% about this, but I think you need to run `uv sync` every time you
switch between running the demo and the server.
