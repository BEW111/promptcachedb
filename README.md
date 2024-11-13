# promptcachedb

Cache LLM prompts to a persistent database

Currently still in progress, but this may be useful if you check some or especially all of the following boxes:
- [ ] Working with open-source LLMs
- [ ] Re-using very long prompts
- [ ] Running multiple LLMs or agents across different servers

## Usage

In the current demo, a prompt is cached, sent to a server, retrieved from the server, and then used locally. You
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
