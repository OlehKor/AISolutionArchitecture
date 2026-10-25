# Setup and deployment

## Prerequisites

- Python 3.11 to 3.13, or Docker with the Compose plugin.
- An OpenAI API key (or Azure OpenAI settings). Embeddings run locally with FastEmbed.
- For the human approval: a Telegram bot token and the administrator's chat id.

## Configuration

Copy `.env.example` to `.env`. Never commit `.env`.

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` | Chat model credentials | required |
| `PARKING_LLM_PROVIDER` | `openai` or `azure` | `openai` |
| `PARKING_CHAT_MODEL` | Model or Azure deployment name | `gpt-4o-mini` |
| `PARKING_TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | empty (reservations stay pending) |
| `PARKING_TELEGRAM_ADMIN_CHAT_ID` | The only chat allowed to decide | `0` |
| `PARKING_MCP_TOKEN` | Bearer secret, at least 24 characters | empty (approvals wait in the retry queue) |
| `PARKING_MCP_HOST`, `PARKING_MCP_PORT` | MCP server bind address | `127.0.0.1`, `8765` |
| `PARKING_MCP_SERVER_URL` | Override of the MCP URL used by clients | derived from host and port |
| `PARKING_MCP_ALLOWED_HOSTS` | Extra `Host` names the MCP server accepts (comma separated) | empty |
| `PARKING_RESERVATIONS_FILE` | File written by the MCP server | `var/confirmed_reservations.txt` |
| `PARKING_WORKFLOW_DB` | Workflow checkpoint database | `var/workflow.sqlite` |
| `PARKING_SQLITE_PATH` | Reservations and live data | `var/parking.sqlite` |

Generate the MCP token:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Telegram: create a bot with `@BotFather`, set `PARKING_TELEGRAM_BOT_TOKEN`, run `parking-assistant admin` and message the bot. Without a chat id it runs in setup mode, prints your chat id and exits. Set `PARKING_TELEGRAM_ADMIN_CHAT_ID` and start it again.

## Option 1: local, one process

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # fill in the values
parking-assistant run
```

`run` starts the MCP server and the administrator agent in background threads and opens the chat in the foreground. Use it for demos.

## Option 2: local, three processes

```bash
parking-assistant mcp-server    # terminal 1
parking-assistant admin         # terminal 2
parking-assistant chat          # terminal 3
```

The processes share `var/parking.sqlite` and `var/workflow.sqlite`; the administrator can be restarted at any time without losing pending requests.

## Option 3: Docker Compose

```bash
docker compose up -d --build        # MCP server + administrator agent
docker compose run --rm chat        # interactive chat
docker compose logs -f admin
docker compose down                 # keeps the volumes
docker compose down -v              # also deletes reservations and state
```

- `mcp` listens on `127.0.0.1:8765` of the host, has a health check on `/health` and stores the file in the `reservations` volume (`/data/confirmed_reservations.txt`).
- `admin` and `chat` share the `state` volume (`/app/var`) and reach the server at `http://mcp:8765/mcp`; the server accepts the `mcp` host name through `PARKING_MCP_ALLOWED_HOSTS`.
- Secrets come from `.env` at run time and are not part of the image. The image runs as a non-root user and contains the embedding model, so it starts without a download.
- Read the stored reservations: `docker compose exec mcp cat /data/confirmed_reservations.txt`.

On macOS without Docker Desktop, Colima works: `brew install colima docker docker-compose`, `colima start`.

## Operating notes

- Run exactly one administrator agent per Telegram bot; two pollers on the same bot conflict.
- Run one chat process at a time per vector store file (Milvus Lite is single-process).
- Back up the `state` and `reservations` volumes (or the `var/` directory).
- Expose the MCP server beyond the host only behind HTTPS and keep the token secret; rotate it by changing `PARKING_MCP_TOKEN` for server and clients and restarting both.
- Rebuild the vector store with `parking-assistant ingest` after changing `data/static/`.

## Checks

```bash
ruff check . && ruff format --check .
pytest                                   # no API key needed
parking-assistant graph                  # print both LangGraph diagrams (Mermaid)
parking-assistant loadtest               # writes reports/load_test_report.md
parking-assistant evaluate --end-to-end  # writes reports/evaluation_report.md (needs an API key)
```

`loadtest` options: `--sessions`, `--concurrency`, `--mcp-requests`, `--mcp-concurrency`, `--real-llm`. It uses random ports and temporary data, so it does not touch running services or your own data.

## Continuous integration

`.github/workflows/ci.yml` runs ruff, the format check and the test suite on every push to `main` and every pull request, and builds the Docker image.
