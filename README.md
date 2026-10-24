# Parking Assistant

![CI](https://github.com/OlehKor/AISolutionArchitecture/actions/workflows/ci.yml/badge.svg)

RAG chatbot for the fictional SkyPark Central parking garage. It answers questions about the garage, collects the details for a reservation, protects confidential data and ships with an evaluation of retrieval quality, latency and data protection. A second agent asks a human administrator, over Telegram, to approve each reservation (human-in-the-loop).

## How it works

```
user message
   |
   v
screen_input (guard rails: injection / sensitive-request filter + PII masking)
   |-- refused --> finalize
   v
classify (LLM: info | reserve | cancel | status | other)
   |-- info/other --> retrieve (Milvus: public chunks only + SQLite live data) --> answer (LLM)
   |-- reserve    --> reserve (LLM slot extraction + validation + pending booking in SQLite)
   |                    '-- booked --> escalate (hand over to the administrator agent)
   |-- cancel     --> cancel
   |-- status     --> report_status (state of this customer's requests)
   v
finalize (prepend new administrator decisions, mask PII in the reply) --> user
```

The flow is a LangGraph graph ([src/parking_assistant/assistant.py](src/parking_assistant/assistant.py)); each chat session keeps its history and reservation draft in the graph state.

- **Static data** (general information, parking details, location, booking process, FAQ) lives in `data/static/` and is stored as embeddings in Milvus (Milvus Lite, a local file). The sub-directory `public/` or `internal/` becomes the `visibility` metadata of each chunk; every search is filtered to `visibility == "public"`, so internal documents are never retrieved.
- **Dynamic data** (opening hours, prices, free spaces, reservations) lives in SQLite (`data/dynamic/`) and is injected into the prompt when the question needs it.
- **Reservation** collects first name, last name, car number and period step by step. Values are validated (letters-only names, plate format, no past or over-long periods), summarised, and stored only after the user confirms.
- **Human-in-the-loop** ([administrator.py](src/parking_assistant/administrator.py)): after the customer confirms the summary the reservation is stored as `pending` and handed to the administrator agent, which writes a short LLM note on the request and sends the details to the administrator on Telegram. The administrator replies in free text ("ok", "no, closed that day"); the agent interprets the reply with the LLM (approve / reject / unclear, asking again when unclear), updates the reservation in SQLite and acknowledges. The customer is told the decision on their next message and can ask for the status at any time. Rejected requests release their space.
- **Guard rails** ([guardrails.py](src/parking_assistant/guardrails.py)):
  - input: regex policies refuse prompt injection and requests for staff contacts, access codes, other customers' data and internal documents;
  - input and output: Presidio (spaCy NER and pattern recognizers) masks card numbers, IBANs, SSNs, e-mails, phone numbers, IPs and credentials; the garage's public contacts are allow-listed.

## Setup

Requires Python 3.11-3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # set OPENAI_API_KEY (or the Azure OpenAI variables)
```

Embeddings run locally with FastEmbed, so only the chat model needs an API key.

### Administrator bot (Telegram)

1. In Telegram, talk to `@BotFather`, run `/newbot` and copy the token.
2. Put the token in `.env` as `PARKING_TELEGRAM_BOT_TOKEN` (never commit it).
3. Run `parking-assistant admin`, then send any message to your bot. The log shows `Ignored a message from unauthorized chat <id>`; put that id in `.env` as `PARKING_TELEGRAM_ADMIN_CHAT_ID` and restart the command.

Only the configured chat can decide; messages from other chats are ignored. Without Telegram settings the chat still works, but reservations stay `pending`.

## Usage

```bash
parking-assistant ingest                  # (re)build the vector store (also runs automatically when missing)
parking-assistant chat                    # talk to the assistant
parking-assistant admin                   # run the administrator agent (keep it running in a second terminal)
parking-assistant evaluate                # retrieval accuracy, latency, data protection -> reports/evaluation_report.md
parking-assistant evaluate --end-to-end   # additionally time full LLM answers
pytest                                    # unit tests (no API key needed)
```

Settings can be overridden with `PARKING_*` environment variables, see [config.py](src/parking_assistant/config.py).

Typical demo: start `parking-assistant admin` in one terminal and `parking-assistant chat` in another, book a space in the chat, answer the request in Telegram, then write to the chat again (or ask "is my booking confirmed?").

## Evaluation

`parking-assistant evaluate` uses the datasets in `data/eval/`:

- `retrieval.json`: questions with the chunk ids that should be retrieved. Reports Recall@K, Precision@K, hit rate@K and MRR for K = 1, 3, 5, plus retrieval latency (mean, p50, p95, max). Duplicate chunk ids are counted once.
- `guardrails.json`: attacks that must be refused, legitimate messages that must pass, personal data that must be masked, and probes that must not retrieve confidential text.

The generated report is written to `reports/evaluation_report.md`.

## Structure

```
data/static/{public,internal}/   source documents for the vector store
data/dynamic/                    SQLite schema and seed data
data/eval/                       evaluation datasets
src/parking_assistant/
    assistant.py                 LangGraph flow and chat entry point (first agent)
    administrator.py             administrator agent: request note, reply interpretation, polling loop
    approval.py                  decision model, request targeting, administrator/customer messages
    telegram.py                  Telegram Bot API channel
    conversation.py              LLM prompts: routing, slot extraction, answering
    reservation.py               pure reservation logic and messages
    knowledge.py                 chunking, Milvus ingestion and public-only search
    database.py, live_data.py    SQLite access (reservations and approval status), live data formatting
    guardrails.py                input filtering and PII masking
    evaluation.py                metrics, runners, report
    composition.py, __main__.py  wiring and CLI
tests/                           pytest suite
```
