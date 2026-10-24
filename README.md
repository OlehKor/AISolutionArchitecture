# Parking Assistant

![CI](https://github.com/OlehKor/AISolutionArchitecture/actions/workflows/ci.yml/badge.svg)

RAG chatbot for the fictional SkyPark Central parking garage. It answers questions about the garage, collects the details for a reservation, protects confidential data and ships with an evaluation of retrieval quality, latency and data protection.

## How it works

```
user message
   |
   v
screen_input (guard rails: injection / sensitive-request filter + PII masking)
   |-- refused --> finalize
   v
classify (LLM: info | reserve | cancel | other)
   |-- info/other --> retrieve (Milvus: public chunks only + SQLite live data) --> answer (LLM)
   |-- reserve    --> reserve (LLM slot extraction + validation + SQLite booking)
   |-- cancel     --> cancel
   v
finalize (guard rails: mask PII in the reply) --> user
```

The flow is a LangGraph graph ([src/parking_assistant/assistant.py](src/parking_assistant/assistant.py)); each chat session keeps its history and reservation draft in the graph state.

- **Static data** (general information, parking details, location, booking process, FAQ) lives in `data/static/` and is stored as embeddings in Milvus (Milvus Lite, a local file). The sub-directory `public/` or `internal/` becomes the `visibility` metadata of each chunk; every search is filtered to `visibility == "public"`, so internal documents are never retrieved.
- **Dynamic data** (opening hours, prices, free spaces, reservations) lives in SQLite (`data/dynamic/`) and is injected into the prompt when the question needs it.
- **Reservation** collects first name, last name, car number and period step by step. Values are validated (letters-only names, plate format, no past or over-long periods), summarised, and stored only after the user confirms.
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

## Usage

```bash
parking-assistant ingest                  # (re)build the vector store (also runs automatically when missing)
parking-assistant chat                    # talk to the assistant
parking-assistant evaluate                # retrieval accuracy, latency, data protection -> reports/evaluation_report.md
parking-assistant evaluate --end-to-end   # additionally time full LLM answers
pytest                                    # unit tests (no API key needed)
```

Settings can be overridden with `PARKING_*` environment variables, see [config.py](src/parking_assistant/config.py).

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
    assistant.py                 LangGraph flow and chat entry point
    conversation.py              LLM prompts: routing, slot extraction, answering
    reservation.py               pure reservation logic and messages
    knowledge.py                 chunking, Milvus ingestion and public-only search
    database.py, live_data.py    SQLite access and live data formatting
    guardrails.py                input filtering and PII masking
    evaluation.py                metrics, runners, report
    composition.py, __main__.py  wiring and CLI
tests/                           pytest suite
```
