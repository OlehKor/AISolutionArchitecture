# Load test report

Generated: 2026-10-25 12:37

Parameters: 24 chat sessions at concurrency 8 (stub LLM), 100 MCP calls at each of [1, 4, 16] parallel clients. Telegram is replaced by a stand-in that always answers, and the chat language model by a rule-based stub unless the parameters say 'real LLM'; LangGraph, SQLite, the vector store, the guard rails and the MCP server are real.

## Chatbot in interactive dialogue mode

- Jobs: 24, errors: 0, duration: 2.39 s, throughput: 10.0 jobs/s
- 24 reservation requests are waiting for the administrator

| Operation | Requests | Mean ms | p50 ms | p95 ms | Max ms |
|---|---|---|---|---|---|
| Chat: information question (guard rails + RAG + live data) | 48 | 113.4 | 110.6 | 141.7 | 149.2 |
| Chat: reservation step (slot filling) | 120 | 85.7 | 80.1 | 120.9 | 137.7 |
| Chat: confirmation (booking + workflow start + request to administrator) | 24 | 122.9 | 124.4 | 142.0 | 145.2 |

## Administrator confirmation (workflow resume)

- Jobs: 24, errors: 0, duration: 0.92 s, throughput: 26.2 jobs/s
- 0 requests still pending
- 20 confirmed, 0 not yet archived
- 20 lines in the reservations file (expected 20)

| Operation | Requests | Mean ms | p50 ms | p95 ms | Max ms |
|---|---|---|---|---|---|
| Administrator decision (resume workflow + database + MCP write) | 24 | 271.7 | 302.8 | 534.5 | 545.8 |

## MCP server recording and storage

- Jobs: 300, errors: 0, duration: 11.20 s, throughput: 26.8 jobs/s
- 1 parallel clients: 38.3 calls/s
- 4 parallel clients: 28.8 calls/s
- 16 parallel clients: 19.5 calls/s
- 300 lines stored for 300 distinct reservations
- 0 duplicate lines after replaying 20 requests

| Operation | Requests | Mean ms | p50 ms | p95 ms | Max ms |
|---|---|---|---|---|---|
| MCP tool call (authenticated HTTP + validated write), 1 parallel clients | 100 | 26.1 | 26.1 | 27.8 | 34.9 |
| MCP tool call (authenticated HTTP + validated write), 4 parallel clients | 100 | 138.7 | 125.6 | 236.9 | 339.7 |
| MCP tool call (authenticated HTTP + validated write), 16 parallel clients | 100 | 796.2 | 837.0 | 955.9 | 961.3 |
