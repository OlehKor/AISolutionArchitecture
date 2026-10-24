"""Parking garage RAG assistant."""

import os

# gRPC (Milvus Lite) prints harmless fork() and keepalive errors over the chat prompt unless set before it loads.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
