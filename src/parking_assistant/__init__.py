"""Parking garage RAG assistant."""

import os

# gRPC (used by Milvus Lite) prints a harmless fork() notice to the console unless set before it loads.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
