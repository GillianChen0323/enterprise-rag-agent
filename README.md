# Enterprise RAG Agent Platform

A RAG agent project with retrieval, reranking, Redis-backed query caching, and evaluation modules.

## Project structure

```text
enterprise_rag_agent/
├── agent/
│   ├── core.py           # Agent orchestration
│   └── factory.py        # Dependency assembly
├── retrieval/
│   ├── pipeline.py       # Asynchronous retrieval pipeline
│   └── reranker.py       # Two-stage reranking
├── ingestion/
│   ├── chunker.py        # Hierarchical chunking
│   ├── embedder.py       # Embeddings and HyDE expansion
│   └── ingest.py         # Document ingestion
├── cache/
│   └── semantic_cache.py # Redis exact-match query cache
├── llm/
│   └── client.py         # Pooled and streaming LLM client
├── config/
│   └── use_cases.py      # Three business configurations
├── api/
│   └── main.py          # FastAPI entry point
├── evaluation/
│   └── evaluator.py     # Retrieval and answer evaluation
├── tests/
│   └── load_test.py     # Locust load testing
└── requirements.txt
```

## Starting the API

Install dependencies and configure the API credentials and backing services used by the project before starting the server:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Implementation notes

- The cache currently uses normalized exact query matching, despite the historical `SemanticCache` name.
- The sparse/BM25 retrieval branch is a placeholder and must be connected to a backend for real sparse retrieval.
- The chunking comparison, latency benchmark, and threshold tuning scripts include simulations. Their outputs are not production performance measurements.
- Load-test latency and concurrency figures are targets to validate in a configured environment.
- Chinese prompts, sample data, and runtime messages are retained intentionally; translating documentation does not change application behavior.
