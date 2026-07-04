RAG_production/
│
├── configs/
│   └── config files for project parameters
│
├── data/
│   ├── pdf/
│   │   └── local source PDFs
│   │
│   ├── processed/
│   │   └── generated processed documents/chunks/metadata
│   │
│   ├── vector_store/
│   │   └── local vector database files if running locally
│   │
│   ├── eval/
│   │   └── golden questions, expected sources, evaluation datasets
│   │
│   └── cache/
│       └── local cache files if needed
│
├── docs/
│   └── architecture notes, interview notes, diagrams, decisions
│
├── logs/
│   └── generated runtime logs
│
├── rag_app/
│   ├── core/
│   │   └── shared config, schemas, constants, common types
│   │
│   ├── ingestion/
│   │   └── document loading using LangChain loaders
│   │
│   ├── chunking/
│   │   └── parent-child chunking and metadata preservation
│   │
│   ├── embeddings/
│   │   └── embedding model setup and embedding generation
│   │
│   ├── storage/
│   │   └── Qdrant vector database integration and document storage
│   │
│   ├── retrieval/
│   │   └── similarity search, metadata filtering, dynamic top-k
│   │
│   ├── reranking/
│   │   └── cross-encoder reranking of retrieved candidates
│   │
│   ├── generation/
│   │   └── prompt templates, LLM calls, grounded answer generation
│   │
│   ├── security/
│   │   └── prompt-injection checks, input validation, safety rules
│   │
│   ├── evaluation/
│   │   └── retrieval quality, answer faithfulness, citation evaluation
│   │
│   ├── observability/
│   │   └── logging, metrics, latency tracking, debug traces
│   │
│   ├── cache/
│   │   └── Redis/local cache helpers later
│   │
│   └── utils/
│       └── small shared helper functions
│
├── scripts/
│   └── command-line scripts for ingestion, indexing, evaluation
│
├── tests/
│   ├── unit/
│   │   └── tests for individual functions/modules
│   │
│   └── integration/
│       └── tests for full pipeline behavior
│
├── main.py
│   └── simple project entry point/demo runner
│
├── README.md
│   └── project explanation, setup, architecture, usage, interview story
│
├── pyproject.toml
│   └── project metadata and dependencies
│
├── requirement.txt
│   └── pip dependency list
│
├── uv.lock
│   └── uv dependency lock file
│
├── .python-version
│   └── selected Python version
│
└── .gitignore
    └── ignored files/folders like .env, .venv, PDFs, logs, vector DB

    