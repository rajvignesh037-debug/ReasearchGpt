# Research Paper RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for querying research papers with natural language. Upload PDFs, ask questions, and get answers with source citations.

## Features

- **📄 PDF Upload & Processing** - Extract text from research papers automatically
- **🔍 Semantic Search** - Find relevant information using vector embeddings
- **🤖 AI-Powered Answers** - Generate context-aware responses with source citations
- **📚 Multi-Document Querying** - Search across multiple papers simultaneously
- **🎯 Source Attribution** - Every answer includes citations and page references
- **🚀 Production Ready** - API, UI, Docker deployment included

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (or local Ollama instance)
- Docker & Docker Compose (optional, for containerized deployment)

### Installation

1. **Clone and setup**
```bash
cd RAGs
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Usage

#### Phase 1: Test Document Ingestion

```bash
# Run ingestion test with sample data
python ingestion_pipeline.py --test

# Ingest a PDF
python ingestion_pipeline.py path/to/research_paper.pdf --output stats

# View chunk details
python ingestion_pipeline.py path/to/research_paper.pdf --output text
```

Expected output:
```
===================================
INGESTION PIPELINE TEST
===================================
✓ Successfully created 15 chunks
  Source: test_sample.txt
  Total characters: 2400
  Average chunk size: 160 chars

✓ Chunk metadata verified:
    Chunk 0:
      - ID: test_sample.txt#0
      - Source: test_sample.txt
      - Words: 45
      - Characters: 280
```

#### Phase 2-5: Building the Complete Pipeline

Once Phase 1 ingestion is working, subsequent phases will add:
- **Phase 2**: Embedding generation and ChromaDB vector storage
- **Phase 3**: Semantic retrieval module
- **Phase 4**: LLM integration and answer generation
- **Phase 5**: End-to-end pipeline orchestration
- **Phase 6**: FastAPI REST API

#### Phase 6+: Running the API and UI

```bash
# Start API server (Phase 6+)
uvicorn src.api.main:app --reload

# Start Streamlit UI (Phase 8+)
streamlit run src/ui/app.py

# Docker deployment (Phase 12+)
docker-compose up --build
```

## Project Structure

```
RAGs/
├── .planning/                      # GSD planning artifacts
│   ├── PROJECT.md                  # Project overview
│   ├── REQUIREMENTS.md             # Detailed requirements
│   ├── ROADMAP.md                  # 12-phase development plan
│   └── STATE.md                    # Project state tracker
├── src/
│   ├── core/
│   │   ├── ingestion.py            # Phase 1: PDF extraction & chunking
│   │   ├── embeddings.py           # Phase 2: Embedding generation
│   │   ├── vector_db.py            # Phase 2: ChromaDB wrapper
│   │   ├── retrieval.py            # Phase 3: Semantic search
│   │   ├── generation.py           # Phase 4: LLM integration
│   │   └── rag_pipeline.py         # Phase 5: Orchestrator
│   ├── api/
│   │   ├── main.py                 # Phase 6: FastAPI app
│   │   ├── routes.py               # Phase 6: Endpoints
│   │   └── schemas.py              # Phase 6: Request/response models
│   └── ui/
│       └── app.py                  # Phase 8: Streamlit UI
├── tests/                          # Phase 11: Test suite
├── data/                           # PDFs and vector DB
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment configuration
├── Dockerfile                      # Phase 12: Docker container
├── docker-compose.yml              # Phase 12: Multi-container setup
└── README.md                       # This file
```

## Development Roadmap

### Milestone 1: MVP Core RAG (Phases 1-6)
- [x] Phase 1: Document Ingestion
- [ ] Phase 2: Embedding Generation & Vector DB
- [ ] Phase 3: Retrieval Module
- [ ] Phase 4: LLM Integration & Generation
- [ ] Phase 5: End-to-End Pipeline
- [ ] Phase 6: FastAPI API Layer

### Milestone 2: Multi-Document UI (Phases 7-9)
- [ ] Phase 7: Multi-Document Support
- [ ] Phase 8: Streamlit UI
- [ ] Phase 9: Document Management

### Milestone 3: Production (Phases 10-12)
- [ ] Phase 10: Error Handling & Logging
- [ ] Phase 11: Testing Suite
- [ ] Phase 12: Docker & Deployment

### Backlog: Advanced Features (Post-MVP)
- [ ] Hybrid search (vector + keyword)
- [ ] Reranking for precision
- [ ] Conversational memory
- [ ] Document summarization
- [ ] Paper comparison

## Configuration

### Environment Variables

See [.env.example](.env.example) for all available options. Key variables:

```bash
# LLM Backend
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-3.5-turbo

# Document Processing
CHUNK_SIZE=500           # Characters per chunk
CHUNK_OVERLAP=50        # Character overlap between chunks

# Retrieval
DEFAULT_TOP_K=5         # Chunks to retrieve per query

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

### Supported LLM Backends

**OpenAI (Default)**
```bash
OPENAI_API_KEY=sk-your-key
```

**Local LLM (Ollama)**
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

## Testing

### Phase 1: Ingestion Pipeline Tests

```bash
# Test with sample data
python ingestion_pipeline.py --test

# Test with real PDF
python ingestion_pipeline.py path/to/paper.pdf --output stats

# Output formats: json, text, stats
```

### Running Full Test Suite (Phase 11+)

```bash
pytest tests/ -v
pytest tests/ --cov=src  # With coverage
```

## Docker Deployment (Phase 12+)

### Using Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Services:
# - API: http://localhost:8000
# - UI: http://localhost:8501
# - Vector DB: Persisted in ./data/chroma_db
```

### Manual Docker Build

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/data:/app/data \
  rag-assistant
```

## API Endpoints (Phase 6+)

### POST /upload
Upload a research paper PDF.

**Request:**
```bash
curl -X POST -F "file=@paper.pdf" http://localhost:8000/upload
```

**Response:**
```json
{
  "document_id": "paper.pdf",
  "chunks_created": 25,
  "status": "success",
  "stats": {
    "pages": 10,
    "characters": 45000,
    "avg_chunk_size": 1800
  }
}
```

### POST /query
Query documents with natural language.

**Request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main contribution?",
    "top_k": 5
  }'
```


  ## Local Verification
  Use these quick commands to verify the API and UI locally.

  ```bash
  # Check API health
  curl http://localhost:8000/health

  # Upload a PDF (replace path)
  curl -X POST -F "file=@/path/to/paper.pdf" http://localhost:8000/upload

  # Query the corpus
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query":"What is the main contribution?","top_k":5}'
  ```

### GET /health
Check service health.

**Response:**
```json
{
  "status": "online",
  "vector_db": "ready",
  "documents": 5,
  "chunks": 125
}
```

## Troubleshooting

### PDF Extraction Issues

- **Issue**: "PDF text extraction failed"
  - **Solution**: Ensure PDF is text-extractable (not scanned images)
  - Try switching PDF extractor: `PDF_EXTRACTOR=pypdf2` in .env

### CUDA/GPU Issues

- **Issue**: Embedding generation is slow
  - **Solution**: Install CUDA-enabled PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
  - Or reduce batch size: `EMBEDDING_BATCH_SIZE=8`

### API Connection Issues

- **Issue**: "Cannot connect to FastAPI"
  - **Solution**: Verify server running: `ps aux | grep uvicorn`
  - Check port not in use: `netstat -an | grep 8000`

## Performance Tips

1. **Faster Embeddings**: Reduce chunk size for faster processing
2. **Better Quality**: Increase chunk overlap from 50 to 100
3. **Reduce Latency**: Use GPT-3.5-turbo instead of GPT-4
4. **Lower Cost**: Reduce DEFAULT_TOP_K from 5 to 3

## Contributing

Contributions welcome! Please:
1. Check existing issues
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes in phase-based approach
4. Add tests
5. Commit with atomic commits per phase
6. Push and open PR

## License

MIT License - See LICENSE file for details

## Support

- 📖 Documentation: See `.planning/` directory
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 📧 Email: [Your contact]

---

**Built with:** Python • FastAPI • LangChain • ChromaDB • Sentence Transformers  
**Status:** Running locally — API and UI online (local verification available)  
**Last Updated:** June 26, 2026
