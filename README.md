# AFRICA-LMM

**AFRICA-LMM — Open Multimodal Foundation Model for African Documents**

AFRICA-LMM is an open-source multimodal AI platform designed to understand, retrieve, and answer questions about African documents.

The system supports document ingestion, PDF extraction, OCR, tables, semantic search, vector retrieval, reranking, RAG-based question answering, citations, conversation history, and a web interface.

---

## Architecture

```text
                         AFRICA-LMM
                              │
                ┌─────────────┴─────────────┐
                │                           │
          Next.js Frontend             FastAPI API
             :3000                       :8001
                │                           │
                └──────────────┬────────────┘
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
             PostgreSQL      Qdrant        ML/RAG
               :5432          :6333         Pipeline
                 │             │             │
                 └─────────────┴─────────────┘
```

### Main pipeline

```text
PDF / Image / Text
        │
        ▼
Document Engine
        │
        ├── PDF extraction
        ├── OCR
        ├── Table extraction
        └── Metadata
        │
        ▼
Chunking
        │
        ▼
Embeddings
        │
        ▼
Qdrant
        │
        ▼
Retrieval
        │
        ▼
Reranking
        │
        ▼
Context Builder
        │
        ▼
LLM / VLM
        │
        ▼
Answer + Citations
```

---

# Requirements

### System

* Ubuntu Linux
* Python 3.13+
* Docker
* Docker Compose
* Node.js / npm
* PostgreSQL 18+
* Git

### Recommended environment

```text
Python 3.13
Node.js 20+
Docker 29+
Docker Compose 5+
PostgreSQL 18+
```

---

# Project structure

```text
africa-lmm/
├── .github/
│   └── workflows/
├── configs/
│   ├── datasets/
│   ├── deployment/
│   ├── experiments/
│   └── models/
├── data/
│   └── test_documents/
├── frontend/
├── src/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   ├── ocr/
│   ├── rag/
│   ├── training/
│   └── vision/
├── tests/
│   ├── benchmarks/
│   ├── integration/
│   └── unit/
├── .env
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

# Configuration

Create a `.env` file at the project root.

Example:

```env
DATABASE_URL=postgresql+psycopg://africa_lmm:africa_lmm_dev_2026@localhost:5432/africa_lmm
```

The Docker Compose configuration overrides the database host for the API container and connects PostgreSQL through:

```text
host.docker.internal:5432
```

---

# 1. Start the backend with Docker

From the project root:

```bash
cd ~/Projects/africa-lmm
```

Start the backend stack:

```bash
docker compose up -d
```

This starts:

* AFRICA-LMM API
* Qdrant

Check the services:

```bash
docker compose ps
```

Expected services:

```text
africa-lmm-api
africa-lmm-qdrant
```

---

# 2. Check API health

```bash
curl http://localhost:8001/health
```

Expected response:

```json
{
  "status": "ok",
  "service": "africa-lmm",
  "version": "0.1.0"
}
```

API documentation:

```text
http://localhost:8001/docs
```

OpenAPI specification:

```text
http://localhost:8001/openapi.json
```

---

# 3. Check Qdrant

Qdrant is available at:

```text
http://localhost:6333
```

Check the service:

```bash
curl http://localhost:6333
```

---

# 4. PostgreSQL

PostgreSQL runs on the host system.

Check the service:

```bash
sudo systemctl status postgresql
```

Check PostgreSQL:

```bash
sudo -u postgres psql -c "SELECT version();"
```

The AFRICA-LMM database is:

```text
africa_lmm
```

Test the database:

```bash
sudo -u postgres psql -d africa_lmm -c "\dt"
```

---

# 5. View backend logs

```bash
docker compose logs -f api
```

Qdrant logs:

```bash
docker compose logs -f qdrant
```

All services:

```bash
docker compose logs -f
```

---

# 6. Stop the backend

Stop the services:

```bash
docker compose down
```

Start them again:

```bash
docker compose up -d
```

---

# 7. Rebuild the API image

Only rebuild when source code or Docker dependencies have changed.

```bash
docker build -t africa-lmm:dev .
```

Then recreate the API:

```bash
docker compose up -d --force-recreate api
```

There is **no need to rebuild** when only environment variables or Docker Compose configuration changes.

---

# 8. Frontend

The frontend is a Next.js application located in:

```text
frontend/
```

Enter the frontend directory:

```bash
cd ~/Projects/africa-lmm/frontend
```

Install dependencies:

```bash
npm install
```

Create:

```text
frontend/.env.local
```

with:

```env
NEXT_PUBLIC_API_URL=http://localhost:8001
```

This variable tells the frontend where the FastAPI backend is running.

---

# 9. Start the frontend

From:

```bash
cd ~/Projects/africa-lmm/frontend
```

run:

```bash
npm run dev
```

The frontend will be available at:

```text
http://localhost:3000
```

Open:

```text
http://localhost:3000/chat
```

---

# 10. Frontend production build

Build the Next.js application:

```bash
npm run build
```

Start the production server:

```bash
npm run start
```

---

# 11. Complete development startup

For normal development, use two terminals.

### Terminal 1 — Backend

```bash
cd ~/Projects/africa-lmm
docker compose up -d
```

Verify:

```bash
curl http://localhost:8001/health
```

### Terminal 2 — Frontend

```bash
cd ~/Projects/africa-lmm/frontend
npm run dev
```

Then open:

```text
http://localhost:3000/chat
```

---

# 12. Verify the API

Health:

```bash
curl -i http://localhost:8001/health
```

Documents:

```bash
curl -i http://localhost:8001/documents
```

Conversations:

```bash
curl -i http://localhost:8001/conversations
```

API documentation:

```text
http://localhost:8001/docs
```

---

# 13. Test document upload from the API

Example:

```bash
curl -X POST \
  http://localhost:8001/documents \
  -F "file=@data/test_documents/multimodal_agriculture.pdf"
```

The API should return information about the indexed document.

Then verify:

```bash
curl http://localhost:8001/documents
```

---

# 14. Ask a question through the API

Example:

```bash
curl -X POST \
  http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quelle est la production de cacao en 2024 ?",
    "top_k": 5,
    "document_ids": []
  }'
```

The response contains:

* the question
* the generated answer
* the conversation ID
* citations
* retrieved chunks

---

# 15. Development environment

Create the Python environment:

```bash
cd ~/Projects/africa-lmm
python3.13 -m venv env
```

Activate it:

```bash
source env/bin/activate
```

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

---

# 16. Run tests

Run the test suite:

```bash
python -m pytest
```

Run with concise output:

```bash
python -m pytest -q
```

---

# 17. Code quality

Run Ruff:

```bash
ruff check src tests
```

Run Black:

```bash
black --check src tests
```

Format the project:

```bash
black src tests
```

---

# 18. Python compilation check

Verify that the source code compiles:

```bash
python -m compileall -q src
```

---

# 19. Docker development workflow

After modifying Python source code:

```bash
docker build -t africa-lmm:dev .
docker compose up -d --force-recreate api
```

After modifying only `docker-compose.yml`:

```bash
docker compose up -d --force-recreate api
```

After modifying only frontend code:

```bash
cd frontend
npm run dev
```

No backend rebuild is required for frontend-only changes.

---

# 20. Useful Docker commands

List containers:

```bash
docker ps
```

List AFRICA-LMM containers:

```bash
docker compose ps
```

Inspect API logs:

```bash
docker compose logs --tail=100 api
```

Follow API logs:

```bash
docker compose logs -f api
```

Restart API:

```bash
docker compose restart api
```

Restart everything:

```bash
docker compose restart
```

Stop everything:

```bash
docker compose down
```

---

# 21. Useful Qdrant commands

Check Qdrant:

```bash
curl http://localhost:6333
```

List collections:

```bash
curl http://localhost:6333/collections
```

---

# 22. Git workflow

Check the repository:

```bash
git status
```

Review changes:

```bash
git diff
```

Add changes:

```bash
git add .
```

Commit:

```bash
git commit -m "feat: update AFRICA-LMM"
```

Push:

```bash
git push
```

---

# 23. Recommended startup sequence

For a fresh development session:

```bash
cd ~/Projects/africa-lmm
docker compose up -d
curl http://localhost:8001/health
```

Then:

```bash
cd ~/Projects/africa-lmm/frontend
npm run dev
```

Open:

```text
http://localhost:3000/chat
```

---

# 24. Service ports

| Service     | Port | Purpose             |
| ----------- | ---: | ------------------- |
| Next.js     | 3000 | Web interface       |
| FastAPI     | 8001 | REST API            |
| Qdrant      | 6333 | Vector database     |
| Qdrant gRPC | 6334 | Qdrant gRPC         |
| PostgreSQL  | 5432 | Relational database |

---

# 25. Current system status

The development stack provides:

* PDF document ingestion
* Native PDF text extraction
* OCR
* Table extraction
* Image processing
* Semantic embeddings
* Qdrant vector search
* Cross-encoder reranking
* RAG pipeline
* Conversation-aware retrieval
* Intent routing
* Context building
* LLM inference
* Answer citations
* PostgreSQL conversation persistence
* FastAPI REST API
* Next.js web interface
* Docker deployment
* MLflow evaluation
* Automated testing
* Code quality checks

---

# License

This project is open source. See [LICENSE](LICENSE) for details.

