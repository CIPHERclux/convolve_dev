# Kairos — Multimodal Mental Health Support System

A production-grade mental health support platform that analyzes **what users say** and **how they say it**, combining natural language processing with acoustic biomarker analysis to detect hidden emotional patterns over time.

Built with **FastAPI**, **React**, and **Qdrant** (vector database), featuring real-time crisis detection, cross-modal masking detection, and persistent emotional memory.

[![CI](https://github.com/YOUR_USERNAME/kairos/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/kairos/actions)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                         │
│         Chat UI · Voice Recording · State Visualizer        │
└────────────────────────┬────────────────────────────────────┘
                         │ REST API
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend                           │
│                                                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │  Routes   │  │ Orchestrator │  │   Safety Service   │    │
│  │ (DI via   │──│  (async      │──│  (deterministic    │    │
│  │ Depends())│  │  processing) │  │   crisis detection)│    │
│  └──────────┘  └──────┬───────┘  └────────────────────┘    │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │              Feature Extraction Engine                │  │
│  │  Acoustic (librosa) · Linguistic (VADER/regex) ·     │  │
│  │  Special Signals (laughter/crying/sighs)             │  │
│  └──────────────────────────────────────────────────────┘  │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │                  Memory Layer                         │  │
│  │  Qdrant (384-dim vectors) · User Profiles (JSON) ·   │  │
│  │  Biomarker Tracker · Baseline Manager (Welford's)    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

### Multimodal Analysis
- **28-dimensional biomarker vector** extracted per turn (acoustic, linguistic, visual, special signals)
- Acoustic features: jitter, shimmer, F0 variance, speech rate, pause rate, TEO, HNR
- Linguistic features: absolutist index, I-ratio, lexical density, rumination score, sentiment
- Special signals: laughter, crying, sighing, vocal strain detection

### Cross-Modal Masking Detection
Identifies contradictions between *what users say* and *how they sound*:
- Voice tremor + positive text → hidden anxiety
- Flat pitch + high volume → suppressed emotions
- Laughter + crying co-occurrence → complex emotional state

### Persistent Emotional Memory
- Semantic search over past interactions (Qdrant, cosine similarity)
- Per-user baseline calibration via **Welford's online algorithm**
- Delta tracking: significant biomarker changes between turns
- Distress trend analysis over rolling windows

### Safety-First Design
- Deterministic regex-based crisis detection (critical/high/moderate/none)
- Biomarker-driven risk escalation (crying > 0.5, sentiment < -0.7)
- Runs **before** LLM calls — never delegates safety to a language model

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 19 + Vite | Chat UI with voice recording |
| Backend | FastAPI + Uvicorn | Async API server |
| ML/NLP | sentence-transformers, librosa, VADER | Feature extraction |
| Memory | Qdrant | Vector similarity search |
| LLM | OpenAI / Groq (configurable) | Response generation |
| CI/CD | GitHub Actions + ruff + pytest | Automated linting & testing |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **FastAPI Dependency Injection** over global state | Testability, thread safety, explicit dependencies |
| **`asyncio.to_thread()`** for audio processing | Prevents event loop starvation from CPU-bound librosa/DSP work |
| **Deterministic safety checks** (regex, not LLM) | Crisis detection must be predictable and auditable |
| **Single 384-dim vector** (not multi-vector) | Simplicity over marginal recall gains; acoustic features stored as payload metadata |
| **Welford's algorithm** for baselines | O(1) memory, numerically stable, no stored history needed |
| **No Celery/Redis** | Lightweight enough for local dev; `to_thread()` sufficient for current scale |

---

## Local Development

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (optional, for Qdrant)

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your LLM API key

# Run
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Qdrant (Optional)

```bash
docker compose up -d
```

The backend gracefully degrades without Qdrant — memory features are disabled but chat still works.

---

## Testing

```bash
cd backend
pip install pytest ruff

# Run tests (74 tests across safety, linguistics, profile extraction)
python -m pytest tests/ -v

# Lint
ruff check app/ tests/
```

### Test Coverage

| Module | Tests | What's Covered |
|--------|-------|---------------|
| `SafetyService` | 28 | Crisis patterns (critical/high/moderate), biomarker escalation, edge cases |
| `LinguisticEngine` | 14 | All 8 features, output shape, value bounds |
| `UserProfile` | 32 | Name/age/location/diagnosis extraction, deduplication, persistence |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/session/start` | Start a new chat session |
| `DELETE` | `/api/v1/session/{id}` | End a session |
| `POST` | `/api/v1/chat/text` | Send text message |
| `POST` | `/api/v1/chat/audio` | Send audio file (with transcription) |
| `GET` | `/api/v1/profile/{user_id}` | Get user profile facts |
| `GET` | `/health` | Health check + Qdrant status |

---

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── dependencies.py      # FastAPI DI providers
│   │   └── routes/              # chat, session, profile, health
│   ├── extraction/
│   │   ├── acoustic_engine.py   # librosa DSP features
│   │   ├── linguistic_engine.py # VADER + regex features
│   │   └── feature_engine.py    # Orchestrates extraction
│   ├── memory/
│   │   ├── memory_controller.py # Qdrant interaction layer
│   │   ├── biomarker_tracker.py # Rolling-window analysis
│   │   ├── baseline_manager.py  # Welford's algorithm
│   │   └── user_profile.py      # Deterministic fact storage
│   ├── services/
│   │   ├── orchestrator.py      # Main coordination service
│   │   ├── llm_service.py       # LLM with tool calling
│   │   ├── safety_service.py    # Crisis detection
│   │   └── session_manager.py   # Session lifecycle
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── model_registry.py    # Thread-safe ML model singleton
│   └── config.py                # pydantic-settings configuration
├── tests/                       # pytest suite
└── pyproject.toml               # ruff + pytest config

frontend/
├── src/
│   ├── App.jsx                  # Main chat interface
│   ├── components/ChatBubble.jsx
│   └── services/api.js          # API client
└── package.json
```

---

## Research Background

This project evolved from a research prototype built for a Qdrant hackathon. The original prototype documentation (Colab instructions, multi-vector architecture, Graph-RAG design) is preserved in [`docs/research-prototype.md`](docs/research-prototype.md).

---

## Ethics & Safety

- Kairos is a **support system**, not therapy or a diagnostic tool
- Crisis detection uses deterministic rules, not probabilistic LLM outputs
- Conservative language — no diagnoses, no certainty claims
- Designed for **privacy-first, local deployment**
