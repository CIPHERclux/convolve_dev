# Kairos — Research Prototype Documentation

> This document preserves the original research prototype documentation, Colab instructions, and detailed Qdrant architecture notes from the initial submission.

---

## Original System Overview

**Kairos** is a **multimodal mental health support system** that detects *hidden emotional patterns over time* by fusing **what a user says**, **how they say it**, and **how those signals evolve temporally**.

Unlike text-only chatbots, Kairos treats every interaction as a **persistent emotional moment**, storing it in a structured memory that enables *longitudinal reasoning, associative recall, and early risk detection*.

> **Core idea:** Mental health signals often hide in *patterns*, not single messages.  
> Kairos is built to remember and reflect those patterns back to the user.

---

## Moment Vector (Core Data Model)

Every interaction is represented as a **multimodal Moment Vector**:

| Channel     | Description                                  | Dimensionality |
|------------|----------------------------------------------|----------------|
| Semantic   | What is said (meaning, topics)               | 384-D          |
| Acoustic   | How it is said (prosody, voice markers)      | 32-D           |
| Trajectory | Emotional evolution across turns             | 160-D          |
| Sparse     | Explicit crisis / keyword signals             | Variable       |

All vectors are stored **together** as a single memory unit.

---

## Why Qdrant Is Essential (Not Optional)

Kairos **cannot exist** without Qdrant's architecture.  
This is not a tooling choice — it is an architectural dependency.

### 1. Native Multi-Vector Storage

Kairos stores **multiple orthogonal vectors per memory point** (semantic, acoustic, trajectory).

Qdrant is one of the *very few* vector databases that supports this natively.

> Example query:  
> *"Find moments where the user sounded anxious, regardless of topic."*

This is **impossible** in single-vector systems.

---

### 2. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

Kairos runs **parallel searches** across:
- semantic similarity
- acoustic similarity
- trajectory similarity
- sparse crisis signals

Qdrant's **Prefetch API + native RRF** fuses these rankings **server-side**, in one round-trip.

---

### 3. Binary Quantization (Edge-Ready Memory)

Semantic vectors are binary-quantized:

- **1536 bytes → ~48 bytes per vector**
- ~32× memory reduction
- Enables **local / edge deployment**
- Preserves high recall via two-stage search

This is critical for **privacy-sensitive mental health data**.

---

### 4. Graph-RAG via Prefetch

Kairos maintains an **entity–emotion graph** that expands memory queries: 
"best friend"
↓
"birthday" → "guilt"

Qdrant executes these expansions **in parallel** using Prefetch and merges them with RRF, enabling **associative recall**.

---

## Three-Layer Memory Design

Kairos uses a **cognitive memory model**, not a flat database:

### Layer 1 — User Profile (Facts)
- Name, relationships, stable attributes
- Deterministic, O(1) access

### Layer 2 — Episodic Memory (Qdrant)
- Multimodal Moment Vectors
- Searchable across sessions
- Supports similarity + filtering

### Layer 3 — Entity–Emotion Graph
- Associative links between entities and emotions
- Drives Graph-RAG expansion
- Continuously updated from memory usage

This bidirectional interaction is a **core innovation**.

---

## How to Run (Google Colab) — Original Prototype

> **Note:** The production version now uses FastAPI + React. See the main [README](../README.md) for current setup instructions.

Kairos was originally designed to run inside **Google Colab** with minimal setup.

### Prerequisites
- A Google account (for Colab access)
- The project ZIP file

### Step-by-Step Instructions

1. **Download the Project ZIP**
   - Download from the Google Drive link provided.
   - Do **not** extract the ZIP locally.

2. **Open Google Colab**
   - Visit: https://colab.research.google.com
   - Click **New Notebook**

3. **Copy the Starter Cell**
   - Locate `colab_starter_cell.py` in the repository.
   - Copy the entire contents of the starter cell.

4. **Paste and Run**
   - Paste into the **first cell** of the Colab notebook.
   - Run the cell.

5. **Upload the ZIP When Prompted**
   - Upload the ZIP file when prompted.
   - The system will automatically extract dependencies and initialize Kairos.

6. **Start Interacting**
   - Once setup completes, interact using text, audio, or video.

### Notes
- No local environment setup is required.
- Qdrant runs in **local embedded mode** inside Colab.
- All processing happens within the Colab session.
- Use Colab T4 for better experience.

---

## Prototype Limitations

This repository represents a **research prototype**:

- Some memories may be missed or under-weighted
- Acoustic features depend on input quality
- Cold-start sessions lack trajectory history
- Not a clinical diagnostic tool

These are **known and documented limitations**, not design oversights.

---

## Ethics & Safety

- Kairos is a **support system**, not therapy
- Explicit crisis pathways with emergency guidance
- Conservative language (no diagnoses or certainty)
- Designed for **privacy-first, local deployment**
