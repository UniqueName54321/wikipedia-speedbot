# wikipedia-speedbot

A turbocharged, GPU-ready Wikipedia speedrunning toolkit inspired by Green Code's Player1 — but upgraded for modern hardware, modular bots, and a blazing-fast SBERT HTTP server.

This project is built for experimentation, extensibility, and **speedrunning Wikipedia with actual style**.

---

## 🚀 Architecture Overview

### **`sbert_server.py`** — The SBERT Embedding Server

A FastAPI + Uvicorn microservice that:

* Loads SentenceTransformer **once** (GPU-accelerated if available)
* Exposes `/embed` for fast batched embeddings
* Prevents cold-boot delays during runs

> **Required if using Player1.5** (recommended).

---

### **`Player1_speedrunner.py`** — Minimal, Fast, & Deterministic

A simplified implementation of Green Code's Player1 algorithm:

* Uses SBERT anchor-text similarity
* Maintains a visited-set to prevent loops
* No beam search
* Fastest deterministic mode

---

### **`Player1p5_speedrunner.py`** — Smart Mode w/ Beam Search

A more strategic version of Player1:

* Uses the **SBERT HTTP server** exclusively
* Adds beam search via summary embeddings
* Selects links using multi-stage ranking
* Much smarter navigation without big performance hits

This is the flagship bot for high-quality runs.

---

### **`Player2_speedrunner.py`** — LLM Smartness

Uses a combination of Player1.5's SBERT model (for link grabbing) and a LLM (for knowing what link is the best one).


## 🧩 Requirements

See `requirements.txt` for full dependencies.

---

## 🏃‍♂️ Getting Started

### 1. Start the SBERT Server

```
python sbert_server.py --port 8000 --device auto
```

This loads SBERT once and keeps it hot.

### 2. Run a Player

```
python Player1p5_speedrunner.py --start "Andrej Karpathy" --target "Cat" --sbert-url http://127.0.0.1:8000
```

---

## 📜 License

MIT

Feel free to fork, hack, mod, or use in your own YouTube video. Let's break Wikipedia together.
