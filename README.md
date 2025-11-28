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

---

### Antivirus / Security Note Regarding Player3

Player3 makes a lot of automated HTTP requests while it runs, for example:

- To **Wikipedia** – to fetch pages and summaries
- To **whatever LLM endpoint you configure** – e.g. OpenRouter, or your own local OpenAI-compatible server
- (Optionally) to a local **SBERT embedding server** if you pass `--sbert-url`

Because of this “bot-like” network behaviour, some antivirus products (e.g. Bitdefender) may occasionally flag the script with a **generic bot / automation heuristic**. This does *not* mean the script is malware by itself – it just means the behaviour matches a pattern they sometimes associate with bots or scrapers.

A few important points:

- The code is **fully open source** – you can audit exactly what it does.
- It does **not** modify the registry, install drivers, or touch system settings.
- It only talks to:
  - Wikipedia
  - The LLM endpoint you explicitly configure
  - An optional local SBERT server (if you run one)

If your antivirus flags it:

- Treat it like any other dev tool:  
  - Review the code yourself and make sure you’re comfortable with it.
  - Run it in a virtual environment / throwaway VM if you prefer.
- Only if you’re satisfied it’s safe, you may choose to add an exception for:
  - Your Python interpreter path
  - This project’s folder

**Do not fully disable your antivirus.** If you’re unsure or the detection looks suspicious, lean on the side of caution.

---


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
