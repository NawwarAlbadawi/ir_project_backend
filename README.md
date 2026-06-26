# 🔍 Information Retrieval System (Backend)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![SOA](https://img.shields.io/badge/Architecture-Microservices-orange)
![Machine Learning](https://img.shields.io/badge/AI-BERT%20%7C%20Word2Vec%20%7C%20LDA-ff69b4)

An advanced, Service-Oriented Architecture (SOA) Information Retrieval System backend. Built with Python and FastAPI, this system is capable of processing natural language queries, running semantic and keyword-based searches across massive datasets (>200k documents), and calculating real-time accuracy metrics (MAP, nDCG).

This project was built to support the 2026 Information Retrieval University Course.

---

## 🌟 Features

*   **Multi-Dataset Support:** Pre-configured for `BEIR/Quora` and `MS MARCO`.
*   **Microservice Architecture (SOA):** Completely decoupled into 7 independent services communicating via REST APIs.
*   **Hybrid Search:** Supports keyword matching (TF-IDF, BM25) and Dense Semantic Search (Word2Vec, BERT).
*   **Reciprocal Rank Fusion (RRF):** Parallel hybrid retrieval combining sparse and dense scoring.
*   **Query Refinement:** Automatic spelling correction and synonym expansion to improve recall.
*   **Topic Detection (Extra Feature):** Real-time topic assignment for search results using Latent Dirichlet Allocation (LDA).
*   **Automated Evaluation:** Built-in benchmarking endpoints calculating MAP, Recall, Precision@10, and nDCG.

---

## 🏗️ System Architecture (Microservices)

The backend is composed of the following services:

| Port | Service Name | Responsibilities |
| :--- | :--- | :--- |
| **8000** | `API Gateway` | Central entry point, request orchestration, and merging. |
| **8001** | `Preprocessing` | Text normalization, tokenization, stemming, lemmatization. |
| **8002** | `Indexing` | Builds and loads Inverted Indices and FAISS Vector indices. |
| **8003** | `Retrieval` | Executes Cosine Similarity and Sparse matching algorithms. |
| **8004** | `Evaluation` | Calculates ranking accuracy metrics (MAP, nDCG, MRR) against Qrels. |
| **8005** | `Refinement` | Query expansion, typo correction, and semantic suggestions. |
| **8006** | `Topic Detection`| Assigns LDA topics to retrieved documents in real-time. |

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
*   Windows OS (PowerShell)
*   Python 3.10+ installed
*   Pre-calculated `IR_Project_Indexes` folder placed in the `/data` directory.

### 2. Running the System
We have provided a fully automated startup script that installs dependencies, checks ports, launches all 7 microservices in the correct order, and automatically loads the Quora/MS MARCO datasets into memory.

Open PowerShell in the project root and run:
```powershell
.\start_local.ps1
```

Once the startup script confirms all services are `OK`, the API Gateway will be accessible at:
*   **Base URL:** `http://localhost:8000`
*   **Interactive API Docs (Swagger):** `http://localhost:8000/docs`

---

## 📱 Connecting the Frontend
If you are running the companion **Flutter Mobile App** on a physical device, remember to update the `kApiBaseUrl` in your Flutter code to match your computer's local Wi-Fi IP address (e.g., `http://192.168.1.X:8000`) instead of `localhost`.
